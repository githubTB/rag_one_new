"""
app.py — 重构版 FastAPI

核心改进：
1. /api/ingest  — 上传文件后完整走解析→向量化→入库流程，带去重
2. /api/query   — 检索 + LLM 问答，返回结构化 citations（文件名、页码、标题路径）
3. /api/search  — 纯检索，返回溯源信息（不走 LLM）
4. /api/files   — 查看已入库文件列表
5. 错误处理和状态反馈更完善
"""

import os
import shutil
import asyncio
import traceback
import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── 内部模块 ──────────────────────────────────────────────────
from embedder import get_embedder
from parser import parse_file, file_sha256, Chunk, HAS_PADDLE, HAS_TESSERACT

HAS_PADDLE_OR_TESS = HAS_PADDLE or HAS_TESSERACT
from vectorstore import (
    init_db, is_file_indexed, register_file, list_indexed_files,
    delete_file_from_index, connect_milvus, get_or_create_collection,
    insert_chunks, multi_stage_search, delete_by_file_hash
)

# ── 配置 ──────────────────────────────────────────────────────
UPLOAD_DIR   = "uploaded_files"
DB_PATH      = "rag_meta.db"
MILVUS_HOST  = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT  = os.getenv("MILVUS_PORT", "19530")
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL    = os.getenv("LLM_MODEL", "qwen3.5:9b")
COLLECTION   = "rag_docs"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── 应用初始化 ────────────────────────────────────────────────
app = FastAPI(title="RAG Knowledge Base API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# 全局状态（启动时初始化）
db_conn   = None
milvus_col = None
embedder  = None


@app.on_event("startup")
async def startup():
    global db_conn, milvus_col, embedder
    print("🚀 启动 RAG 服务...")

    db_conn = init_db(DB_PATH)
    print("✅ SQLite 元数据库已初始化")

    embedder = get_embedder()
    print("✅ Embedding 模型已加载")

    try:
        connect_milvus(MILVUS_HOST, MILVUS_PORT)
        dim = _get_embed_dim()
        milvus_col = get_or_create_collection(COLLECTION, dim=dim)
    except Exception as e:
        print(f"⚠️  Milvus 连接失败: {e}，将在第一次请求时重试")


# ── 响应模型 ──────────────────────────────────────────────────

class Citation(BaseModel):
    id: int
    file_name: str
    page: int
    block_type: str
    heading_path: str
    text_snippet: str   # 原文片段（前200字）
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    llm_available: bool


class SearchResponse(BaseModel):
    query: str
    results: list[Citation]


# ── 工具函数 ──────────────────────────────────────────────────

def _get_embed_dim() -> int:
    """获取 embedding 维度，兼容不同版本的 sentence-transformers"""
    try:
        # sentence-transformers >= 2.x
        return embedder.get_sentence_embedding_dimension()
    except AttributeError:
        # 降级：encode 一个样本取维度
        return len(embedder.encode("test", normalize_embeddings=True))


def _ensure_milvus():
    global milvus_col
    if milvus_col is None:
        connect_milvus(MILVUS_HOST, MILVUS_PORT)
        milvus_col = get_or_create_collection(COLLECTION, dim=_get_embed_dim())


def _hits_to_citations(hits: list[dict]) -> list[Citation]:
    return [
        Citation(
            id=i + 1,
            file_name=h["file_name"],
            page=h["page"],
            block_type=h["block_type"],
            heading_path=h["heading_path"],
            text_snippet=h["text"][:300],
            score=h.get("rerank_score", h["score"]),
        )
        for i, h in enumerate(hits)
    ]


def _build_rag_prompt(query: str, hits: list[dict]) -> str:
    """
    构建带来源标注的 RAG Prompt
    LLM 被要求在答案中用 [来源N] 标注引用
    """
    context_parts = []
    for i, h in enumerate(hits):
        src = f"[来源{i+1}] 文件：{h['file_name']} 第{h['page']}页"
        if h["heading_path"]:
            src += f" / {h['heading_path']}"
        context_parts.append(f"{src}\n{h['text']}")

    context = "\n\n---\n\n".join(context_parts)

    return f"""你是一个知识库问答助手。请基于以下参考资料回答问题。
在答案中，每当引用某个来源的内容时，请在句子末尾用 [来源N] 标注（N为来源编号）。
如果参考资料中没有相关信息，请明确说明"参考资料中未找到相关信息"，不要编造内容。

参考资料：
{context}

问题：{query}

回答（请包含来源标注）："""


async def _call_llm(prompt: str) -> str:
    """调用本地 Ollama LLM"""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": LLM_MODEL, "prompt": prompt, "stream": False}
            )
            return resp.json().get("response", "")
    except Exception as e:
        raise RuntimeError(f"LLM 调用失败: {e}")


# ── API 路由 ──────────────────────────────────────────────────

@app.post("/api/ingest")
async def ingest_files(
    files: list[UploadFile] = File(...),
    milvus_host: str = Form(default=None),
    milvus_port: str = Form(default=None),
):
    """
    上传并解析文件，入库到 Milvus
    - 自动按文件哈希去重（重复文件跳过）
    - 支持前端传入 milvus_host / milvus_port 覆盖默认配置
    - 返回每个文件的处理结果
    """
    global milvus_col

    # 如果前端传了 Milvus 配置，用前端的覆盖环境变量
    host = milvus_host or MILVUS_HOST
    port = milvus_port or MILVUS_PORT

    # 重新连接（配置变了才重连）
    try:
        connect_milvus(host, port)
        milvus_col = get_or_create_collection(COLLECTION, dim=_get_embed_dim())
    except Exception as e:
        raise HTTPException(status_code=503,
                            detail=f"Milvus 连接失败 ({host}:{port}): {e}")
    results = []

    for upload in files:
        # 保存文件
        save_path = os.path.join(UPLOAD_DIR, upload.filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(upload.file, f)

        fhash = file_sha256(save_path)

        # 去重检查
        if is_file_indexed(db_conn, fhash):
            results.append({
                "file": upload.filename,
                "status": "skipped",
                "reason": "文件已入库（内容相同）"
            })
            continue

        try:
            # 解析文件
            logger.info(f"开始解析: {upload.filename}")
            chunks: list[Chunk] = parse_file(save_path)
            if not chunks:
                ext = os.path.splitext(upload.filename)[1].lower()
                is_image = ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
                if is_image and not HAS_PADDLE_OR_TESS:
                    # 图片但 OCR 引擎未安装 → 警告，不算失败
                    logger.warning(f"OCR 引擎未安装，跳过图片: {upload.filename}")
                    results.append({
                        "file":   upload.filename,
                        "status": "skipped",
                        "reason": "OCR 引擎未安装。请在服务器执行: sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim"
                    })
                else:
                    logger.warning(f"未提取到内容: {upload.filename}")
                    results.append({
                        "file":   upload.filename,
                        "status": "failed",
                        "reason": "未提取到任何内容，请确认文件格式正确且非空"
                    })
                continue

            logger.info(f"解析完成: {upload.filename}，共 {len(chunks)} 个 chunk，开始 Embedding")

            # Embedding 向量化（批量，减少开销）
            texts = [c.text for c in chunks]
            embeddings_list = embedder.encode(texts, batch_size=32,
                                              show_progress_bar=False,
                                              normalize_embeddings=True).tolist()

            logger.info(f"Embedding 完成，写入 Milvus...")

            # 插入 Milvus
            chunk_emb_pairs = list(zip(chunks, embeddings_list))
            insert_chunks(milvus_col, chunk_emb_pairs)

            # 记录到 SQLite
            ext = os.path.splitext(upload.filename)[1].upper().lstrip(".")
            register_file(db_conn, upload.filename, save_path, fhash, ext, len(chunks))

            logger.info(f"✅ 入库成功: {upload.filename}")
            results.append({
                "file":        upload.filename,
                "status":      "success",
                "chunk_count": len(chunks),
                "types":       list({c.block_type for c in chunks})
            })

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"❌ 处理失败: {upload.filename}\n{tb}")
            # reason 包含异常类型 + 消息，方便前端展示
            reason = f"{type(e).__name__}: {e}"
            results.append({
                "file":   upload.filename,
                "status": "failed",
                "reason": reason,
                "traceback": tb.splitlines()[-3:]  # 最后3行最关键
            })

    return {"results": results}


@app.get("/api/query", response_model=QueryResponse)
async def query(
    q: str = Query(..., description="问题"),
    top_k: int = Query(5, ge=1, le=20),
    file_name: Optional[str] = Query(None, description="限定在某个文件内检索")
):
    """
    RAG 问答：检索相关 chunk → 构建带来源 Prompt → LLM 生成答案
    返回答案 + 结构化 citations（文件名、页码、标题路径、原文片段）
    """
    _ensure_milvus()

    # 1. Query 向量化
    q_emb = embedder.encode(q, normalize_embeddings=True).tolist()

    # 2. 多阶段检索
    hits = multi_stage_search(milvus_col, q, q_emb, top_k=top_k, file_filter=file_name)
    if not hits:
        return QueryResponse(
            query=q, answer="知识库中未找到相关内容。",
            citations=[], llm_available=False
        )

    # 直接返回检索结果，不调用LLM
    answer = "以下是检索到的相关内容：\n\n"
    for i, h in enumerate(hits):
        answer += f"[来源{i+1}] {h['file_name']} 第{h['page']}页\n{h['text'][:300]}\n\n"
    llm_ok = False

    return QueryResponse(
        query=q,
        answer=answer,
        citations=_hits_to_citations(hits),
        llm_available=llm_ok
    )


@app.get("/api/search", response_model=SearchResponse)
async def search(
    q: str = Query(...),
    top_k: int = Query(5, ge=1, le=20),
    file_name: Optional[str] = Query(None)
):
    """纯检索（不走 LLM），返回带溯源信息的结果列表"""
    _ensure_milvus()
    q_emb = embedder.encode(q, normalize_embeddings=True).tolist()
    hits = multi_stage_search(milvus_col, q, q_emb, top_k=top_k, file_filter=file_name)
    return SearchResponse(query=q, results=_hits_to_citations(hits))


@app.get("/api/files")
async def list_files():
    """查看所有已入库的文件"""
    files = list_indexed_files(db_conn)
    return {"total": len(files), "files": files}


@app.delete("/api/files/{file_name}")
async def delete_file(file_name: str):
    """从知识库删除指定文件（向量 + 元数据一并删除）"""
    _ensure_milvus()

    # 查找 file_hash
    row = db_conn.execute(
        "SELECT file_hash FROM files WHERE file_name = ?", (file_name,)
    ).fetchone()
    if not row:
        raise HTTPException(404, f"文件 {file_name} 未在知识库中")

    fhash = row[0]
    delete_by_file_hash(milvus_col, fhash)
    delete_file_from_index(db_conn, fhash)

    return {"message": f"{file_name} 已从知识库删除"}


@app.get("/api/health")
async def health():
    milvus_ok = milvus_col is not None
    return {
        "status":   "ok" if milvus_ok else "degraded",
        "milvus":   milvus_ok,
        "embedder": embedder is not None,
        "llm_url":  OLLAMA_URL,
        "llm_model": LLM_MODEL,
    }


# ── 静态文件 ──────────────────────────────────────────────────
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

"""
app.py — RAG 知识库 API（基于 qwen3.5:9b）

接口：
  POST /api/ingest          上传文件 → 解析 → 向量化 → 入库（自动去重）
  GET  /api/query           向量检索 + qwen3.5:9b 生成带来源标注的答案
  GET  /api/search          纯向量检索（不走 LLM）
  GET  /api/files           已入库文件列表
  DELETE /api/files/{name}  删除文件
  GET  /api/health          服务健康状态
"""

import os
import warnings

# ── 最先加载 .env，让所有后续 os.environ.get() 都能读到 ──────
from dotenv import load_dotenv
import pathlib

# 优先读 .env，不存在时自动降级读 .env.example
_env_file = ".env" if pathlib.Path(".env").exists() else ".env.example"
load_dotenv(_env_file, override=False)
print(f"📋 加载配置文件: {_env_file}")

# 关闭 PaddleOCR 启动时的联网模型源检查（加速启动，离线也能用）
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# 过滤 requests/urllib3 版本不匹配的无害警告
warnings.filterwarnings("ignore", message="urllib3", category=Warning)
warnings.filterwarnings("ignore", message="chardet", category=Warning)

import shutil
import traceback
import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from embedder import get_embedder
from parser import parse_file, file_sha256, Chunk, HAS_PADDLE, HAS_TESSERACT, OCRUnavailableError
from vectorstore import (
    init_db, is_file_indexed, register_file, list_indexed_files,
    list_categories, delete_file_from_index, connect_milvus, get_or_create_collection,
    insert_chunks, smart_search, delete_by_file_hash
)

# ── 配置（全部可用环境变量覆盖）─────────────────────────────
UPLOAD_DIR  = os.getenv("UPLOAD_DIR",  "uploaded_files")
DB_PATH     = os.getenv("DB_PATH",     "rag_meta.db")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION  = os.getenv("COLLECTION",  "rag_docs")

# ── LLM 提供商配置 ────────────────────────────────────────────
# LLM_PROVIDER:
#   ollama — 本地 Ollama，走 /api/generate
#   api    — 任意 OpenAI 兼容接口（DeepSeek / 千问 / Moonshot / 本地 vLLM 等）
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "ollama")       # ollama | api
OLLAMA_URL      = os.getenv("OLLAMA_URL",   "http://localhost:11434")

# api 模式专用（LLM_PROVIDER=api 时生效）
LLM_API_BASE    = os.getenv("LLM_API_BASE", "").rstrip("/")   # 必填：API base URL，末尾不加斜杠
LLM_API_KEY     = os.getenv("LLM_API_KEY",  "")             # 必填：API Key
LLM_MODEL       = os.getenv("LLM_MODEL",    "qwen3.5:2b")   # 模型名，ollama/api 各自填对应值
LLM_OCR_MODEL   = os.getenv("LLM_OCR_MODEL","minicpm-v")    # OCR vision 模型（仅 Ollama 模式）

# LLM 生成参数
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS",    "2048"))

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── 应用 ─────────────────────────────────────────────────────
app = FastAPI(title="RAG Knowledge Base", version="2.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

db_conn    = None
milvus_col = None
embedder   = None


@app.on_event("startup")
async def startup():
    global db_conn, milvus_col, embedder
    logger.info(f"🚀 启动 RAG 服务...")
    logger.info(f"🤖 LLM 提供商: {LLM_PROVIDER.upper()}  |  📝 问答模型: {LLM_MODEL}  |  🖼  OCR 模型: {LLM_OCR_MODEL}")

    db_conn  = init_db(DB_PATH)
    logger.info("✅ SQLite 元数据库已初始化")

    embedder = get_embedder()
    logger.info("✅ Embedding 模型已加载")

    try:
        connect_milvus(MILVUS_HOST, MILVUS_PORT)
        milvus_col = get_or_create_collection(COLLECTION, dim=_embed_dim())
    except Exception as e:
        logger.warning(f"⚠️  Milvus 连接失败: {e}，将在第一次请求时重试")


# ── 响应模型 ──────────────────────────────────────────────────

class Citation(BaseModel):
    id: int
    file_name: str
    category: str
    page: int
    block_type: str
    heading_path: str
    text_snippet: str
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    llm_available: bool


class SearchResponse(BaseModel):
    query: str
    results: list[Citation]


# ── 内部工具 ──────────────────────────────────────────────────

def _embed_dim() -> int:
    try:
        return embedder.get_sentence_embedding_dimension()
    except AttributeError:
        return len(embedder.encode("test", normalize_embeddings=True))


def _ensure_milvus():
    global milvus_col
    if milvus_col is None:
        connect_milvus(MILVUS_HOST, MILVUS_PORT)
        milvus_col = get_or_create_collection(COLLECTION, dim=_embed_dim())


def _hits_to_citations(hits: list[dict]) -> list[Citation]:
    return [
        Citation(
            id=i + 1,
            file_name=h["file_name"],
            category=h.get("category", ""),
            page=h["page"],
            block_type=h["block_type"],
            heading_path=h["heading_path"],
            text_snippet=h["text"][:300],
            score=round(h.get("rerank_score", h["score"]), 4),
        )
        for i, h in enumerate(hits)
    ]


def _detect_query_type(query: str) -> str:
    """检测查询类型"""
    query_lower = query.lower()
    
    # 列表类查询
    if any(keyword in query_lower for keyword in ["所有", "全部", "清单", "列表", "汇总", "统计", "有哪些", "all", "list", "complete"]):
        return "list"
    
    # 定义类查询
    elif any(keyword in query_lower for keyword in ["什么是", "定义", "概念", "含义", "意思"]):
        return "definition"
    
    # 比较类查询
    elif any(keyword in query_lower for keyword in ["比较", "对比", "区别", "不同", "优势", "劣势"]):
        return "comparison"
    
    # 因果类查询
    elif any(keyword in query_lower for keyword in ["为什么", "原因", "因为", "所以", "导致", "影响"]):
        return "cause_effect"
    
    # 方法类查询
    elif any(keyword in query_lower for keyword in ["如何", "怎么", "方法", "步骤", "教程", "指南"]):
        return "method"
    
    # 其他查询
    else:
        return "general"


def _build_prompt(query: str, hits: list[dict]) -> str:
    """构建带来源标注的 RAG Prompt，专门针对 qwen 系列模型优化"""
    # 检测查询类型
    query_type = _detect_query_type(query)
    
    # 构建上下文
    context_parts = []
    for i, h in enumerate(hits):
        loc = f"文件：{h['file_name']} 第{h['page']}页"
        if h.get("category"):
            loc = f"分类：{h['category']} / {loc}"
        if h["heading_path"]:
            loc += f" / {h['heading_path']}"
        # 突出显示关键词
        text = h['text']
        # 提取查询中的关键词
        query_keywords = [kw for kw in query.split() if len(kw) > 1]
        for kw in query_keywords:
            if kw in text:
                text = text.replace(kw, f"**{kw}**")
        context_parts.append(f"[来源{i + 1}] {loc}\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    # 提取本次涉及的分类列表，提示 AI 注意边界
    categories = list(dict.fromkeys(h["category"] for h in hits if h.get("category")))
    category_hint = ""
    if categories:
        category_hint = f"\n注意：以下参考资料来自【{'、'.join(categories)}】分类，回答时请结合分类背景作答，不要混淆不同分类的内容。\n"

    # 根据查询类型定制提示词
    query_type_hint = ""
    if query_type == "list":
        query_type_hint = "\n**列表类查询**：请将找到的所有相关信息都列出来，确保完整性，不要遗漏任何信息。\n"
    elif query_type == "definition":
        query_type_hint = "\n**定义类查询**：请提供清晰、准确的定义，并结合参考资料中的内容进行解释。\n"
    elif query_type == "comparison":
        query_type_hint = "\n**比较类查询**：请对相关内容进行详细比较，分析其异同点。\n"
    elif query_type == "cause_effect":
        query_type_hint = "\n**因果类查询**：请分析原因和结果，提供逻辑清晰的解释。\n"
    elif query_type == "method":
        query_type_hint = "\n**方法类查询**：请提供详细的步骤和操作指南。\n"

    return f"""你是一个专业的知识库问答助手。请严格基于以下参考资料回答用户问题。
{category_hint}
{query_type_hint}
要求：
1. **前面的来源更可靠**：来源按相关度排序，[来源1]、[来源2]...越靠前越重要，优先采纳前面来源的信息
2. **精准引用**：每引用某条来源的内容时，在句末用 [来源N] 标注（N 为来源编号）
3. **信息完整性**：若参考资料中没有相关信息，明确回答"参考资料中未找到相关信息"，不要编造
4. **回答质量**：回答要简洁、准确、有条理，使用专业术语
5. **多来源验证**：若有多个来源支持同一观点，可同时标注多个来源，如 [来源1][来源2]
6. **格式规范**：对于列表、步骤等内容，使用适当的格式（如数字编号、 bullet points）

参考资料（按相关度从高到低排序）：
{context}

用户问题：{query}

请基于上述资料作答，优先相信排在前面的来源："""


async def _call_llm(prompt: str) -> str:
    """
    统一 LLM 调用入口，根据 LLM_PROVIDER 路由：
      ollama — 本地 Ollama /api/generate
      api    — 任意 OpenAI 兼容接口 /chat/completions
               （DeepSeek / 千问 / Moonshot / vLLM 等）
    """
    import httpx

    if LLM_PROVIDER == "api":
        if not LLM_API_KEY:
            raise RuntimeError("LLM_PROVIDER=api 时必须设置 LLM_API_KEY")
        if not LLM_API_BASE:
            raise RuntimeError("LLM_PROVIDER=api 时必须设置 LLM_API_BASE")
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type":  "application/json",
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{LLM_API_BASE}/chat/completions",
                json=payload, headers=headers
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    else:
        # ── Ollama（本地/局域网）────────────────────────────────
        payload = {
            "model":    LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream":   False,
            "think":    False,   # 关闭思考模式，避免 content 为空
            "options": {
                "temperature": LLM_TEMPERATURE,
                "num_predict": LLM_MAX_TOKENS,
            }
        }
        async with httpx.AsyncClient(timeout=180) as client:
            url = f"{OLLAMA_URL}/api/chat"
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            msg = resp.json().get("message", {})
            content = msg.get("content", "").strip()
            thinking = msg.get("thinking", "").strip()
            logger.info(f"🤖 content({len(content)}字) thinking({len(thinking)}字)")
            if not content and thinking:
                logger.info("🤖 content 为空，使用完整 thinking 内容")
                content = thinking
            return content


# ── API 路由 ──────────────────────────────────────────────────

@app.post("/api/ingest")
async def ingest_files(
    files: list[UploadFile] = File(...),
    milvus_host: str = Form(default=None),
    milvus_port: str = Form(default=None),
    category: str = Form(default=""),
):
    """上传文件，解析 → 向量化 → 入库 Milvus（按内容哈希去重）"""
    global milvus_col

    host = milvus_host or MILVUS_HOST
    port = milvus_port or MILVUS_PORT

    try:
        connect_milvus(host, port)
        milvus_col = get_or_create_collection(COLLECTION, dim=_embed_dim())
    except Exception as e:
        raise HTTPException(503, f"Milvus 连接失败 ({host}:{port}): {e}")

    results = []

    for upload in files:
        save_path = os.path.join(UPLOAD_DIR, upload.filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(upload.file, f)

        fhash = file_sha256(save_path)

        # 去重
        if is_file_indexed(db_conn, fhash):
            results.append({"file": upload.filename, "status": "skipped",
                             "reason": "文件已入库（内容相同）"})
            continue

        try:
            logger.info(f"开始解析: {upload.filename}")
            chunks: list[Chunk] = parse_file(save_path)

            if not chunks:
                results.append({"file": upload.filename, "status": "failed",
                                 "reason": "未提取到任何内容，请确认文件格式正确且非空"})
                continue

            logger.info(f"解析完成: {len(chunks)} 个 chunk，开始 Embedding")

            texts = [c.text for c in chunks]
            embeddings_list = embedder.encode(
                texts, batch_size=32, show_progress_bar=False,
                normalize_embeddings=True
            ).tolist()

            insert_chunks(milvus_col, list(zip(chunks, embeddings_list)), category=category)

            ext_label = os.path.splitext(upload.filename)[1].upper().lstrip(".")
            register_file(db_conn, upload.filename, save_path, fhash,
                          ext_label, len(chunks), category=category)

            logger.info(f"✅ 入库成功: {upload.filename} ({len(chunks)} chunks)")
            results.append({
                "file":        upload.filename,
                "status":      "success",
                "chunk_count": len(chunks),
                "types":       list({c.block_type for c in chunks}),
                "chunks":      [
                    {
                        "index": c.chunk_index,
                        "type":  c.block_type,
                        "page":  c.page,
                        "text":  c.text,   # 完整内容
                    }
                    for c in chunks
                ],
            })

        except OCRUnavailableError as e:
            # OCR 引擎未安装：单独状态，前端展示安装指引
            logger.warning(f"⚠️  OCR 不可用，跳过图片: {upload.filename}")
            results.append({
                "file":   upload.filename,
                "status": "ocr_unavailable",
                "reason": str(e),
            })

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"❌ 处理失败: {upload.filename}\n{tb}")
            results.append({
                "file":      upload.filename,
                "status":    "failed",
                "reason":    f"{type(e).__name__}: {e}",
                "traceback": tb.splitlines()[-3:]
            })

    return {"results": results}


async def _detect_category(q: str, categories: list[str]) -> Optional[str]:
    """用 LLM 判断问题属于哪个分类，返回分类名或 None（跨分类/无法判断）"""
    if not categories:
        return None
    if len(categories) == 1:
        return categories[0]   # 只有一个分类，直接用
    import httpx
    cats_str = "、".join(f'"{c}"' for c in categories)
    prompt = (
        f"知识库中有以下分类：{cats_str}。\n"
        f"用户问题：「{q}」\n"
        f"请判断这个问题最可能属于哪个分类？\n"
        f"只输出分类名称本身，不要任何解释。"
        f"如果无法判断或问题跨多个分类，输出「全部」。"
    )
    try:
        if LLM_PROVIDER == "api":
            if not LLM_API_KEY or not LLM_API_BASE:
                return None
            payload = {
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 32,
                "stream": False,
            }
            headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(f"{LLM_API_BASE}/chat/completions", json=payload, headers=headers)
                r.raise_for_status()
                result = r.json()["choices"][0]["message"]["content"].strip().strip('"').strip()
        else:
            payload = {
                "model":    LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
                "think":    False,
                "options":  {"temperature": 0, "num_predict": 32},
            }
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
                r.raise_for_status()
                msg = r.json().get("message", {})
                content = msg.get("content", "").strip()
                if not content:
                    content = msg.get("thinking", "").strip()
                result = content.strip('"').strip()
        if result in categories:
            logger.info(f"🗂  自动路由到分类: {result}")
            return result
        if "全部" in result or "无法" in result:
            return None
        # 模糊匹配
        for c in categories:
            if c in result or result in c:
                logger.info(f"🗂  模糊匹配分类: {c}")
                return c
    except Exception as e:
        logger.warning(f"分类识别失败，使用全库检索: {e}")
    return None


@app.get("/api/query", response_model=QueryResponse)
async def query(
    q: str = Query(..., description="问题"),
    top_k: int = Query(5, ge=1, le=20),
    file_name: Optional[str] = Query(None, description="限定文件名检索"),
    category: Optional[str] = Query(None, description="手动限定分类（留空则自动路由）"),
):
    _ensure_milvus()
    q_emb = embedder.encode(q, normalize_embeddings=True).tolist()

    # 自动分类路由：手动指定分类时直接用，否则让 LLM 判断
    resolved_category = category
    if not resolved_category:
        all_cats = list_categories(db_conn)
        resolved_category = await _detect_category(q, all_cats)

    hits, search_mode = smart_search(milvus_col, q, q_emb, top_k=top_k,
                                      file_filter=file_name, category_filter=resolved_category)
    logger.info(f"🔍 检索模式: {search_mode} | 命中 {len(hits)} 条")

    # 如果限定分类后没结果，降级到全库检索
    if not hits and resolved_category:
        logger.info(f"分类 '{resolved_category}' 无结果，降级全库检索")
        hits, search_mode = smart_search(milvus_col, q, q_emb, top_k=top_k, file_filter=file_name)
        logger.info(f"🔍 降级检索模式: {search_mode} | 命中 {len(hits)} 条")

    if not hits:
        return QueryResponse(
            query=q, answer="知识库中未找到相关内容。",
            citations=[], llm_available=False
        )

    # 打印送入 LLM 的 top-k chunks（列表类查询结果多，只显示摘要）
    display_limit = 10 if search_mode == "full_text" else len(hits)
    logger.info(f"📋 送入 LLM 的 {len(hits)} chunks（查询: {q[:50]}...）")
    for i, h in enumerate(hits[:display_limit]):
        logger.info(f"  ── chunk[{i+1}] {h['file_name']} 第{h['page']}页 [{h['block_type']}] score={h.get('score', 0):.4f}")
        text_preview = h['text'][:200].replace(chr(10), ' ')
        logger.info(f"    {text_preview}...")
    if len(hits) > display_limit:
        logger.info(f"  ... 还有 {len(hits) - display_limit} 条结果未显示")

    # 调用 qwen3.5:9b
    llm_ok = False
    answer = ""
    try:
        prompt = _build_prompt(q, hits)
        answer = await _call_llm(prompt)
        llm_ok = True
        logger.info(f"LLM 回答生成成功，长度 {len(answer)} 字符")
    except Exception as e:
        logger.warning(f"LLM 调用失败，降级为检索摘要: {e}")
        # 降级：直接返回检索片段
        lines = [f"[来源{i+1}] {h['file_name']} 第{h['page']}页\n{h['text'][:300]}"
                 for i, h in enumerate(hits)]
        answer = "（LLM 暂不可用，以下为检索到的相关内容）\n\n" + "\n\n".join(lines)

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
    file_name: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
):
    """纯向量检索（不走 LLM），返回带溯源信息的结果"""
    _ensure_milvus()
    q_emb = embedder.encode(q, normalize_embeddings=True).tolist()
    hits, search_mode = smart_search(milvus_col, q, q_emb, top_k=top_k,
                                      file_filter=file_name, category_filter=category)
    logger.info(f"🔍 检索模式: {search_mode} | 命中 {len(hits)} 条")
    return SearchResponse(query=q, results=_hits_to_citations(hits))


@app.get("/api/categories")
async def get_categories():
    """返回所有已有分类列表"""
    return {"categories": list_categories(db_conn)}


@app.get("/api/files")
async def list_files():
    files = list_indexed_files(db_conn)
    return {"total": len(files), "files": files}


@app.delete("/api/files/{file_name}")
async def delete_file(file_name: str):
    _ensure_milvus()
    row = db_conn.execute(
        "SELECT file_hash FROM files WHERE file_name = ?", (file_name,)
    ).fetchone()
    if not row:
        raise HTTPException(404, f"文件 {file_name} 未在知识库中")
    delete_by_file_hash(milvus_col, row[0])
    delete_file_from_index(db_conn, row[0])
    return {"message": f"{file_name} 已从知识库删除"}


@app.delete("/api/collection")
async def drop_collection():
    """
    彻底清空 Milvus Collection（drop + 重建）以及 SQLite 文件记录。
    比逐个文件删除更彻底，能清掉元数据之外的孤立向量。
    """
    global milvus_col
    _ensure_milvus()

    try:
        from pymilvus import utility
        if utility.has_collection(COLLECTION):
            utility.drop_collection(COLLECTION)
            logger.info(f"✅ Milvus Collection '{COLLECTION}' 已 drop")

        milvus_col = get_or_create_collection(COLLECTION, dim=_embed_dim())
        logger.info(f"✅ Milvus Collection '{COLLECTION}' 已重建")
    except Exception as e:
        raise HTTPException(500, f"Milvus 清空失败: {e}")

    # 清空 SQLite 文件记录
    db_conn.execute("DELETE FROM files")
    db_conn.commit()
    logger.info("✅ SQLite 元数据已清空")

    return {"message": f"知识库 '{COLLECTION}' 已彻底清空并重建"}


@app.get("/api/health")
async def health():
    """服务健康检查，包含 LLM 连通性探测"""
    from embedder import MODEL_NAME as EMBED_MODEL_NAME
    milvus_ok = milvus_col is not None
    llm_reachable = False
    try:
        import httpx
        if LLM_PROVIDER == "api":
            # api 模式：只验证配置是否填写，不发网络请求（避免每次轮询消耗 API 配额）
            llm_reachable = bool(LLM_API_KEY and LLM_API_BASE)
        else:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{OLLAMA_URL}/api/tags")
                llm_reachable = r.status_code == 200
    except Exception:
        pass

    return {
        "status":        "ok" if (milvus_ok and llm_reachable) else "degraded",
        "milvus":        milvus_ok,
        "embedder":      embedder is not None,
        "embed_model":   EMBED_MODEL_NAME,
        "llm_provider":  LLM_PROVIDER,
        "llm_url":       LLM_API_BASE if LLM_PROVIDER == "api" else OLLAMA_URL,
        "llm_ocr_model": LLM_OCR_MODEL,
        "llm_model":     LLM_MODEL,
        "llm_reachable": llm_reachable,
    }


# ── 静态文件 ──────────────────────────────────────────────────
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
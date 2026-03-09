# RAG 知识库系统 v2.0 — 重构说明

## 文件对照表（哪些文件被替换了）

| 原文件 | 新文件 | 说明 |
|--------|--------|------|
| `loader.py` | `parser.py` | 完全重写，保留完整溯源元数据 |
| `pipeline.py` | `parser.py` | chunking 逻辑合并到 parser |
| `vectorstore.py` | `vectorstore.py` | Milvus schema 新增溯源字段，加 SQLite 去重 |
| `embedder.py` | `embedder.py` | 改为单例，保留向后兼容接口 |
| `app.py` | `app.py` | API 重构，新增 /api/ingest，query 返回 citations |
| `main.py` | 保留不变 | CLI 入口，字段名需小改（见下） |

PDF 转 Word 的几个脚本（`pdf_to_word*.py` 等）不属于 RAG 核心流程，保留不动。

---

## 核心改进说明

### 1. parser.py — 为什么比原来好

**原来的问题（loader.py + pipeline.py）：**
```python
# 原来 — LangChain loader 丢失结构
loader = PyPDFLoader(filepath)
docs = loader.load()  # 只有 page_content 和 source，没有标题层级

# chunking 按字符数硬切，语义边界错误
text_splitter = RecursiveCharacterTextSplitter(chunk_size=380)
```

**现在：**
```python
# 每个 chunk 携带完整溯源信息
chunk = Chunk(
    text="...",
    file_name="report.pdf",
    page=5,
    block_type="table",          # text | heading | table | image
    heading_path="第三章 > 3.2 财务分析",  # 标题路径！
    table_headers=["季度", "收入", "利润"],
)

# 转 LangChain Document 时元数据完整保留
doc = chunk.to_langchain_doc()
# doc.metadata = {file_name, page, block_type, heading_path, ...}
```

### 2. vectorstore.py — Milvus Collection 字段设计

原来只存了 `source` 和 `type`，现在字段完整：

```
id             INT64     主键
embedding      FLOAT_VEC 向量（768维）
file_name      VARCHAR   文件名
file_hash      VARCHAR   SHA256（去重用）
page           INT32     页码
block_type     VARCHAR   text/heading/table/image
heading_path   VARCHAR   标题路径 "章 > 节 > 小节"
heading_level  INT32     标题级别
table_headers  VARCHAR   表格表头（|分隔）
chunk_index    INT32     在文件中的顺序
text           VARCHAR   内容（截断到4096字符）
```

### 3. app.py — 溯源输出格式

```json
{
  "query": "Q3净利润是多少",
  "answer": "根据财务报告，Q3净利润为2300万元 [来源1]，华东区贡献最大 [来源2]。",
  "citations": [
    {
      "id": 1,
      "file_name": "2024_annual_report.pdf",
      "page": 23,
      "block_type": "table",
      "heading_path": "财务摘要 > Q3 利润分析",
      "text_snippet": "Q3净利润2300万元，同比增长12.1%...",
      "score": 0.943
    },
    {
      "id": 2,
      "file_name": "regional_breakdown.xlsx",
      "page": 1,
      "block_type": "table",
      "heading_path": "Sheet: 区域分析",
      "text_snippet": "华东区收入占比38%...",
      "score": 0.891
    }
  ],
  "llm_available": true
}
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt

# OCR 选一个安装：
# PaddleOCR（中文效果好，推荐）
pip install paddlepaddle paddleocr

# 或 Tesseract（轻量）
# apt-get install tesseract-ocr tesseract-ocr-chi-sim
# pip install pytesseract Pillow
```

### 2. 启动 Milvus（Docker）

```bash
# 官方 standalone 模式（免费）
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh | sh
```

### 3. 启动 LLM（Ollama）

```bash
ollama serve
ollama pull qwen3.5:9b   # 推荐中文模型
```

### 4. 启动服务

```bash
python app.py
# 或
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 5. 使用 API

```bash
# 上传文件
curl -X POST http://localhost:8000/api/ingest \
  -F "files=@report.pdf" \
  -F "files=@data.xlsx"

# 问答（带溯源）
curl "http://localhost:8000/api/query?q=Q3净利润是多少&top_k=5"

# 纯检索
curl "http://localhost:8000/api/search?q=能源管理&top_k=10"

# 查看已入库文件
curl "http://localhost:8000/api/files"

# 限定在某个文件内检索
curl "http://localhost:8000/api/query?q=预算&file_name=budget_2024.xlsx"
```

---

## main.py 小改动（如果还用 CLI）

```python
# 原来
from vectorstore import build_vectorstore, multi_stage_search
from pipeline import unstructured_splitter, split_by_heading

# 改为
from parser import parse_directory
from vectorstore import (
    connect_milvus, get_or_create_collection,
    insert_chunks, multi_stage_search
)
from embedder import get_embedder

# 用法
chunks = parse_directory("data/")
embedder = get_embedder()
embeddings_list = embedder.encode([c.text for c in chunks],
                                   normalize_embeddings=True).tolist()
connect_milvus()
col = get_or_create_collection()
insert_chunks(col, list(zip(chunks, embeddings_list)))
hits = multi_stage_search(col, "能源管理",
                           embedder.encode("能源管理", normalize_embeddings=True).tolist())
```

---

## 常见问题

**Q: PaddleOCR 安装很慢/失败？**
改用 Tesseract：`apt-get install tesseract-ocr tesseract-ocr-chi-sim && pip install pytesseract Pillow`

**Q: Milvus 连不上？**
检查 Docker 是否运行：`docker ps | grep milvus`，端口默认 19530。

**Q: 模型文件太大？**
embedder.py 里改为 `BAAI/bge-small-zh-v1.5`（轻量版，效果稍差）。

**Q: 想指定不同 LLM 模型？**
设置环境变量：`LLM_MODEL=qwen3.5:9b python app.py`
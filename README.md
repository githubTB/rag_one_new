# RAG One — 智能知识库（qwen3.5:9b 驱动）

基于 Milvus + qwen3.5:9b 的本地 RAG 问答系统，支持 PDF / Word / Excel / 图片，带完整溯源引用。

## 文件结构

```
├── app.py          # FastAPI 服务入口（主力 LLM：qwen3.5:9b）
├── parser.py       # 文档解析（PDF/DOCX/XLSX/图片 OCR）
├── vectorstore.py  # Milvus 向量库 + SQLite 元数据
├── embedder.py     # Embedding 单例（BAAI/bge-base-zh-v1.5）
├── static/         # 前端界面
├── requirements.txt
└── .env.example    # 环境变量模板
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt

# OCR 选一个（推荐 PaddleOCR，中文效果好）：
pip install paddlepaddle paddleocr

# 或 Tesseract（轻量）：
# apt-get install tesseract-ocr tesseract-ocr-chi-sim
# pip install pytesseract
```

### 2. 启动 Milvus

```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh | sh
```

### 3. 启动 Ollama + 拉取模型

```bash
ollama serve
ollama pull qwen3.5:9b
```

### 4. 配置环境变量（可选）

```bash
cp .env.example .env
# 按需修改 .env
```

### 5. 启动服务

```bash
python app.py
# 或
uvicorn app:app --host 0.0.0.0 --port 8000
```

访问 http://localhost:8000 打开 Web 界面。

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/ingest` | 上传文件并入库（自动去重） |
| GET  | `/api/query?q=...&top_k=5` | RAG 问答（含 LLM 生成答案） |
| GET  | `/api/search?q=...` | 纯向量检索（不走 LLM） |
| GET  | `/api/files` | 已入库文件列表 |
| DELETE | `/api/files/{name}` | 删除文件 |
| GET  | `/api/health` | 服务健康状态 |

### 问答响应示例

```json
{
  "query": "Q3净利润是多少",
  "answer": "Q3净利润为2300万元 [来源1]，华东区贡献最大 [来源2]。",
  "citations": [
    {
      "id": 1,
      "file_name": "annual_report.pdf",
      "page": 23,
      "block_type": "table",
      "heading_path": "财务摘要 > Q3 利润",
      "text_snippet": "Q3净利润2300万元，同比增长12.1%...",
      "score": 0.943
    }
  ],
  "llm_available": true
}
```

## 支持格式

| 格式 | 说明 |
|------|------|
| PDF  | 文字页直接提取；扫描页自动 OCR；表格用 pdfplumber 提取 |
| DOCX | 按 body 顺序解析，保留标题层级、表格、嵌入图片 OCR |
| XLSX | 每个 Sheet 作为一个 table chunk |
| 图片 | PNG/JPG/BMP/TIFF，OCR 提取文字 |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_MODEL` | `qwen3.5:9b` | Ollama 模型名 |
| `LLM_TEMPERATURE` | `0.3` | 生成温度（越低越稳定） |
| `LLM_MAX_TOKENS` | `2048` | 最大输出 token 数 |
| `MILVUS_HOST` | `localhost` | Milvus 主机 |
| `MILVUS_PORT` | `19530` | Milvus 端口 |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama 地址 |

## 常见问题

**Q: Milvus 连不上？**
检查 Docker：`docker ps | grep milvus`，默认端口 19530。

**Q: qwen3.5:9b 响应慢？**
调低 `LLM_MAX_TOKENS`，或改用 `qwen3.5:3b`（速度更快）：
```bash
LLM_MODEL=qwen3.5:3b python app.py
```

**Q: 想换更强的 Embedding 模型？**
在 `embedder.py` 里改 `MODEL_NAME = "BAAI/bge-m3"`。

**Q: OCR 效果差？**
优先安装 PaddleOCR（对中文识别率远高于 Tesseract）：
```bash
pip install paddlepaddle paddleocr
```
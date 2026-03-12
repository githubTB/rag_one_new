"""
vectorstore.py — Milvus 向量库 + SQLite 文件元数据管理

字段设计：id、embedding、file_name、file_hash、page、block_type、
          heading_path、heading_level、table_headers、chunk_index、text、category
"""

import re
import sqlite3
from typing import Optional

# ── jieba 分词（延迟加载）────────────────────────────────────────
_jieba = None

# 化工/报告类专业词，jieba 默认词典不含，需手动补充
_DOMAIN_WORDS = [
    "法人代表", "统一社会信用代码", "排污许可证", "营业执照",
    "聚四氢呋喃", "1,4-丁二醇", "丁二醇", "乙炔", "甲醛", "甲醇",
    "PTMEG", "BDO", "BYD", "BED", "THF", "NMP",
    "碳排放", "碳中和", "碳达峰", "温室气体",
    "绿色工厂", "清洁生产", "节能减排", "能源管理",
    "危险废物", "固体废物", "废水处理", "排污口",
    "环境影响评价", "竣工验收", "三同时",
    "单位产品能耗", "综合能耗", "标准煤",
]


def _get_jieba():
    """延迟加载 jieba，失败时静默回退到 re.split 模式"""
    global _jieba
    if _jieba is None:
        try:
            import jieba as _j
            _j.setLogLevel("ERROR")
            for w in _DOMAIN_WORDS:
                _j.add_word(w)
            _jieba = _j
            print("✅ jieba 分词已加载")
        except ImportError:
            _jieba = "unavailable"
            print("⚠️  jieba 未安装，关键词检索降级为 re.split 模式")
    return None if _jieba == "unavailable" else _jieba


def _tokenize(query: str) -> list[str]:
    """
    中文分词：优先用 jieba，不可用时退回 re.split。
    返回过滤停用词后的有效关键词列表（按长度降序，最多 8 个）。
    """
    stop = {
        "的", "了", "是", "在", "和", "与", "或", "有", "这", "那", "中", "为", "对",
        "所有", "全部", "哪些", "什么", "请问", "告诉我", "查询", "列出", "列举",
        "一下", "一些", "相关", "情况", "如何", "怎么", "是否", "有没有",
    }
    jb = _get_jieba()
    if jb:
        tokens = list(jb.cut(query.strip()))
    else:
        tokens = re.split(r"[\s，。！？；、,\.!?;]+", query.strip())

    words = [t for t in tokens if len(t) >= 2 and t not in stop
             and not re.match(r"^[\s\W]+$", t)]

    seen: set = set()
    result = []
    for w in sorted(words, key=len, reverse=True):
        if w not in seen:
            seen.add(w)
            result.append(w)
    return result[:8]


# ── Reranker（延迟加载）────────────────────────────────────────
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("BAAI/bge-reranker-base", local_files_only=True)
            print("✅ Reranker 已加载（本地缓存）")
        except Exception:
            try:
                from sentence_transformers import CrossEncoder
                _reranker = CrossEncoder("BAAI/bge-reranker-base")
                print("✅ Reranker 已加载（网络）")
            except Exception as e:
                print(f"⚠️  Reranker 不可用: {e}")
                _reranker = "unavailable"
    return None if _reranker == "unavailable" else _reranker


# ── SQLite 文件元数据 ─────────────────────────────────────────

def init_db(db_path: str = "rag_meta.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name   TEXT NOT NULL,
            file_path   TEXT NOT NULL,
            file_hash   TEXT NOT NULL UNIQUE,
            file_type   TEXT,
            category    TEXT DEFAULT '',
            chunk_count INTEGER DEFAULT 0,
            status      TEXT DEFAULT 'pending',
            created_at  TEXT DEFAULT (datetime('now')),
            updated_at  TEXT DEFAULT (datetime('now'))
        )
    """)
    # 兼容旧库：如果 category 列不存在则添加
    try:
        conn.execute("ALTER TABLE files ADD COLUMN category TEXT DEFAULT ''")
        conn.commit()
    except Exception:
        pass  # 列已存在，忽略
    conn.commit()
    return conn


def is_file_indexed(conn: sqlite3.Connection, file_hash: str) -> bool:
    row = conn.execute(
        "SELECT status FROM files WHERE file_hash = ?", (file_hash,)
    ).fetchone()
    return row is not None and row[0] == "indexed"


def register_file(conn: sqlite3.Connection, file_name: str, file_path: str,
                  file_hash: str, file_type: str, chunk_count: int,
                  category: str = ""):
    conn.execute("""
        INSERT OR REPLACE INTO files
        (file_name, file_path, file_hash, file_type, category, chunk_count, status, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, 'indexed', datetime('now'))
    """, (file_name, file_path, file_hash, file_type, category, chunk_count))
    conn.commit()


def list_indexed_files(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("""
        SELECT file_name, file_type, category, chunk_count, created_at
        FROM files WHERE status = 'indexed'
        ORDER BY category, created_at DESC
    """).fetchall()
    return [{"file_name": r[0], "file_type": r[1], "category": r[2],
             "chunk_count": r[3], "indexed_at": r[4]} for r in rows]


def list_categories(conn: sqlite3.Connection) -> list[str]:
    """返回所有已有分类（去重排序）"""
    rows = conn.execute("""
        SELECT DISTINCT category FROM files
        WHERE status = 'indexed' AND category != ''
        ORDER BY category
    """).fetchall()
    return [r[0] for r in rows]


def delete_file_from_index(conn: sqlite3.Connection, file_hash: str):
    conn.execute("DELETE FROM files WHERE file_hash = ?", (file_hash,))
    conn.commit()


# ── Milvus 操作 ───────────────────────────────────────────────

def connect_milvus(host: str = "localhost", port: str = "19530"):
    from pymilvus import connections
    connections.connect("default", host=host, port=port)
    print(f"✅ Milvus 已连接 {host}:{port}")


def get_or_create_collection(collection_name: str = "rag_docs", dim: int = 768):
    from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility

    if utility.has_collection(collection_name):
        print(f"✅ 复用已有 Collection: {collection_name}")
        col = Collection(collection_name)
        col.load()
        return col

    fields = [
        FieldSchema("id",            DataType.INT64,         is_primary=True, auto_id=True),
        FieldSchema("embedding",     DataType.FLOAT_VECTOR,  dim=dim),
        FieldSchema("file_name",     DataType.VARCHAR,       max_length=512),
        FieldSchema("file_hash",     DataType.VARCHAR,       max_length=64),
        FieldSchema("category",      DataType.VARCHAR,       max_length=256),
        FieldSchema("page",          DataType.INT32),
        FieldSchema("block_type",    DataType.VARCHAR,       max_length=32),
        FieldSchema("heading_path",  DataType.VARCHAR,       max_length=1024),
        FieldSchema("heading_level", DataType.INT32),
        FieldSchema("table_headers", DataType.VARCHAR,       max_length=1024),
        FieldSchema("chunk_index",   DataType.INT32),
        FieldSchema("text",          DataType.VARCHAR,       max_length=4096),
    ]
    schema = CollectionSchema(fields, description="RAG knowledge base with citations")
    col = Collection(collection_name, schema)
    col.create_index("embedding", {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 256}
    })
    col.load()
    print(f"✅ 创建 Collection: {collection_name} (dim={dim})")
    return col


def insert_chunks(col, chunks_with_embeddings: list[tuple],
                  category: str = "") -> list[int]:
    if not chunks_with_embeddings:
        return []

    data = {k: [] for k in [
        "embedding", "file_name", "file_hash", "category", "page", "block_type",
        "heading_path", "heading_level", "table_headers", "chunk_index", "text"
    ]}

    for chunk, emb in chunks_with_embeddings:
        data["embedding"].append(emb)
        data["file_name"].append(chunk.file_name[:512])
        data["file_hash"].append(chunk.file_hash[:64])
        data["category"].append(category[:256])
        data["page"].append(chunk.page)
        data["block_type"].append(chunk.block_type[:32])
        data["heading_path"].append(chunk.heading_path[:1024])
        data["heading_level"].append(chunk.heading_level)
        data["table_headers"].append("|".join(chunk.table_headers)[:1024])
        data["chunk_index"].append(chunk.chunk_index)
        data["text"].append(chunk.text[:4000])

    result = col.insert(list(data.values()))
    col.flush()
    return result.primary_keys


def vector_search(col, query_embedding: list[float], top_k: int = 20,
                  file_filter: Optional[str] = None,
                  category_filter: Optional[str] = None) -> list[dict]:
    exprs = []
    if file_filter:
        exprs.append(f'file_name == "{file_filter}"')
    if category_filter:
        exprs.append(f'category == "{category_filter}"')
    expr = " and ".join(exprs) if exprs else None

    results = col.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=top_k,
        expr=expr,
        output_fields=["file_name", "file_hash", "category", "page", "block_type",
                        "heading_path", "table_headers", "chunk_index", "text"]
    )
    hits = []
    for hit in results[0]:
        e = hit.entity
        hits.append({
            "text":          e.get("text", ""),
            "score":         round(hit.score, 4),
            "file_name":     e.get("file_name", ""),
            "file_hash":     e.get("file_hash", ""),
            "category":      e.get("category", ""),
            "page":          e.get("page", 1),
            "block_type":    e.get("block_type", "text"),
            "heading_path":  e.get("heading_path", ""),
            "table_headers": e.get("table_headers", ""),
            "chunk_index":   e.get("chunk_index", 0),
        })
    return hits


def delete_by_file_hash(col, file_hash: str):
    col.delete(f'file_hash == "{file_hash}"')
    col.flush()
    print(f"✅ 已从 Milvus 删除: {file_hash[:8]}...")


def _keyword_search(col, query: str, top_k: int = 20,
                    file_filter: Optional[str] = None,
                    category_filter: Optional[str] = None) -> list[dict]:
    """
    关键词兜底检索：用 jieba 分词后，Milvus expr like OR 匹配 text 字段。
    用于捕捉向量检索遗漏的精确词（人名、编号、专业术语等）。
    """
    keywords = _tokenize(query)
    if not keywords:
        return []
    # Milvus like 最多用前 5 个关键词（避免 expr 过长）
    kw_for_expr = keywords[:5]

    exprs = []
    if file_filter:
        exprs.append(f'file_name == "{file_filter}"')
    if category_filter:
        exprs.append(f'category == "{category_filter}"')
    # text 非空保护（避免 None 导致 BM25 报错）
    exprs.append('text != ""')
    # 每个关键词 OR 匹配
    kw_expr = " or ".join(f'text like "%{kw}%"' for kw in kw_for_expr)
    exprs.append(f"({kw_expr})")
    expr = " and ".join(exprs)

    try:
        results = col.query(
            expr=expr,
            output_fields=["file_name", "file_hash", "category", "page", "block_type",
                           "heading_path", "table_headers", "chunk_index", "text"],
            limit=top_k
        )
        hits = []
        for e in results:
            text = e.get("text") or ""
            if len(text.strip()) < 10:  # 过滤垃圾短 chunk
                continue
            hit_count = sum(1 for kw in keywords if kw in text)
            hits.append({
                "text":          text,
                "score":         0.0,
                "kw_score":      hit_count / len(keywords),
                "file_name":     e.get("file_name", ""),
                "file_hash":     e.get("file_hash", ""),
                "category":      e.get("category", ""),
                "page":          e.get("page", 1),
                "block_type":    e.get("block_type", "text"),
                "heading_path":  e.get("heading_path", ""),
                "table_headers": e.get("table_headers", ""),
                "chunk_index":   e.get("chunk_index", 0),
            })
        return hits
    except Exception as e:
        print(f"⚠️  关键词检索失败: {e}")
        return []


def multi_stage_search(col, query: str, query_embedding: list[float],
                       top_k: int = 5, file_filter: Optional[str] = None,
                       category_filter: Optional[str] = None) -> list[dict]:
    """向量检索 + 关键词兜底 → 融合去重 → Reranker 精排（可选）"""
    MIN_TEXT_LEN = 10  # 过滤太短的垃圾 chunk

    # 1. 向量检索（多取几个用于候选，最终截断到 top_k）
    vec_hits = vector_search(col, query_embedding, top_k=top_k * 4,
                             file_filter=file_filter, category_filter=category_filter)
    for h in vec_hits:
        h.setdefault("kw_score", 0.0)

    # 2. 关键词兜底
    kw_hits = _keyword_search(col, query, top_k=top_k * 2,
                              file_filter=file_filter, category_filter=category_filter)

    # 3. 融合去重（以 file_name+chunk_index 为唯一键）
    seen: dict[str, dict] = {}
    for h in vec_hits:
        if len((h.get("text") or "").strip()) < MIN_TEXT_LEN:
            continue
        key = f"{h['file_name']}#{h['chunk_index']}"
        seen[key] = h
    for h in kw_hits:
        key = f"{h['file_name']}#{h['chunk_index']}"
        if key not in seen:
            seen[key] = h
        else:
            seen[key]["kw_score"] = max(seen[key].get("kw_score", 0), h["kw_score"])

    candidates = list(seen.values())
    if not candidates:
        return []

    # 4. Reranker 精排 或 融合分排序
    reranker = get_reranker()
    if reranker:
        try:
            pairs = [[query, c["text"]] for c in candidates]
            scores = reranker.predict(pairs)
            for i, c in enumerate(candidates):
                c["rerank_score"] = float(scores[i])

            # 关键词精确命中的 chunk 设保底 rerank 分，防止被 Reranker 错误压低
            # 保底值 = 当前最高分 * 0.85，确保精确命中的 chunk 不被排到截断线外
            kw_hits_idx = [i for i, c in enumerate(candidates) if c.get("kw_score", 0) >= 0.5]
            if kw_hits_idx:
                max_score = max(c["rerank_score"] for c in candidates)
                floor = max_score * 0.85
                for i in kw_hits_idx:
                    if candidates[i]["rerank_score"] < floor:
                        candidates[i]["rerank_score"] = floor
                        print(f"    📌 关键词命中保底: chunk '{candidates[i]['text'][:40]}...' rerank提升到{floor:.3f}")

            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        except Exception as e:
            print(f"⚠️  Reranker 精排失败，降级融合分: {e}")
            candidates.sort(key=lambda x: x["score"] * 0.7 + x.get("kw_score", 0) * 0.3, reverse=True)
    else:
        # 关键词精确命中（kw_score=1.0）优先，再按向量分排
        candidates.sort(
            key=lambda x: (x.get("kw_score", 0) >= 1.0,
                           x["score"] * 0.6 + x.get("kw_score", 0) * 0.4),
            reverse=True
        )

    # 严格按 top_k 截断
    return candidates[:top_k]


# ── 统一检索入口（兼容 app.py 里的 smart_search 调用）────────

def smart_search(col, query: str, query_embedding: list[float],
                 top_k: int = 5, file_filter: Optional[str] = None,
                 category_filter: Optional[str] = None) -> tuple[list[dict], str]:
    """
    统一检索入口，返回 (hits, search_mode)。

    list 类查询（含"所有/全部/有哪些"等）适当扩大候选数，
    但无论如何严格截断到 top_k，不把大量噪音送给 LLM。

    search_mode: vector | hybrid | rerank | list_* | empty
    """
    LIST_KEYWORDS = {"所有", "全部", "清单", "列表", "汇总", "统计", "有哪些"}
    is_list_query = any(kw in query for kw in LIST_KEYWORDS)

    # list 查询最多扩到 top_k*2，上限 20
    effective_top_k = min(top_k * 2, 20) if is_list_query else top_k

    hits = multi_stage_search(
        col, query, query_embedding,
        top_k=effective_top_k,
        file_filter=file_filter,
        category_filter=category_filter,
    )

    if not hits:
        mode = "empty"
    elif get_reranker():
        mode = "rerank"
    elif any(h.get("kw_score", 0) > 0 for h in hits):
        mode = "hybrid"
    else:
        mode = "vector"

    if is_list_query:
        mode = f"list_{mode}"

    return hits, mode
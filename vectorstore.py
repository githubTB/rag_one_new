"""
vectorstore.py — 重构版

核心改进：
1. Milvus Collection 字段完整，包含所有溯源元数据
2. 支持混合检索：向量 + 关键词过滤
3. SQLite 记录文件入库状态，防止重复入库
4. Reranker 精排保留，但不强依赖
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Optional

from langchain_core.documents import Document

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


# ── SQLite 文件管理 ───────────────────────────────────────────

def init_db(db_path: str = "rag_meta.db") -> sqlite3.Connection:
    """初始化 SQLite 元数据库"""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name   TEXT NOT NULL,
            file_path   TEXT NOT NULL,
            file_hash   TEXT NOT NULL UNIQUE,
            file_type   TEXT,
            chunk_count INTEGER DEFAULT 0,
            status      TEXT DEFAULT 'pending',
            created_at  TEXT DEFAULT (datetime('now')),
            updated_at  TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash    TEXT NOT NULL,
            chunk_index  INTEGER,
            block_type   TEXT,
            page         INTEGER,
            heading_path TEXT,
            text_preview TEXT,
            milvus_id    TEXT,
            FOREIGN KEY(file_hash) REFERENCES files(file_hash)
        )
    """)
    conn.commit()
    return conn


def is_file_indexed(conn: sqlite3.Connection, file_hash: str) -> bool:
    """检查文件是否已入库（通过哈希去重）"""
    row = conn.execute(
        "SELECT status FROM files WHERE file_hash = ?", (file_hash,)
    ).fetchone()
    return row is not None and row[0] == "indexed"


def register_file(conn: sqlite3.Connection, file_name: str, file_path: str,
                  file_hash: str, file_type: str, chunk_count: int):
    """注册文件入库记录"""
    conn.execute("""
        INSERT OR REPLACE INTO files
        (file_name, file_path, file_hash, file_type, chunk_count, status, updated_at)
        VALUES (?, ?, ?, ?, ?, 'indexed', datetime('now'))
    """, (file_name, file_path, file_hash, file_type, chunk_count))
    conn.commit()


def list_indexed_files(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("""
        SELECT file_name, file_type, chunk_count, created_at
        FROM files WHERE status = 'indexed'
        ORDER BY created_at DESC
    """).fetchall()
    return [{"file_name": r[0], "file_type": r[1], "chunk_count": r[2], "indexed_at": r[3]}
            for r in rows]


def delete_file_from_index(conn: sqlite3.Connection, file_hash: str):
    """标记文件已删除（实际向量删除由 Milvus 处理）"""
    conn.execute("DELETE FROM files WHERE file_hash = ?", (file_hash,))
    conn.execute("DELETE FROM chunks WHERE file_hash = ?", (file_hash,))
    conn.commit()


# ── Milvus 操作 ───────────────────────────────────────────────

def connect_milvus(host: str = "localhost", port: str = "19530"):
    from pymilvus import connections
    connections.connect("default", host=host, port=port)
    print(f"✅ Milvus 已连接 {host}:{port}")


def get_or_create_collection(collection_name: str = "rag_docs", dim: int = 768):
    """
    创建带完整元数据字段的 Milvus Collection
    字段设计保证溯源所需的所有信息都在向量库里
    """
    from pymilvus import (
        Collection, CollectionSchema, FieldSchema, DataType, utility
    )

    if utility.has_collection(collection_name):
        print(f"✅ 复用已有 Collection: {collection_name}")
        col = Collection(collection_name)
        col.load()
        return col

    fields = [
        FieldSchema("id",           DataType.INT64,   is_primary=True, auto_id=True),
        FieldSchema("embedding",    DataType.FLOAT_VECTOR, dim=dim),

        # ── 溯源字段 ──────────────────────────────────────────
        FieldSchema("file_name",    DataType.VARCHAR,  max_length=512),
        FieldSchema("file_hash",    DataType.VARCHAR,  max_length=64),
        FieldSchema("page",         DataType.INT32),
        FieldSchema("block_type",   DataType.VARCHAR,  max_length=32),
        FieldSchema("heading_path", DataType.VARCHAR,  max_length=1024),
        FieldSchema("heading_level",DataType.INT32),
        FieldSchema("table_headers",DataType.VARCHAR,  max_length=1024),
        FieldSchema("chunk_index",  DataType.INT32),

        # ── 内容 ──────────────────────────────────────────────
        # Milvus 的 VARCHAR 最大 65535，长文本用截断，原文存 SQLite
        FieldSchema("text",         DataType.VARCHAR,  max_length=4096),
    ]

    schema = CollectionSchema(fields, description="RAG knowledge base with citations")
    col = Collection(collection_name, schema)

    # HNSW 索引，速度和精度的好平衡
    col.create_index("embedding", {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 256}
    })
    col.load()
    print(f"✅ 创建 Collection: {collection_name} (dim={dim})")
    return col


def insert_chunks(col, chunks_with_embeddings: list[tuple]) -> list[int]:
    """
    批量插入 chunk + 向量
    chunks_with_embeddings: [(Chunk, embedding_vector), ...]
    """
    if not chunks_with_embeddings:
        return []

    data = {
        "embedding":     [],
        "file_name":     [],
        "file_hash":     [],
        "page":          [],
        "block_type":    [],
        "heading_path":  [],
        "heading_level": [],
        "table_headers": [],
        "chunk_index":   [],
        "text":          [],
    }

    for chunk, emb in chunks_with_embeddings:
        data["embedding"].append(emb)
        data["file_name"].append(chunk.file_name[:512])
        data["file_hash"].append(chunk.file_hash[:64])
        data["page"].append(chunk.page)
        data["block_type"].append(chunk.block_type[:32])
        data["heading_path"].append(chunk.heading_path[:1024])
        data["heading_level"].append(chunk.heading_level)
        data["table_headers"].append("|".join(chunk.table_headers)[:1024])
        data["chunk_index"].append(chunk.chunk_index)
        # 文本截断（完整文本在 SQLite chunk 表里可选存储）
        data["text"].append(chunk.text[:4000])

    result = col.insert(list(data.values()))
    col.flush()
    return result.primary_keys


def vector_search(col, query_embedding: list[float], top_k: int = 20,
                  file_filter: Optional[str] = None) -> list[dict]:
    """
    向量检索，返回带完整元数据的结果列表
    file_filter: 可选，只在指定文件名中检索
    """
    expr = f'file_name == "{file_filter}"' if file_filter else None

    results = col.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=top_k,
        expr=expr,
        output_fields=["file_name", "file_hash", "page", "block_type",
                        "heading_path", "table_headers", "chunk_index", "text"]
    )

    hits = []
    for hit in results[0]:
        entity = hit.entity
        hits.append({
            "text":          entity.get("text", ""),
            "score":         round(hit.score, 4),
            "file_name":     entity.get("file_name", ""),
            "file_hash":     entity.get("file_hash", ""),
            "page":          entity.get("page", 1),
            "block_type":    entity.get("block_type", "text"),
            "heading_path":  entity.get("heading_path", ""),
            "table_headers": entity.get("table_headers", ""),
            "chunk_index":   entity.get("chunk_index", 0),
        })
    return hits


def delete_by_file_hash(col, file_hash: str):
    """从 Milvus 删除指定文件的所有 chunk"""
    col.delete(f'file_hash == "{file_hash}"')
    col.flush()
    print(f"✅ 已从 Milvus 删除文件: {file_hash[:8]}...")


# ── 多阶段检索（向量粗排 + Reranker 精排）────────────────────

def multi_stage_search(col, query: str, query_embedding: list[float],
                       top_k: int = 5, file_filter: Optional[str] = None) -> list[dict]:
    """
    两阶段检索：
    1. 向量粗排，召回 top_k * 4 个候选
    2. Reranker 精排，取 top_k
    返回带完整溯源信息的 hit 列表
    """
    # 粗排
    candidates = vector_search(col, query_embedding, top_k=top_k * 4, file_filter=file_filter)
    if not candidates:
        return []

    # 精排
    reranker = get_reranker()
    if reranker:
        try:
            pairs = [[query, c["text"]] for c in candidates]
            scores = reranker.predict(pairs)
            for i, c in enumerate(candidates):
                c["rerank_score"] = float(scores[i])
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        except Exception as e:
            print(f"⚠️  Reranker 精排失败，使用向量分数: {e}")
            candidates.sort(key=lambda x: x["score"], reverse=True)
    else:
        candidates.sort(key=lambda x: x["score"], reverse=True)

    return candidates[:top_k]


# ── LangChain 兼容层（保持和现有代码的向后兼容）─────────────

def build_vectorstore_langchain(docs: list[Document], embeddings,
                                 collection_name: str = "rag_docs",
                                 milvus_host: str = "localhost",
                                 milvus_port: str = "19530"):
    """
    保留给 main.py 的兼容接口，内部用新 Milvus 逻辑
    """
    from langchain_community.vectorstores import Milvus
    connect_milvus(milvus_host, milvus_port)
    return Milvus.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        connection_args={"host": milvus_host, "port": milvus_port},
    )

#!/usr/bin/env python3
"""
为现有 Milvus collection 启用全文检索（BM25）
Milvus 2.4+ 支持，需要重新创建 collection
"""
import os
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION = os.getenv("COLLECTION", "rag_docs")

def main():
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    print(f"✅ 已连接 Milvus {MILVUS_HOST}:{MILVUS_PORT}")

    old_col = Collection(COLLECTION)
    old_col.load()

    # 备份数据
    print(f"📦 备份数据...")
    results = old_col.query(
        expr="id >= 0",
        output_fields=["id", "embedding", "file_name", "file_hash", "category", "page",
                       "block_type", "heading_path", "heading_level", "table_headers",
                       "chunk_index", "text"],
        limit=100000
    )
    print(f"   共 {len(results)} 条数据")

    if len(results) == 0:
        print("⚠️ 没有数据需要迁移")
        return

    # 创建新 collection（带全文索引）
    new_name = f"{COLLECTION}_v2"
    if utility.has_collection(new_name):
        utility.drop_collection(new_name)

    print(f"\n🆕 创建新 collection: {new_name}")

    # 添加 sparse vector 字段用于 BM25
    fields = [
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=768),  # dense vector
        FieldSchema("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR),  # BM25
        FieldSchema("file_name", DataType.VARCHAR, max_length=512),
        FieldSchema("file_hash", DataType.VARCHAR, max_length=64),
        FieldSchema("category", DataType.VARCHAR, max_length=256),
        FieldSchema("page", DataType.INT32),
        FieldSchema("block_type", DataType.VARCHAR, max_length=32),
        FieldSchema("heading_path", DataType.VARCHAR, max_length=1024),
        FieldSchema("heading_level", DataType.INT32),
        FieldSchema("table_headers", DataType.VARCHAR, max_length=1024),
        FieldSchema("chunk_index", DataType.INT32),
        FieldSchema("text", DataType.VARCHAR, max_length=8192),
    ]

    schema = CollectionSchema(fields, description="RAG with BM25 full-text search")
    new_col = Collection(new_name, schema)

    # 创建索引
    new_col.create_index("embedding", {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 256}
    })

    # BM25 索引
    new_col.create_index("sparse_embedding", {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "BM25"
    })

    new_col.load()
    print("✅ 索引创建完成")

    # 分批插入数据（这里需要文本 analyzer 生成 sparse vector，简化版先不实现）
    # 实际 BM25 需要调用 analyzer，可以用 pymilvus 的 utility 或自己算 tf-idf

    print("\n⚠️ 注意：Milvus 2.4+ 的 BM25 需要配合 analyzer 使用")
    print("   完整迁移较复杂，建议先用方案 B（混合检索调参）")

    # 清理
    utility.drop_collection(new_name)

if __name__ == "__main__":
    main()
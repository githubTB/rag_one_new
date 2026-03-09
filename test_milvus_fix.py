#!/usr/bin/env python3
"""
测试Milvus文本字段长度限制修复
"""

import sys
from vectorstore import build_vectorstore
from embedder import embeddings
from langchain_core.documents import Document

# 创建一个长文本测试文档
def create_test_doc():
    # 创建一个长度为80000的文本
    long_text = "测试文本 " * 40000  # 80000字符
    print(f"创建测试文档，长度: {len(long_text)}字符")
    
    return [Document(
        page_content=long_text,
        metadata={"source": "test.txt", "type": "Text"}
    )]

# 测试Milvus向量库构建
def test_milvus_build():
    print("=== 测试Milvus向量库构建 ===")
    
    try:
        # 创建测试文档
        docs = create_test_doc()
        
        # 构建向量库
        vectorstore = build_vectorstore(
            docs,
            embeddings,
            vector_store_type="milvus",
            milvus_host="localhost",
            milvus_port="19530"
        )
        
        print("✅ Milvus向量库构建成功！")
        return True
    except Exception as e:
        print(f"❌ Milvus向量库构建失败: {e}")
        return False

if __name__ == "__main__":
    success = test_milvus_build()
    sys.exit(0 if success else 1)

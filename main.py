from vectorstore import connect_milvus, build_vectorstore, query_milvus_vector, multi_stage_search
from embedder import embeddings
from pymilvus import utility
from pipeline import unstructured_splitter, smart_chunk, split_by_heading

if __name__ == "__main__":
    # docs = query_milvus_vector("能源管理", 5, embeddings)
    # for i, d in enumerate(docs):
    #     print(f"文档 {i + 1}:", d)

    # 数据目录
    data_dir = "data/"

    # 1. 加载文档
    docs = unstructured_splitter(data_dir)

    # 2. 切片
    splits = split_by_heading(docs)
    # splits = smart_chunk(docs)

    # 3. 连接 Milvus
    connect_milvus()

    # 删除历史数据
    collection_name = "rag_docs"
    if collection_name in utility.list_collections():
        utility.drop_collection(collection_name)

    # 4. 写入向量库
    vectorstore = build_vectorstore(splits, embeddings)

    # 5. 搜索 top-k
    top_k = 5
    query = "能源管理"
    # docs = vectorstore.similarity_search(query, k=top_k)
    docs = multi_stage_search(vectorstore, query, top_k)

    # 6. 输出搜索结果
    for i, doc in enumerate(docs):
        print(f"--- 文档 {i + 1} ---")
        print(doc.page_content)

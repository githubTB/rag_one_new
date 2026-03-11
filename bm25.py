"""
bm25.py — BM25 全文检索实现

实现了 BM25 (Best Match 25) 算法，用于全文检索
支持：
- 词频统计
- 逆文档频率计算
- 短语匹配
- 布尔逻辑
"""

import math
import re
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """初始化 BM25
        
        Args:
            k1: 词频饱和度参数，控制词频对得分的影响
            b: 文档长度归一化参数，0.75 是常见值
        """
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.inverted_index = defaultdict(list)  # 词 -> [(文档ID, 词频)]
        self.document_freq = defaultdict(int)  # 词 -> 包含该词的文档数
        self.total_documents = 0
    
    def add_document(self, doc_id: int, text: str):
        """添加文档到索引
        
        Args:
            doc_id: 文档唯一标识
            text: 文档文本内容
        """
        # 分词
        tokens = self._tokenize(text)
        if not tokens:
            return
        
        # 计算词频
        freq = Counter(tokens)
        doc_length = len(tokens)
        
        # 更新索引
        for term, count in freq.items():
            self.inverted_index[term].append((doc_id, count))
            self.document_freq[term] += 1
        
        # 更新文档信息
        self.documents.append((doc_id, text))
        self.doc_lengths.append(doc_length)
        self.total_documents += 1
        
        # 更新平均文档长度
        self.avg_doc_length = sum(self.doc_lengths) / self.total_documents
    
    def _tokenize(self, text: str) -> List[str]:
        """分词
        
        Args:
            text: 文本内容
            
        Returns:
            分词后的词列表
        """
        # 移除标点，转换小写
        text = re.sub(r'[\s\u3000]+', ' ', text)
        text = re.sub(r'[\p{P}\p{S}]+', ' ', text, flags=re.UNICODE)
        text = text.lower()
        
        # 分词（简单的基于空格分词，可根据需要替换为更复杂的分词器）
        tokens = text.split()
        
        # 过滤停用词
        stop_words = {
            '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', 'that', 'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'with',
            'on', 'at', 'by', 'from', 'as', 'but', 'or', 'if', 'so', 'can', 'will', 'would'
        }
        
        return [token for token in tokens if token and token not in stop_words]
    
    def calculate_score(self, doc_id: int, query: List[str]) -> float:
        """计算文档对查询的 BM25 得分
        
        Args:
            doc_id: 文档ID
            query: 查询分词后的词列表
            
        Returns:
            BM25 得分
        """
        score = 0.0
        doc_length = self.doc_lengths[doc_id]
        
        for term in query:
            if term not in self.inverted_index:
                continue
            
            # 计算逆文档频率 (IDF)
            df = self.document_freq[term]
            idf = math.log((self.total_documents - df + 0.5) / (df + 0.5) + 1.0)
            
            # 找到该文档中该词的词频
            term_freq = 0
            for (did, tf) in self.inverted_index[term]:
                if did == doc_id:
                    term_freq = tf
                    break
            
            # 计算 BM25 得分
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """搜索查询
        
        Args:
            query: 查询文本
            top_k: 返回前 k 个结果
            
        Returns:
            列表，元素为 (文档ID, 得分)
        """
        # 分词查询
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # 收集所有可能相关的文档
        candidate_docs = set()
        for term in query_tokens:
            if term in self.inverted_index:
                for (doc_id, _) in self.inverted_index[term]:
                    candidate_docs.add(doc_id)
        
        # 计算每个候选文档的得分
        scores = []
        for doc_id in candidate_docs:
            score = self.calculate_score(doc_id, query_tokens)
            if score > 0:
                scores.append((doc_id, score))
        
        # 按得分排序并返回前 top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def get_document(self, doc_id: int) -> Optional[str]:
        """获取文档内容
        
        Args:
            doc_id: 文档ID
            
        Returns:
            文档文本内容
        """
        if 0 <= doc_id < len(self.documents):
            return self.documents[doc_id][1]
        return None


def build_bm25_index(chunks: List[Dict]) -> BM25:
    """从 chunks 构建 BM25 索引
    
    Args:
        chunks: 文档 chunks 列表，每个 chunk 包含 'text' 字段
        
    Returns:
        构建好的 BM25 索引
    """
    bm25 = BM25()
    
    for i, chunk in enumerate(chunks):
        text = chunk.get('text', '')
        bm25.add_document(i, text)
    
    return bm25


def bm25_search(col, query: str, top_k: int = 20, 
                file_filter: Optional[str] = None, 
                category_filter: Optional[str] = None) -> List[Dict]:
    """执行 BM25 搜索
    
    Args:
        col: Milvus 集合
        query: 查询文本
        top_k: 返回前 k 个结果
        file_filter: 文件名过滤
        category_filter: 分类过滤
        
    Returns:
        搜索结果列表
    """
    # 构建查询表达式
    exprs = []
    if file_filter:
        exprs.append(f'file_name == "{file_filter}"')
    if category_filter:
        exprs.append(f'category == "{category_filter}"')
    expr = " and ".join(exprs) if exprs else None
    
    # 从 Milvus 获取所有可能的文档
    try:
        results = col.query(
            expr=expr,
            output_fields=["file_name", "file_hash", "category", "page", "block_type",
                          "heading_path", "table_headers", "chunk_index", "text"],
            limit=1000  # 限制最大返回数量
        )
    except Exception as e:
        print(f"⚠️  BM25 搜索失败: {e}")
        return []
    
    if not results:
        return []
    
    # 构建 BM25 索引
    bm25 = build_bm25_index(results)
    
    # 执行搜索
    bm25_results = bm25.search(query, top_k=top_k)
    
    # 转换结果格式
    hits = []
    for doc_id, score in bm25_results:
        if 0 <= doc_id < len(results):
            hit = results[doc_id].copy()
            hit['score'] = round(score, 4)
            hit['bm25_score'] = round(score, 4)
            hits.append(hit)
    
    return hits
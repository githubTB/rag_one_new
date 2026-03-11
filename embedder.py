"""
embedder.py — 单例 Embedding 模型

推荐模型（按效果排序）：
  中文：BAAI/bge-m3（多语言，效果最好）
        BAAI/bge-base-zh-v1.5（轻量，默认）
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"

_embedder = None


def get_embedder(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """获取 Embedding 模型单例"""
    global _embedder
    if _embedder is None:
        print(f"📦 加载 Embedding 模型: {model_name}")
        _embedder = SentenceTransformer(model_name, device="cpu")
        print(f"✅ Embedding 模型已加载 (dim={_embedder.get_sentence_embedding_dimension()})")
    return _embedder


class _LangChainCompatEmbedder:
    """向后兼容接口，保留 embed_documents / embed_query"""
    def __init__(self):
        self._model = None

    def _get(self):
        if self._model is None:
            self._model = get_embedder()
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get().encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._get().encode(text, normalize_embeddings=True).tolist()


embeddings = _LangChainCompatEmbedder()
"""
embedder.py — 重构版

改为单例模式，避免重复加载模型
支持多种模型，中文场景推荐 bge-m3
"""

from sentence_transformers import SentenceTransformer

# 推荐模型（按效果排序）：
#   中文：BAAI/bge-m3（多语言，效果最好）
#         BAAI/bge-base-zh-v1.5（轻量，原项目在用）
#   英文：BAAI/bge-large-en-v1.5
MODEL_NAME = "BAAI/bge-base-zh-v1.5"

_embedder = None


def get_embedder(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """获取 Embedding 模型单例"""
    global _embedder
    if _embedder is None:
        print(f"📦 加载 Embedding 模型: {model_name}")
        _embedder = SentenceTransformer(model_name, device="cpu")
        print(f"✅ Embedding 模型已加载 (dim={_embedder.get_sentence_embedding_dimension()})")
    return _embedder


# ── 向后兼容：保留原 embeddings 变量给 main.py 用 ────────────
class _LangChainCompatEmbedder:
    """让旧代码的 embeddings.embed_documents() 接口继续工作"""
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

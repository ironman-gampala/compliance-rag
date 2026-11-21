"""Load the embedding model once and turn text into vectors."""

from rag.config import EMBED_MODEL, QUERY_PREFIX, RERANK_MODEL

_embedder = None
_reranker = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def embed_passages(texts: list[str], show_progress: bool = False):
    if not texts:
        return []
    vectors = get_embedder().encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return vectors.tolist()


def embed_query(question: str):
    vector = get_embedder().encode(
        [QUERY_PREFIX + question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector[0].tolist()

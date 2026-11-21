"""Local Chroma collection. Embeddings are computed outside Chroma."""

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings
from chromadb.utils.embedding_functions import register_embedding_function

from rag.config import CHROMA_PATH, COLLECTION_NAME
from rag.types import Chunk


@register_embedding_function
class PassthroughEmbedding(EmbeddingFunction[Documents]):
    """Chroma requires an embedding function. This one is never called.

    Ingest and query compute bge-small vectors themselves and pass them in.
    """

    def __call__(self, input: Documents) -> Embeddings:
        raise RuntimeError("Pass embeddings explicitly; this collection does not embed text.")

    @staticmethod
    def name() -> str:
        return "passthrough"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "PassthroughEmbedding":
        return PassthroughEmbedding()

    def default_space(self) -> str:
        return "cosine"


def get_client():
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(client=None):
    client = client or get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=PassthroughEmbedding(),
    )


def reset_collection(client=None) -> None:
    client = client or get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        # A missing collection is the state reset is supposed to produce.
        pass


def delete_source(source: str, collection=None) -> None:
    collection = collection or get_collection()
    existing = collection.get(where={"source": source}, include=["metadatas"])
    if existing["ids"]:
        collection.delete(ids=existing["ids"])


def add_chunks(chunks: list[Chunk], embeddings: list[list[float]], collection=None) -> None:
    if not chunks:
        return
    collection = collection or get_collection()
    batch = 200
    for start in range(0, len(chunks), batch):
        group = chunks[start : start + batch]
        vectors = embeddings[start : start + batch]
        collection.add(
            ids=[chunk.id for chunk in group],
            documents=[chunk.text for chunk in group],
            embeddings=vectors,
            metadatas=[
                {
                    "source": chunk.source,
                    "page": chunk.page,
                    "region": chunk.region,
                    "doc_type": chunk.doc_type,
                }
                for chunk in group
            ],
        )


def _row_to_chunk(chunk_id, text, meta, score=None) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        source=meta["source"],
        page=int(meta["page"]),
        region=meta.get("region") or "",
        doc_type=meta.get("doc_type") or "",
        score=score,
    )


def vector_query(query_embedding: list[float], k: int, where: dict | None = None) -> list[Chunk]:
    collection = get_collection()
    count = collection.count()
    if count == 0 or k <= 0:
        return []
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(k, count),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    result = collection.query(**kwargs)
    hits = []
    for chunk_id, text, meta, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        # Chroma cosine space returns distance = 1 - cosine similarity.
        similarity = 1.0 - float(distance)
        hits.append(_row_to_chunk(chunk_id, text, meta, score=similarity))
    return hits


def all_chunks(where: dict | None = None) -> list[Chunk]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    kwargs = {"include": ["documents", "metadatas"]}
    if where:
        kwargs["where"] = where
    result = collection.get(**kwargs)
    return [
        _row_to_chunk(chunk_id, text, meta)
        for chunk_id, text, meta in zip(result["ids"], result["documents"], result["metadatas"])
    ]

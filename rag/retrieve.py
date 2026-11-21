"""Vector search, keyword search, and the reranker."""

import re

from rag.config import CANDIDATE_POOL
from rag.embeddings import embed_query, get_reranker
from rag.store import all_chunks, vector_query
from rag.types import Chunk


def build_where(region: str | None = None, doc_type: str | None = None) -> dict | None:
    clauses = []
    if region:
        clauses.append({"region": region})
    if doc_type:
        clauses.append({"doc_type": doc_type})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Merge ranked id lists. A document near the top of several lists wins.

    Raw BM25 scores and cosine similarities are not on the same scale, so this
    uses rank instead of the score. `k` dampens the lead of rank 1.
    """
    scores: dict[str, float] = {}
    order: list[str] = []
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            if doc_id not in scores:
                order.append(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(order, key=lambda doc_id: scores[doc_id], reverse=True)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_search(question: str, k: int, where: dict | None = None) -> list[Chunk]:
    corpus = all_chunks(where)
    if not corpus or k <= 0:
        return []
    from rank_bm25 import BM25Okapi

    tokenized = [_tokenize(chunk.text) for chunk in corpus]
    scores = BM25Okapi(tokenized).get_scores(_tokenize(question))
    ranked = sorted(zip(corpus, scores), key=lambda pair: float(pair[1]), reverse=True)
    hits = []
    for chunk, score in ranked[:k]:
        chunk.score = float(score)
        if chunk.score <= 0:
            continue
        hits.append(chunk)
    return hits


_SECTION = re.compile(r"\d{3,4}\.\d+(?:\([a-z0-9]+\))*", re.IGNORECASE)


def section_labels(text: str) -> set[str]:
    return {match.group(0).lower() for match in _SECTION.finditer(text)}


_TOPIC_WORDS = ("day", "late", "timely", "weekend", "holiday", "business", "statement", "cutoff", "p.m", "due")


def _window_around_section(text: str, match: re.Match[str], before: int = 250, after: int = 500) -> str:
    """Text beside one section cite, stopping before the next section number."""
    nxt = _SECTION.search(text, match.end())
    end = match.end() + after
    if nxt:
        end = min(end, nxt.start())
    return text[max(0, match.start() - before) : end]


def section_bases_near_topics(text: str, topics: tuple[str, ...] | list[str]) -> set[str]:
    """Section numbers whose own paragraph uses at least two words from the question."""
    lowered = text.lower()
    bases = set()
    for match in _SECTION.finditer(lowered):
        window = _window_around_section(lowered, match)
        if sum(1 for topic in topics if topic in window) >= 2:
            bases.add(match.group(0).split("(", 1)[0])
    return bases


def expand_related_subsections(
    question: str,
    selected: list[Chunk],
    where: dict | None,
    limit: int = 2,
) -> list[Chunk]:
    """Add the examination paragraph that states a fuller subsection.

    A narrative chunk often says 'section 1026.5(b)' while the exact cite
    1026.5(b)(2)(ii) is a later page the reranker never keeps. Without that
    page the model cannot cite the subsection and refuses to answer.
    """
    topics = [word for word in _TOPIC_WORDS if word in question.lower()]
    if not topics or not selected:
        return []
    bases: set[str] = set()
    for hit in selected:
        bases |= section_bases_near_topics(hit.text, topics)
    if not bases:
        return []
    have_ids = {hit.id for hit in selected}
    have_labels: set[str] = set()
    for hit in selected:
        have_labels |= section_labels(hit.text)
    best_by_base: dict[str, tuple[int, Chunk, int]] = {}
    for chunk in all_chunks(where):
        if chunk.id in have_ids:
            continue
        lowered = chunk.text.lower()
        hits_for_base: dict[str, tuple[int, int]] = {}
        for match in _SECTION.finditer(lowered):
            label = match.group(0)
            base = label.split("(", 1)[0]
            if "(" not in label or base not in bases or label in have_labels:
                continue
            window = _window_around_section(lowered, match)
            topic_hits = sum(1 for topic in topics if topic in window)
            depth = label.count("(")
            current = hits_for_base.get(base)
            if current is None or (depth, topic_hits) > current:
                hits_for_base[base] = (depth, topic_hits)
        for base, (depth, topic_hits) in hits_for_base.items():
            current = best_by_base.get(base)
            if current is None or (depth, topic_hits) > (current[0], current[2]):
                best_by_base[base] = (depth, chunk, topic_hits)
    ranked = sorted(best_by_base.values(), key=lambda item: (item[0], item[2]), reverse=True)
    return [chunk for depth, chunk, topic_hits in ranked if topic_hits >= 2][:limit]


def supplement_with_specific_sections(
    selected: list[Chunk],
    fused: list[Chunk],
    limit: int = 2,
    head: int = 8,
) -> list[Chunk]:
    """Keep a more specific subsection the reranker pushed just below the cutoff.

    The narrative of a rule often outranks the examination paragraph that states
    the subsection, for example 1026.52(b) versus 1026.52(b)(2)(i)(A). The model
    cannot cite or apply a limit that never reaches it.
    """
    selected_labels: set[str] = set()
    for hit in selected:
        selected_labels |= section_labels(hit.text)
    selected_bases = {label.split("(", 1)[0] for label in selected_labels}
    chosen = list(selected)
    chosen_ids = {hit.id for hit in chosen}
    added = 0
    for hit in fused[:head]:
        if hit.id in chosen_ids:
            continue
        labels = section_labels(hit.text)
        more_specific = [
            label
            for label in labels
            if "(" in label and label.split("(", 1)[0] in selected_bases and label not in selected_labels
        ]
        if not more_specific:
            continue
        chosen.append(hit)
        chosen_ids.add(hit.id)
        added += 1
        if added >= limit:
            break
    return chosen


def rerank_hits(question: str, hits: list[Chunk]) -> list[Chunk]:
    if not hits:
        return []
    model = get_reranker()
    pairs = [(question, hit.text) for hit in hits]
    scores = model.predict(pairs)
    ranked = sorted(zip(hits, scores), key=lambda pair: float(pair[1]), reverse=True)
    ordered = []
    for hit, score in ranked:
        hit.score = float(score)
        ordered.append(hit)
    return ordered


def search(
    question: str,
    k: int = 5,
    region: str | None = None,
    doc_type: str | None = None,
    hybrid: bool = False,
    rerank: bool = False,
) -> list[Chunk]:
    """Return the passages the model is allowed to use.

    Default is vector search only (step 1). `--hybrid` adds BM25 and fuses the
    two lists. `--rerank` rescores the shortlist with a cross-encoder.
    """
    where = build_where(region, doc_type)
    pool = max(k, CANDIDATE_POOL) if (hybrid or rerank) else k
    vector_hits = vector_query(embed_query(question), pool, where)

    def with_related(hits: list[Chunk], fused: list[Chunk] | None = None) -> list[Chunk]:
        chosen = hits
        if fused:
            chosen = supplement_with_specific_sections(chosen, fused)
        return chosen + expand_related_subsections(question, chosen, where)

    if not hybrid and not rerank:
        return with_related(vector_hits[:k])

    if hybrid:
        keyword_hits = bm25_search(question, pool, where)
        fused_ids = reciprocal_rank_fusion(
            [[hit.id for hit in vector_hits], [hit.id for hit in keyword_hits]]
        )
        by_id = {hit.id: hit for hit in vector_hits + keyword_hits}
        ordered = [by_id[doc_id] for doc_id in fused_ids if doc_id in by_id]
    else:
        ordered = vector_hits

    fused = ordered
    if rerank:
        selected = rerank_hits(question, fused[:pool])[:k]
        return with_related(selected, fused)
    return with_related(fused[:k], fused)

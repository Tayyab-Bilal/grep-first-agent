"""Weighted reciprocal-rank fusion for the one place vectors help: searching inside document text."""

from __future__ import annotations

from typing import Any


def weighted_rrf(
    rankings: dict[str, list[str]], weights: dict[str, float], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked id lists; each list contributes weight / (k + rank), rank starting at 1."""
    scores: dict[str, float] = {}
    for name, ids in rankings.items():
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weights.get(name, 1.0) / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def hybrid_content_search(
    keyword_hits: list[dict[str, Any]],
    dense_hits: list[dict[str, Any]],
    keyword_weight: float = 3,
    dense_weight: float = 1,
    dense_min: float = 0.45,
    relevance_floor: float = 0.20,
    k: int = 60,
) -> list[dict[str, Any]]:
    """Chunk hits in, one result per document out. Lexical leads (3:1).

    Hits are {"doc_id", "snippet"} (dense hits also carry "score", a cosine). Dense hits under
    `dense_min` are noise and dropped. Fused scores are normalised so 1.0 means "rank 1 in both
    lists"; documents under `relevance_floor` are cut. Dense hits are passed in: no embedding here.
    """
    dense_hits = [h for h in dense_hits if h["score"] >= dense_min]
    rankings: dict[str, list[str]] = {}
    snippets: dict[str, list[str]] = {}
    for name, hits in (("keyword", keyword_hits), ("dense", dense_hits)):
        ranked: list[str] = []
        for h in hits:  # a document's rank is its best chunk's rank
            if h["doc_id"] not in ranked:
                ranked.append(h["doc_id"])
            snips = snippets.setdefault(h["doc_id"], [])
            if h["snippet"] not in snips:
                snips.append(h["snippet"])
        rankings[name] = ranked
    weights = {"keyword": keyword_weight, "dense": dense_weight}
    best_possible = (keyword_weight + dense_weight) / (k + 1)
    return [
        {"doc_id": d, "score": s / best_possible, "snippets": snippets[d][:2]}
        for d, s in weighted_rrf(rankings, weights, k)
        if s / best_possible >= relevance_floor
    ]

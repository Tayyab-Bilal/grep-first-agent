"""Hybrid search inside document text: the one place vectors help.

The two indexes are fakes. In production the keyword side and the dense side were separate
searches against a vector database; here they are small in-memory classes so the tests are
deterministic. Both are *shared* across users, so their hits must be filtered to the hierarchy.
"""

from __future__ import annotations

import asyncio
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol

from .matching import STOPWORDS, tokenize
from .seed import SEED


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    text: str


class KeywordIndex(Protocol):
    async def search(self, query: str) -> list[dict[str, Any]]: ...  # [{"doc_id", "snippet"}]


class DenseIndex(Protocol):
    async def search(self, query: str) -> list[dict[str, Any]]: ...  # [{"doc_id", "snippet", "score"}]


def default_chunks() -> list[Chunk]:
    """One chunk per sentence of every seeded document, for every user (a shared index)."""
    return [
        Chunk(d["id"], sentence.rstrip(".") + ".")
        for user in SEED.values()
        for d in user["document"]
        for sentence in d["content"].split(". ")
    ]


class FakeKeywordIndex:
    """Ranks chunks by how many query words they contain literally."""

    def __init__(self, chunks: list[Chunk], delay: float = 0.0, events: list[str] | None = None) -> None:
        self.chunks, self.delay, self.events = chunks, delay, events

    async def search(self, query: str) -> list[dict[str, Any]]:
        if self.events is not None:
            self.events.append("keyword:start")
        await asyncio.sleep(self.delay)
        words = set(tokenize(query)) - STOPWORDS
        scored = [(len(words & set(tokenize(c.text))), c) for c in self.chunks]
        if self.events is not None:
            self.events.append("keyword:end")
        return [{"doc_id": c.doc_id, "snippet": c.text} for n, c in sorted(scored, key=lambda x: -x[0]) if n]


# A stand-in for "meaning": words a real embedding model would place close together.
SYNONYMS = {"outage": "incident", "breakdown": "incident", "pager": "call", "costs": "price"}
_DENSE_STOP = STOPWORDS | {"are", "by", "is", "at"}


def _vector(text: str) -> Counter[str]:
    words = [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in tokenize(text) if t not in _DENSE_STOP]
    return Counter(SYNONYMS.get(w, w) for w in words)


class FakeDenseIndex:
    """Cosine similarity over synonym-folded word counts. Finds `outage` -> `incidents`."""

    def __init__(self, chunks: list[Chunk], delay: float = 0.0, events: list[str] | None = None) -> None:
        self.chunks, self.delay, self.events = chunks, delay, events

    async def search(self, query: str) -> list[dict[str, Any]]:
        if self.events is not None:
            self.events.append("dense:start")
        await asyncio.sleep(self.delay)
        q = _vector(query)
        hits = []
        for c in self.chunks:
            v = _vector(c.text)
            dot = sum(q[w] * v[w] for w in q)
            if dot:
                score = dot / (math.sqrt(sum(x * x for x in q.values())) * math.sqrt(sum(x * x for x in v.values())))
                hits.append({"doc_id": c.doc_id, "snippet": c.text, "score": round(score, 3)})
        if self.events is not None:
            self.events.append("dense:end")
        return sorted(hits, key=lambda h: -h["score"])

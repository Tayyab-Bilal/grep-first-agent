"""Pure string matching: a cheap cascade that finds literal identifiers embeddings blur."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

PROSE_CAP = 200
_WORD = re.compile(r"\w+", re.UNICODE)  # \w is Unicode-aware, so Arabic script tokenizes too


# Filler words from vague questions; counting them as hits makes every long text "match".
# Simplification: English-only list; per-language lists or IDF weighting if more locales matter.
STOPWORDS = frozenset("the a an of to in on for and or my me about everything all with from".split())


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass(frozen=True)
class MatchScore:
    phrase_hit: bool
    token_hits: int
    fuzzy: float

    @property
    def key(self) -> tuple[bool, int, float]:
        # Tuple, not a weighted sum: a long text with many soft hits must not outrank an exact one.
        return (self.phrase_hit, self.token_hits, self.fuzzy)


def _fuzzy(query: str, q_tokens: list[str], field: str) -> float:
    # Simplification: whole-field ratio on the first 200 chars only (SequenceMatcher is quadratic);
    # swap in rapidfuzz if fields get large.
    f_tokens = tokenize(field)
    best = SequenceMatcher(None, query, field.lower()[:PROSE_CAP]).ratio()
    n = len(q_tokens)
    for i in range(max(len(f_tokens) - n + 1, 0)):
        window = " ".join(f_tokens[i : i + n])
        best = max(best, SequenceMatcher(None, " ".join(q_tokens), window).ratio())
    return best


def score(query: str, fields: list[str]) -> MatchScore:
    q = query.lower().strip()
    q_tokens = tokenize(q)
    if not q_tokens:
        return MatchScore(False, 0, 0.0)
    phrase_hit = any(q in f.lower() for f in fields)
    field_tokens = {t for f in fields for t in tokenize(f)}
    token_hits = sum(1 for t in set(q_tokens) - STOPWORDS if t in field_tokens)
    fuzzy = max((_fuzzy(q, q_tokens, f) for f in fields), default=0.0)
    return MatchScore(phrase_hit, token_hits, fuzzy)


def search(
    query: str,
    records: list[dict[str, Any]],
    fields: list[str],
    fuzzy_threshold: float = 0.6,
    limit: int = 10,
) -> list[tuple[dict[str, Any], MatchScore]]:
    """Return (record, score) best first. Prose fields are capped so results stay prompt-sized."""
    hits = []
    for rec in records:
        s = score(query, [str(rec.get(f, "")) for f in fields])
        if s.phrase_hit or s.token_hits or s.fuzzy >= fuzzy_threshold:
            capped = dict(rec)
            for f in fields:
                if isinstance(capped.get(f), str):
                    capped[f] = capped[f][:PROSE_CAP]
            hits.append((capped, s))
    hits.sort(key=lambda h: h[1].key, reverse=True)
    return hits[:limit]

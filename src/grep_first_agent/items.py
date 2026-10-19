"""The compact pointer shape every tool returns, and how the agent reads tool output."""

from __future__ import annotations

from typing import Any

from .matching import MatchScore


def why(s: MatchScore) -> str:
    if s.phrase_hit:
        return "exact phrase"
    if s.token_hits:
        return f"{s.token_hits} word match"
    return f"fuzzy {s.fuzzy:.2f}"


def make_item(kind: str, rec: dict[str, Any], s: MatchScore, fields: list[str], out_kind: str | None = None) -> dict[str, Any]:
    """`fields[0]` is the display name; the first other non-empty field becomes the snippet."""
    snippet = next((str(rec[f]) for f in fields[1:] if rec.get(f)), "")
    item = {"id": rec["id"], "kind": out_kind or kind, "name": rec["name"], "snippet": snippet, "why": why(s)}
    if rec.get("path"):
        item["path"] = rec["path"]  # breadcrumb, used to word clarifying questions
    return item


def rows_of(parsed: Any) -> list[dict[str, Any]]:
    """Tool output is a list, or an object with `items` (pointers) or `candidates` (resolver ids)."""
    if isinstance(parsed, list):
        return parsed
    return parsed.get("items", parsed.get("candidates", []))

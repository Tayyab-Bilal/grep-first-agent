"""The scout's output contract: pointers only, never an answer for the user."""

from __future__ import annotations

import json
from typing import Any, Literal, get_args

from pydantic import BaseModel, ValidationError


class ContractError(ValueError):
    pass


# The nine buckets a pointer may live in. Anything else is rejected, so a model that invents a
# bucket ("people", "tasks") fails validation and gets the repair turn instead of corrupting callers.
Kind = Literal["task", "project", "document", "email", "meeting", "chat", "workflow", "kpi", "violation"]
KINDS: tuple[str, ...] = get_args(Kind)


class Pointer(BaseModel):
    id: str
    name: str
    confidence: float = 1.0


class DiscoveryResult(BaseModel):
    intent: str
    scope: str | None = None
    pointers: dict[Kind, list[Pointer]] = {}
    confidence: float = 0.0
    ambiguity: str | None = None
    incomplete: list[Kind] = []  # kinds whose list may be partial: never present as the whole set


def _text(raw: str | list[Any]) -> str:
    if isinstance(raw, str):
        return raw
    # Some providers return content blocks: [{"type": "text", "text": "..."}]
    return "\n".join(b["text"] if isinstance(b, dict) else str(b) for b in raw
                     if not isinstance(b, dict) or b.get("type") == "text")


def parse_result(raw: str | list[Any]) -> DiscoveryResult:
    """Take the first valid JSON object found anywhere in the text; models love to add prose."""
    text = _text(raw)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
            return DiscoveryResult.model_validate(obj)
        except (ValueError, ValidationError):
            continue
    raise ContractError("no valid DiscoveryResult JSON object found")

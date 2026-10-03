"""Two guards: impersonation and telemetry scrubbing."""

from __future__ import annotations

import json
import re
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Identity:
    tenant: str
    user: str


# Who the surrounding request really belongs to. A web layer sets this; tools only read it.
ambient_identity: ContextVar[Identity | None] = ContextVar("ambient_identity", default=None)


def check_impersonation(claimed: Identity) -> None:
    """An impersonated call (a background job acting as a user) only proceeds when the ambient
    tenant and user are exactly who the call claims to be. Otherwise a mis-wired job could read
    another user's data with a valid token."""
    if ambient_identity.get() != claimed:
        raise PermissionError("impersonated call does not match the ambient tenant and user")


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        # Snippets are record text (PII-heavy); drop them. Everything else keeps its shape.
        return {k: _scrub(v) for k, v in value.items() if k != "snippet"}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, str):
        return _EMAIL.sub("[email]", value)
    return value


def scrub_tool_result(raw: str) -> str:
    """Make a tool result safe to log: no snippets, no email addresses."""
    try:
        return json.dumps(_scrub(json.loads(raw)))
    except ValueError:  # not JSON (an error string): still mask addresses
        return _EMAIL.sub("[email]", raw)

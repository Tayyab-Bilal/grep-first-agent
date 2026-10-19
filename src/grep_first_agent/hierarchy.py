"""Hierarchy filter: results are limited to the projects the user is allowed to see.

The rule is "drop on miss". A record whose project is outside the hierarchy, or that has no
project at all, is hidden. Showing it "just in case" was the original cross-hierarchy leak.
"""

from __future__ import annotations

from typing import Any

# Kinds that live inside a project. Emails and group chats belong to the user, not a project.
SCOPED = frozenset({"task", "project", "document", "meeting", "kpi", "violation", "workflow"})


def is_visible(kind: str, record: dict[str, Any], allowed: set[str]) -> bool:
    if kind not in SCOPED:
        return True
    project_id = record.get("id") if kind == "project" else record.get("project_id")
    return project_id in allowed  # None (no project) is never in the set: dropped


def visible(kind: str, records: list[dict[str, Any]], allowed: set[str]) -> list[dict[str, Any]]:
    return [r for r in records if is_visible(kind, r, allowed)]

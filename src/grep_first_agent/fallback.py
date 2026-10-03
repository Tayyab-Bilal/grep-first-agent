"""Layer 2 and 3 of the tool set: fallback snapshots and shared name-resolution.

Snapshots are for vague requests where no typed search fits ("what is going on in Operations?").
Each one fans out 2-5 backend calls in parallel. Resolvers turn a spoken name into an id.
All of them apply the same hierarchy filter as the typed searches.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .backend import WorkspaceAPI
from .cache import ProjectionCache
from .hierarchy import visible
from .items import why
from .matching import search

Records = list[dict[str, Any]]


def _best(query: str, records: Records, fields: list[str] | None = None) -> dict[str, Any] | None:
    hits = search(query, records, fields or ["name"], limit=1)
    return hits[0][0] if hits else None


def _pointer(kind: str, rec: dict[str, Any]) -> dict[str, Any]:
    return {"id": rec["id"], "kind": kind, "name": rec["name"], "snippet": "", "why": "snapshot"}


def _candidates(query: str, records: Records, fields: list[str]) -> dict[str, Any]:
    hits = search(query, records, fields, limit=5)
    return {"candidates": [{"id": r["id"], "name": r["name"], "why": why(s)} for r, s in hits]}


# --- Shared name-resolution tools ---------------------------------------------------------------

async def resolve_user(api: WorkspaceAPI, token: str, query: str) -> dict[str, Any]:
    return _candidates(query, await api.list_users(token), ["name", "email"])


async def resolve_project(cache: ProjectionCache, token: str, query: str) -> dict[str, Any]:
    data = await cache.get(token)
    return _candidates(query, visible("project", data["project"], set(data["hierarchy"])), ["name", "description"])


async def resolve_space(api: WorkspaceAPI, cache: ProjectionCache, token: str, query: str) -> dict[str, Any]:
    allowed = set((await cache.get(token))["hierarchy"])
    spaces = [s for s in await api.list_spaces(token) if allowed & set(s["project_ids"])]
    return _candidates(query, spaces, ["name"])


# --- Fallback snapshots (each fans out in parallel) ---------------------------------------------

async def describe_user_context(api: WorkspaceAPI, cache: ProjectionCache, token: str, query: str = "") -> dict[str, Any]:
    """4 parallel calls: profile, tasks, meetings, emails. `query` is unused (the user is the subject)."""
    profile, tasks, meetings, emails = await asyncio.gather(
        api.get_profile(token), api.list_tasks(token), api.list_meetings(token), api.list_emails(token)
    )
    allowed = set((await cache.get(token))["hierarchy"])
    open_tasks = [t for t in visible("task", tasks, allowed) if t.get("status") == "open"]
    meetings = visible("meeting", meetings, allowed)
    return {
        "summary": {"name": profile["name"], "role": profile["role"], "timezone": profile["timezone"],
                    "open_tasks": len(open_tasks), "meetings": len(meetings), "emails": len(emails)},
        "items": [_pointer("task", t) for t in open_tasks[:5]] + [_pointer("meeting", m) for m in meetings[:3]],
    }


async def describe_project(api: WorkspaceAPI, cache: ProjectionCache, token: str, query: str) -> dict[str, Any]:
    """5 parallel calls: members, tasks, documents, meetings, violations of one project."""
    data = await cache.get(token)
    allowed = set(data["hierarchy"])
    project = _best(query, visible("project", data["project"], allowed))
    if project is None:
        return {"summary": None, "items": []}
    pid = project["id"]
    members, tasks, docs, meetings, violations = await asyncio.gather(
        api.project_members(token, pid),
        *(api.project_items(token, pid, k) for k in ("task", "document", "meeting", "violation")),
    )
    parts = {"task": tasks, "document": docs, "meeting": meetings, "violation": violations}
    parts = {k: visible(k, v, allowed) for k, v in parts.items()}
    items = [_pointer("project", project)] + [_pointer(k, r) for k, rs in parts.items() for r in rs[:3]]
    summary = {"project": project["name"], "members": [m["name"] for m in members],
               **{f"{k}s": len(rs) for k, rs in parts.items()}}
    return {"summary": summary, "items": items}


async def describe_space(api: WorkspaceAPI, cache: ProjectionCache, token: str, query: str) -> dict[str, Any]:
    """3 parallel calls: the space's projects, its members, and the user's KPIs."""
    allowed = set((await cache.get(token))["hierarchy"])
    spaces = [s for s in await api.list_spaces(token) if allowed & set(s["project_ids"])]
    space = _best(query, spaces)
    if space is None:
        return {"summary": None, "items": []}
    projects, members, kpis = await asyncio.gather(
        api.space_projects(token, space["id"]), api.space_members(token, space["id"]), api.list_kpis(token)
    )
    projects = visible("project", projects, allowed)
    in_space = {p["id"] for p in projects}
    kpis = [k for k in visible("kpi", kpis, allowed) if k["project_id"] in in_space]
    summary = {"space": space["name"], "projects": [p["name"] for p in projects], "members": [m["name"] for m in members]}
    return {"summary": summary, "items": [_pointer("project", p) for p in projects] + [_pointer("kpi", k) for k in kpis]}

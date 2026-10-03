"""The scout's tool set, in three layers. The token is closed over: the LLM can't see or change it.

1. Typed searches: one per entity type (plus hybrid document-text search).
2. Fallback snapshots: `describe_*`, for vague requests no typed search fits.
3. Shared name resolvers: `resolve_*`, spoken name -> id.

Production had 14 typed searches, 3 snapshots and 3 resolvers. This repo has 11, 3 and 3.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.tools import StructuredTool

from . import fallback
from .backend import WorkspaceAPI
from .cache import ProjectionCache
from .chat_history import ChatHistory
from .docsearch import DenseIndex, FakeDenseIndex, FakeKeywordIndex, KeywordIndex, default_chunks
from .hierarchy import visible
from .items import make_item
from .matching import search
from .rrf import hybrid_content_search
from .security import Identity, check_impersonation

# kind -> fields to grep. The first field is the display name.
FIELDS = {
    "task": ["name", "description", "status", "path"],
    "project": ["name", "description"],
    "document": ["name"],  # by name; text is searched by `search_document_text`
    "email": ["name", "sender", "body"],
    "meeting": ["name", "notes", "attendees"],
    "group_chat": ["name", "last_message", "members"],
    "kpi": ["name", "description", "value"],
    "violation": ["name", "description", "severity"],
}
POINTER_KIND = {"group_chat": "chat"}  # backend kind -> pointer bucket
WORKFLOW_FIELDS = ["name", "description", "source_prompt", "source_document"]
MAX_WORKFLOW_PAGES = 3

PinResolver = Callable[[str, str], Awaitable[dict[str, Any] | None]]


def make_pin_resolver(api: WorkspaceAPI, token: str) -> PinResolver:
    """Look one pinned record up by id (a single call, no scan), hierarchy-checked."""

    async def resolve(kind: str, record_id: str) -> dict[str, Any] | None:
        rec, allowed = await asyncio.gather(api.get_record(token, kind, record_id), api.allowed_project_ids(token))
        return rec if rec is not None and visible(kind, [rec], set(allowed)) else None

    return resolve


def make_tools(
    api: WorkspaceAPI,
    cache: ProjectionCache,
    token: str,
    chats: ChatHistory | None = None,
    tenant: str = "",
    user: str = "",
    *,
    keyword_index: KeywordIndex | None = None,
    dense_index: DenseIndex | None = None,
    impersonating: bool = False,
) -> list[StructuredTool]:
    """Build every tool for one user. With `impersonating=True` each call first checks that the
    ambient identity (see `security.ambient_identity`) is this tenant and user."""
    chunks = default_chunks()
    keyword_index = keyword_index or FakeKeywordIndex(chunks)
    dense_index = dense_index or FakeDenseIndex(chunks)
    claimed = Identity(tenant, user)

    def tool(name: str, description: str, run: Callable[[str], Awaitable[Any]]) -> StructuredTool:
        async def guarded(query: str = "") -> str:
            if impersonating:
                check_impersonation(claimed)
            return json.dumps(await run(query))

        return StructuredTool.from_function(coroutine=guarded, name=name, description=description)

    async def view(kind: str) -> list[dict[str, Any]]:
        data = await cache.get(token)
        return visible(kind, data[kind], set(data["hierarchy"]))

    def grep_tool(kind: str) -> StructuredTool:
        fields = FIELDS[kind]

        async def run(query: str) -> list[dict[str, Any]]:
            hits = search(query, await view(kind), fields)
            return [make_item(kind, r, s, fields, POINTER_KIND.get(kind)) for r, s in hits]

        return tool(f"search_{kind}s", f"Find {kind.replace('_', ' ')}s of the current user by name or text.", run)

    async def search_workflows(query: str) -> dict[str, Any]:
        """Page through the workflow API. If paging stops early, say so: `complete` is False."""
        allowed = set((await cache.get(token))["hierarchy"])
        rows: list[dict[str, Any]] = []
        total = 0
        for _ in range(MAX_WORKFLOW_PAGES):
            page = await api.list_workflows(token, offset=len(rows))
            rows += page.items
            total = page.total
            if not page.items or len(rows) >= total:
                break
        fetched = len(rows)
        rows = visible("workflow", rows, allowed)
        by_name = {r["id"] for r, _ in search(query, rows, WORKFLOW_FIELDS[:2])}
        items = [
            {**make_item("workflow", r, s, WORKFLOW_FIELDS), "why": "name match" if r["id"] in by_name else "built from a matching prompt or document"}
            for r, s in search(query, rows, WORKFLOW_FIELDS)
        ]
        return {"kind": "workflow", "items": items, "complete": fetched >= total,
                "fetched": fetched, "total": total}

    async def search_document_text(query: str) -> list[dict[str, Any]]:
        # Keyword and dense searches (and the cache read) run concurrently.
        kw_hits, dense_hits, data = await asyncio.gather(
            keyword_index.search(query), dense_index.search(query), cache.get(token)
        )
        docs = {d["id"]: d for d in visible("document", data["document"], set(data["hierarchy"]))}
        # The indexes are shared across users, so drop every hit the user may not see.
        kw_hits = [h for h in kw_hits if h["doc_id"] in docs]
        dense_hits = [h for h in dense_hits if h["doc_id"] in docs]
        return [
            {"id": r["doc_id"], "kind": "document", "name": docs[r["doc_id"]]["name"],
             "snippet": " ... ".join(r["snippets"]), "why": f"hybrid {r['score']:.2f}"}
            for r in hybrid_content_search(kw_hits, dense_hits)
        ]

    async def search_chats(query: str) -> list[dict[str, Any]]:
        assert chats is not None
        return [
            {"id": r["id"], "kind": "chat", "name": r["name"], "snippet": r["body"][:200], "why": "text match"}
            for r in chats.search(tenant, user, query)
        ]

    tools = [grep_tool(k) for k in FIELDS]
    tools += [
        tool("search_workflows", "Find workflows by name, or by the prompt or document they were built from. "
             "Check `complete`: when false the list is partial.", search_workflows),
        tool("search_document_text", "Find documents by what is written inside them (hybrid keyword + semantic).",
             search_document_text),
        tool("describe_user_context", "Fallback: snapshot of the current user (profile, open tasks, meetings).",
             lambda q: fallback.describe_user_context(api, cache, token, q)),
        tool("describe_project", "Fallback: snapshot of one project, by name.",
             lambda q: fallback.describe_project(api, cache, token, q)),
        tool("describe_space", "Fallback: snapshot of one space (a group of projects), by name.",
             lambda q: fallback.describe_space(api, cache, token, q)),
        tool("resolve_user", "Turn a person's name into a user id.", lambda q: fallback.resolve_user(api, token, q)),
        tool("resolve_project", "Turn a project name into a project id.", lambda q: fallback.resolve_project(cache, token, q)),
        tool("resolve_space", "Turn a space name into a space id.", lambda q: fallback.resolve_space(api, cache, token, q)),
    ]
    if chats is not None:
        tools.append(tool("search_chats", "Find the current user's past AI chats by title or text.", search_chats))
    return tools

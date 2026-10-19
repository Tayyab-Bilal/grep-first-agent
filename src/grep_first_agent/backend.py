"""The workspace API the agent searches through. Each call returns only the caller's records,
except where the seed deliberately over-returns (see `seed.py`) to test the hierarchy filter."""

from __future__ import annotations

import asyncio
import copy
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol

from .seed import HIERARCHY, PROJECT_MEMBERS, SEED, TOKENS

# Kinds fetched into the per-user projection (workflows are paged, so they are fetched separately).
KINDS = ("task", "project", "document", "email", "meeting", "group_chat", "kpi", "violation")
WORKFLOW_PAGE_SIZE = 2

Records = list[dict[str, Any]]


@dataclass
class WorkflowPage:
    items: Records
    total: int


class WorkspaceAPI(Protocol):
    async def list_tasks(self, token: str) -> Records: ...
    async def list_projects(self, token: str) -> Records: ...
    async def list_documents(self, token: str) -> Records: ...
    async def list_emails(self, token: str) -> Records: ...
    async def list_meetings(self, token: str) -> Records: ...
    async def list_group_chats(self, token: str) -> Records: ...
    async def list_kpis(self, token: str) -> Records: ...
    async def list_violations(self, token: str) -> Records: ...
    async def list_workflows(self, token: str, offset: int = 0) -> WorkflowPage: ...
    async def allowed_project_ids(self, token: str) -> list[str]: ...
    async def list_users(self, token: str) -> Records: ...
    async def list_spaces(self, token: str) -> Records: ...
    async def get_profile(self, token: str) -> dict[str, Any]: ...
    async def project_members(self, token: str, project_id: str) -> Records: ...
    async def project_items(self, token: str, project_id: str, kind: str) -> Records: ...
    async def space_projects(self, token: str, space_id: str) -> Records: ...
    async def space_members(self, token: str, space_id: str) -> Records: ...
    async def get_record(self, token: str, kind: str, record_id: str) -> dict[str, Any] | None: ...


class FakeWorkspaceAPI:
    """Seeded in-memory API. `delay` makes every call slow, `fail` makes labelled calls raise."""

    def __init__(self, delay: float = 0.0, fail: set[str] | None = None, page_size: int = WORKFLOW_PAGE_SIZE) -> None:
        self.delay = delay
        self.fail = fail or set()
        self.page_size = page_size
        self.calls: Counter[str] = Counter()

    async def _enter(self, label: str, token: str) -> str:
        """Count the call, wait, maybe fail, and return the user the token belongs to."""
        self.calls[label] += 1
        await asyncio.sleep(self.delay)
        if label in self.fail:
            raise RuntimeError(f"{label} service unavailable")
        return TOKENS[token]  # unknown token -> KeyError, like a 401

    async def _list(self, kind: str, token: str) -> Records:
        user = await self._enter(kind, token)
        return copy.deepcopy(SEED[user][kind])

    async def list_tasks(self, token: str) -> Records:
        return await self._list("task", token)

    async def list_projects(self, token: str) -> Records:
        return await self._list("project", token)

    async def list_documents(self, token: str) -> Records:
        return await self._list("document", token)

    async def list_emails(self, token: str) -> Records:
        return await self._list("email", token)

    async def list_meetings(self, token: str) -> Records:
        return await self._list("meeting", token)

    async def list_group_chats(self, token: str) -> Records:
        return await self._list("group_chat", token)

    async def list_kpis(self, token: str) -> Records:
        return await self._list("kpi", token)

    async def list_violations(self, token: str) -> Records:
        return await self._list("violation", token)

    async def list_workflows(self, token: str, offset: int = 0) -> WorkflowPage:
        user = await self._enter("workflow", token)
        rows = SEED[user]["workflow"]
        return WorkflowPage(copy.deepcopy(rows[offset : offset + self.page_size]), total=len(rows))

    async def allowed_project_ids(self, token: str) -> list[str]:
        user = await self._enter("hierarchy", token)
        return list(HIERARCHY[user])

    async def list_users(self, token: str) -> Records:
        return await self._list("users", token)

    async def list_spaces(self, token: str) -> Records:
        return await self._list("space", token)

    async def get_profile(self, token: str) -> dict[str, Any]:
        user = await self._enter("profile", token)
        return dict(SEED[user]["profile"])

    async def project_members(self, token: str, project_id: str) -> Records:
        user = await self._enter("project_members", token)
        ids = PROJECT_MEMBERS.get(project_id, [])
        return [copy.deepcopy(u) for u in SEED[user]["users"] if u["id"] in ids]

    async def project_items(self, token: str, project_id: str, kind: str) -> Records:
        user = await self._enter(f"project_{kind}", token)
        return [copy.deepcopy(r) for r in SEED[user][kind] if r.get("project_id") == project_id]

    async def space_projects(self, token: str, space_id: str) -> Records:
        user = await self._enter("space_projects", token)
        space = next(s for s in SEED[user]["space"] if s["id"] == space_id)
        return [copy.deepcopy(p) for p in SEED[user]["project"] if p["id"] in space["project_ids"]]

    async def space_members(self, token: str, space_id: str) -> Records:
        user = await self._enter("space_members", token)
        space = next(s for s in SEED[user]["space"] if s["id"] == space_id)
        ids = {m for p in space["project_ids"] for m in PROJECT_MEMBERS.get(p, [])}
        return [copy.deepcopy(u) for u in SEED[user]["users"] if u["id"] in ids]

    async def get_record(self, token: str, kind: str, record_id: str) -> dict[str, Any] | None:
        user = await self._enter(f"get_{kind}", token)
        key = "group_chat" if kind == "chat" else kind
        return next((copy.deepcopy(r) for r in SEED[user].get(key, []) if r["id"] == record_id), None)

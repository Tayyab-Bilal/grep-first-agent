"""Search past AI chats. SQL narrows by literal terms, Python ranks with the shared matcher."""

from __future__ import annotations

import sqlite3
from typing import Any

from .matching import search

FALLBACK_CAP = 500
_ESCAPE = str.maketrans({"\\": "\\\\", "%": "\\%", "_": "\\_"})


class ChatHistory:
    # In Postgres this is ILIKE + pg_trgm similarity(); the per-user filter shrinks the
    # trigram scan, so no new index is needed.
    def __init__(self, conn: sqlite3.Connection | None = None) -> None:
        self.conn = conn or sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS chats "
            "(id TEXT PRIMARY KEY, tenant TEXT, user TEXT, title TEXT, body TEXT)"
        )

    def add(self, id: str, tenant: str, user: str, title: str, body: str) -> None:
        self.conn.execute("INSERT INTO chats VALUES (?,?,?,?,?)", (id, tenant, user, title, body))

    def search(self, tenant: str, user: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
        terms = query.split()
        if not terms:
            return []
        # User text is escaped so '%' and '_' are literals, not wildcards.
        likes = " OR ".join(
            "title LIKE ? ESCAPE '\\' OR body LIKE ? ESCAPE '\\'" for _ in terms
        )
        params: list[str] = [tenant, user]
        for t in terms:
            params += [f"%{t.translate(_ESCAPE)}%"] * 2
        rows = self.conn.execute(
            f"SELECT id, title, body FROM chats WHERE tenant = ? AND user = ? AND ({likes})", params
        ).fetchall()
        if not rows:
            # Pure typo: no literal term matched, so let fuzzy ranking look at the user's recent
            # chats. Still hard-scoped; capped because fuzzy scoring is Python-side.
            rows = self.conn.execute(
                "SELECT id, title, body FROM chats WHERE tenant = ? AND user = ? "
                "ORDER BY rowid DESC LIMIT ?", (tenant, user, FALLBACK_CAP)
            ).fetchall()
        records = [{"id": i, "name": t, "body": b} for i, t, b in rows]
        return [r for r, _ in search(query, records, ["name", "body"], limit=limit)]

"""SQLite cache for analysis results and shingle->search-result lookups.

Keyed by content hash, never by user. Lets us reuse expensive web-search and
embedding calls across requests and across users — anonymised by design.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init()

    def _init(self) -> None:
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    hash TEXT PRIMARY KEY,
                    created_at INTEGER NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS shingle_search (
                    shingle_hash TEXT PRIMARY KEY,
                    created_at INTEGER NOT NULL,
                    results TEXT NOT NULL
                );
                """
            )

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get_analysis(self, h: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT payload FROM analyses WHERE hash = ?", (h,)).fetchone()
        return json.loads(row[0]) if row else None

    def put_analysis(self, h: str, payload: dict[str, Any]) -> None:
        import time

        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO analyses (hash, created_at, payload) VALUES (?, ?, ?)",
                (h, int(time.time()), json.dumps(payload, ensure_ascii=False)),
            )

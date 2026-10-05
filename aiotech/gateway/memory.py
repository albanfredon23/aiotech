"""
Mémoire persistante par agent (SQLite).

Correction v3 : l'API instanciait la mémoire en ":memory:", donc rien ne survivait à un
redémarrage malgré le nom "mémoire persistante". Le chemin vient maintenant de la
configuration (AIOTECH_DB_PATH). L'élagage se fait par session et non plus sur tout l'agent.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional


class AgentMemory:
    def __init__(self, db_path: str = ":memory:", max_messages_per_session: int = 200):
        self.max_messages = max_messages_per_session
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL, session_id TEXT NOT NULL,
                role TEXT NOT NULL, content TEXT NOT NULL, created_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS facts (
                agent_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
                updated_at REAL NOT NULL, PRIMARY KEY (agent_id, key));
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL, session_id TEXT NOT NULL,
                summary TEXT NOT NULL, created_at REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_msg ON messages(agent_id, session_id, id);
            """
        )
        self._conn.commit()

    def add_message(self, agent_id: str, session_id: str, role: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (agent_id, session_id, role, content, created_at) VALUES (?,?,?,?,?)",
                (agent_id, session_id, role, content, time.time()),
            )
            self._conn.execute(
                "DELETE FROM messages WHERE agent_id=? AND session_id=? AND id NOT IN ("
                "SELECT id FROM messages WHERE agent_id=? AND session_id=? ORDER BY id DESC LIMIT ?)",
                (agent_id, session_id, agent_id, session_id, self.max_messages),
            )
            self._conn.commit()

    def get_history(self, agent_id: str, session_id: str, limit: int = 20) -> List[Dict[str, str]]:
        rows = self._conn.execute(
            "SELECT role, content FROM messages WHERE agent_id=? AND session_id=? ORDER BY id DESC LIMIT ?",
            (agent_id, session_id, limit),
        ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def set_fact(self, agent_id: str, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO facts (agent_id, key, value, updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(agent_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (agent_id, key, json.dumps(value, ensure_ascii=False), time.time()),
            )
            self._conn.commit()

    def get_fact(self, agent_id: str, key: str) -> Optional[Any]:
        row = self._conn.execute("SELECT value FROM facts WHERE agent_id=? AND key=?", (agent_id, key)).fetchone()
        return json.loads(row[0]) if row else None

    def get_all_facts(self, agent_id: str) -> Dict[str, Any]:
        rows = self._conn.execute("SELECT key, value FROM facts WHERE agent_id=?", (agent_id,)).fetchall()
        return {k: json.loads(v) for k, v in rows}

    def store_summary(self, agent_id: str, session_id: str, summary: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO summaries (agent_id, session_id, summary, created_at) VALUES (?,?,?,?)",
                (agent_id, session_id, summary, time.time()),
            )
            self._conn.commit()

    def get_last_summary(self, agent_id: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT summary FROM summaries WHERE agent_id=? ORDER BY id DESC LIMIT 1", (agent_id,)
        ).fetchone()
        return row[0] if row else None

    def stats(self, agent_id: str) -> Dict[str, int]:
        q = lambda sql: self._conn.execute(sql, (agent_id,)).fetchone()[0]  # noqa: E731
        return {
            "messages": q("SELECT COUNT(*) FROM messages WHERE agent_id=?"),
            "facts": q("SELECT COUNT(*) FROM facts WHERE agent_id=?"),
            "summaries": q("SELECT COUNT(*) FROM summaries WHERE agent_id=?"),
        }

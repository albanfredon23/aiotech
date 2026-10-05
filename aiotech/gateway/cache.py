"""
Cache sémantique (SQLite + embeddings lexicaux).

Corrections par rapport à v3 :
- v3 utilisait un vecteur aléatoire dérivé d'un SHA-256 : deux questions différant d'une
  virgule avaient une similarité ~0. C'était un cache exact déguisé. Ici la similarité est
  un recouvrement lexical réel (Jaccard pondéré sur traits hachés, sans cosinus).
- garde "identifiants" : deux requêtes dont les nombres/références diffèrent (X-12 / X-13,
  2023 / 2024) ne partagent jamais de réponse, quel que soit le score ;
- la clé inclut une empreinte du contexte (prompt système, historique, documents retenus) :
  la même question dans une autre conversation ne renvoie pas une réponse hors sujet ;
- les erreurs ne sont jamais mises en cache ; durée de vie (TTL) ; éviction des plus anciens.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from typing import Any, Dict, Optional

import numpy as np

from aiotech.embeddings import HashingEmbedder, weighted_jaccard
from aiotech.text import identifiers


def context_fingerprint(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


class SemanticCache:
    def __init__(self, db_path: str = ":memory:", threshold: float = 0.85,
                 max_entries: int = 5000, ttl_seconds: float = 24 * 3600,
                 embedder: Optional[HashingEmbedder] = None, identifier_guard: bool = True):
        self.threshold = threshold
        self.identifier_guard = identifier_guard
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self.embedder = embedder or HashingEmbedder(dim=1024)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                query TEXT NOT NULL,
                identifiers TEXT NOT NULL,
                emb BLOB NOT NULL,
                response TEXT NOT NULL,
                tokens_in INTEGER NOT NULL DEFAULT 0,
                tokens_out INTEGER NOT NULL DEFAULT 0,
                cost_usd REAL,
                created_at REAL NOT NULL
            )"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_key ON cache(model, fingerprint)")
        self._conn.commit()
        self._stats = {"hits": 0, "misses": 0, "tokens_saved": 0, "cost_saved_usd": 0.0,
                       "rejected_by_identifier_guard": 0}

    def lookup(self, model: str, query: str, fingerprint: str = "") -> Optional[Dict[str, Any]]:
        q_emb = self.embedder.embed_one(query)
        q_ids = sorted(identifiers(query))
        cutoff = time.time() - self.ttl
        rows = self._conn.execute(
            "SELECT emb, response, tokens_in, tokens_out, cost_usd, identifiers, query "
            "FROM cache WHERE model=? AND fingerprint=? AND created_at>=?",
            (model, fingerprint, cutoff),
        ).fetchall()
        best_sim, best_row, guard_rejected = -1.0, None, False
        for row in rows:
            sim = weighted_jaccard(q_emb, np.frombuffer(row[0], dtype=np.float32))
            if sim < self.threshold:
                continue
            if self.identifier_guard and json.loads(row[5]) != q_ids:
                guard_rejected = True
                continue
            if sim > best_sim:
                best_sim, best_row = sim, row
        if best_row is None:
            self._stats["misses"] += 1
            if guard_rejected:
                self._stats["rejected_by_identifier_guard"] += 1
            return None
        self._stats["hits"] += 1
        self._stats["tokens_saved"] += int(best_row[2]) + int(best_row[3])
        if best_row[4] is not None:
            self._stats["cost_saved_usd"] += float(best_row[4])
        return {
            "content": best_row[1],
            "similarity": round(best_sim, 4),
            "cached_query": best_row[6],
            "tokens_in": int(best_row[2]),
            "tokens_out": int(best_row[3]),
            "cost_usd": best_row[4],
        }

    def store(self, model: str, query: str, response: str, tokens_in: int = 0,
              tokens_out: int = 0, cost_usd: Optional[float] = None, fingerprint: str = "") -> None:
        if not response:
            return
        emb = self.embedder.embed_one(query)
        self._conn.execute(
            "INSERT INTO cache (model, fingerprint, query, identifiers, emb, response, tokens_in, "
            "tokens_out, cost_usd, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (model, fingerprint, query, json.dumps(sorted(identifiers(query))), emb.tobytes(),
             response, int(tokens_in), int(tokens_out), cost_usd, time.time()),
        )
        count = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        if count > self.max_entries:
            self._conn.execute(
                "DELETE FROM cache WHERE id IN (SELECT id FROM cache ORDER BY created_at ASC LIMIT ?)",
                (count - self.max_entries,),
            )
        self._conn.commit()

    def purge_expired(self) -> int:
        cur = self._conn.execute("DELETE FROM cache WHERE created_at < ?", (time.time() - self.ttl,))
        self._conn.commit()
        return cur.rowcount

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._stats["hits"] + self._stats["misses"]
        return {**self._stats, "cost_saved_usd": round(self._stats["cost_saved_usd"], 6),
                "hit_rate": round(self._stats["hits"] / total, 4) if total else 0.0,
                "total_queries": total}

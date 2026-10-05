"""
Multi-locataires : clés API hachées, quotas mensuels, journal d'audit.

Corrections par rapport à v3 :
- l'identifiant était md5(nom) avec INSERT OR IGNORE : recréer un locataire du même nom
  renvoyait une clé API qui n'était jamais enregistrée (clé inutilisable). Les noms sont
  désormais uniques et un doublon lève une erreur explicite ;
- la vérification de quota et l'enregistrement de l'usage sont atomiques (verrou), pour
  éviter de dépasser le quota sous charge concurrente.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

PLANS = {
    "starter": (10_000, 5_000_000),
    "pro": (250_000, 100_000_000),
    "enterprise": (10_000_000, 5_000_000_000),
}


class TenantExists(ValueError):
    pass


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


class TenantManager:
    def __init__(self, db_path: str = ":memory:"):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tenants (
                id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, api_key_hash TEXT NOT NULL UNIQUE,
                plan TEXT NOT NULL, quota_req_month INTEGER NOT NULL, quota_tokens_month INTEGER NOT NULL,
                created_at REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1);
            CREATE TABLE IF NOT EXISTS usage (
                tenant_id TEXT NOT NULL, period TEXT NOT NULL, requests INTEGER NOT NULL DEFAULT 0,
                tokens_in INTEGER NOT NULL DEFAULT 0, tokens_out INTEGER NOT NULL DEFAULT 0,
                cost_usd REAL NOT NULL DEFAULT 0, saved_usd REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (tenant_id, period));
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL,
                event TEXT NOT NULL, detail TEXT, ts REAL NOT NULL);
            """
        )
        self._conn.commit()

    @staticmethod
    def _period() -> str:
        t = time.gmtime()
        return f"{t.tm_year}-{t.tm_mon:02d}"

    def create_tenant(self, name: str, plan: str = "starter") -> Dict[str, Any]:
        if plan not in PLANS:
            raise ValueError(f"plan inconnu : {plan} (choix : {', '.join(PLANS)})")
        api_key = "aio-" + secrets.token_urlsafe(32)
        tenant_id = "t_" + secrets.token_hex(6)
        q_req, q_tok = PLANS[plan]
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO tenants (id, name, api_key_hash, plan, quota_req_month, quota_tokens_month, created_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (tenant_id, name, _hash_key(api_key), plan, q_req, q_tok, time.time()),
                )
            except sqlite3.IntegrityError as exc:
                raise TenantExists(f"un locataire nommé « {name} » existe déjà") from exc
            self._audit(tenant_id, "created", f"plan={plan}")
            self._conn.commit()
        return {"tenant_id": tenant_id, "name": name, "plan": plan, "api_key": api_key}

    def authenticate(self, api_key: str) -> Optional[str]:
        if not api_key:
            return None
        h = _hash_key(api_key)
        row = self._conn.execute(
            "SELECT id, api_key_hash FROM tenants WHERE api_key_hash=? AND active=1", (h,)
        ).fetchone()
        if row and hmac.compare_digest(row[1], h):
            return row[0]
        return None

    def check_and_reserve(self, tenant_id: str) -> Dict[str, Any]:
        """Vérifie le quota et réserve une requête de façon atomique."""
        with self._lock:
            t = self._conn.execute(
                "SELECT quota_req_month, quota_tokens_month, plan FROM tenants WHERE id=? AND active=1",
                (tenant_id,),
            ).fetchone()
            if not t:
                return {"allowed": False, "reason": "unknown_tenant"}
            q_req, q_tok, plan = t
            period = self._period()
            u = self._conn.execute(
                "SELECT requests, tokens_in + tokens_out FROM usage WHERE tenant_id=? AND period=?",
                (tenant_id, period),
            ).fetchone()
            used_req, used_tok = (u if u else (0, 0))
            if used_req >= q_req:
                return {"allowed": False, "reason": "quota_requests_exceeded", "used": used_req, "limit": q_req}
            if used_tok >= q_tok:
                return {"allowed": False, "reason": "quota_tokens_exceeded", "used": used_tok, "limit": q_tok}
            self._conn.execute(
                "INSERT INTO usage (tenant_id, period, requests) VALUES (?,?,1) "
                "ON CONFLICT(tenant_id, period) DO UPDATE SET requests = requests + 1",
                (tenant_id, period),
            )
            self._conn.commit()
            return {"allowed": True, "plan": plan, "used_requests": used_req + 1, "limit_requests": q_req,
                    "used_tokens": used_tok, "limit_tokens": q_tok}

    def record_usage(self, tenant_id: str, tokens_in: int = 0, tokens_out: int = 0,
                     cost_usd: float = 0.0, saved_usd: float = 0.0, requests: int = 0) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage (tenant_id, period, requests, tokens_in, tokens_out, cost_usd, saved_usd) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(tenant_id, period) DO UPDATE SET "
                "requests = requests + excluded.requests, tokens_in = tokens_in + excluded.tokens_in, "
                "tokens_out = tokens_out + excluded.tokens_out, cost_usd = cost_usd + excluded.cost_usd, "
                "saved_usd = saved_usd + excluded.saved_usd",
                (tenant_id, self._period(), requests, tokens_in, tokens_out, cost_usd or 0.0, saved_usd or 0.0),
            )
            self._conn.commit()

    def get_usage(self, tenant_id: str) -> Dict[str, Any]:
        period = self._period()
        r = self._conn.execute(
            "SELECT requests, tokens_in, tokens_out, cost_usd, saved_usd FROM usage WHERE tenant_id=? AND period=?",
            (tenant_id, period),
        ).fetchone() or (0, 0, 0, 0.0, 0.0)
        return {"tenant_id": tenant_id, "period": period, "requests": r[0], "tokens_in": r[1],
                "tokens_out": r[2], "cost_usd": round(r[3], 6), "saved_usd": round(r[4], 6)}

    def list_tenants(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute("SELECT id, name, plan, active FROM tenants ORDER BY created_at DESC").fetchall()
        return [{"id": r[0], "name": r[1], "plan": r[2], "active": bool(r[3])} for r in rows]

    def deactivate_tenant(self, tenant_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("UPDATE tenants SET active=0 WHERE id=?", (tenant_id,))
            if cur.rowcount:
                self._audit(tenant_id, "deactivated", "")
            self._conn.commit()
            return cur.rowcount > 0

    def _audit(self, tenant_id: str, event: str, detail: str = "") -> None:
        self._conn.execute("INSERT INTO audit_log (tenant_id, event, detail, ts) VALUES (?,?,?,?)",
                           (tenant_id, event, detail, time.time()))

    def get_audit_log(self, tenant_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT event, detail, ts FROM audit_log WHERE tenant_id=? ORDER BY id DESC LIMIT ?",
            (tenant_id, limit),
        ).fetchall()
        return [{"event": e, "detail": d, "ts": t} for e, d, t in rows]

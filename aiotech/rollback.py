"""
Retour arrière automatique des mutations déployées (repris de v3).

Une mutation promue est observée sur une fenêtre de requêtes ; elle est annulée si son taux
d'erreur ou sa latence médiane dépassent les seuils, et déclarée stable après 3 fenêtres.
"""
from __future__ import annotations

import statistics
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class MutationStatus(Enum):
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    STABLE = "stable"


class MutationRecord:
    def __init__(self, mutation_id: str, name: str):
        self.mutation_id = mutation_id
        self.name = name
        self.status = MutationStatus.CANDIDATE
        self.promoted_at: Optional[float] = None
        self.latencies: List[float] = []
        self.errors = 0
        self.requests = 0
        self.rollback_reason: Optional[str] = None

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0

    @property
    def median_latency(self) -> float:
        return float(statistics.median(self.latencies)) if self.latencies else 0.0


class RollbackManager:
    def __init__(self, error_rate_threshold: float = 0.10, latency_degradation: float = 1.30,
                 min_requests: int = 10, on_rollback: Optional[Callable[[str, str], None]] = None):
        self.error_thr = error_rate_threshold
        self.latency_deg = latency_degradation
        self.min_req = min_requests
        self.on_rollback = on_rollback
        self._mutations: Dict[str, MutationRecord] = {}
        self._baseline_latency: Optional[float] = None
        self._history: List[Dict[str, Any]] = []

    def set_baseline(self, median_latency_ms: float) -> None:
        self._baseline_latency = median_latency_ms

    def promote(self, mutation_id: str, name: str) -> MutationRecord:
        rec = MutationRecord(mutation_id, name)
        rec.status = MutationStatus.PROMOTED
        rec.promoted_at = time.time()
        self._mutations[mutation_id] = rec
        self._history.append({"event": "promoted", "mutation_id": mutation_id, "name": name, "ts": time.time()})
        return rec

    def record(self, mutation_id: str, latency_ms: float, success: bool) -> Optional[str]:
        rec = self._mutations.get(mutation_id)
        if not rec or rec.status not in (MutationStatus.PROMOTED,):
            return None
        rec.requests += 1
        rec.latencies.append(latency_ms)
        if not success:
            rec.errors += 1
        if rec.requests < self.min_req:
            return None
        if rec.error_rate > self.error_thr:
            return self._rollback(rec, f"error_rate={rec.error_rate:.2%} > {self.error_thr:.2%}")
        if self._baseline_latency and rec.median_latency > self._baseline_latency * self.latency_deg:
            return self._rollback(rec, f"latency={rec.median_latency:.1f}ms > "
                                       f"{self._baseline_latency * self.latency_deg:.1f}ms")
        if rec.requests >= self.min_req * 3:
            rec.status = MutationStatus.STABLE
            self._history.append({"event": "stable", "mutation_id": mutation_id, "ts": time.time()})
        return None

    def _rollback(self, rec: MutationRecord, reason: str) -> str:
        rec.status = MutationStatus.ROLLED_BACK
        rec.rollback_reason = reason
        self._history.append({"event": "rollback", "mutation_id": rec.mutation_id, "name": rec.name,
                              "reason": reason, "ts": time.time()})
        if self.on_rollback:
            self.on_rollback(rec.mutation_id, reason)
        return reason

    def status(self, mutation_id: str) -> Optional[Dict[str, Any]]:
        rec = self._mutations.get(mutation_id)
        if not rec:
            return None
        return {"mutation_id": rec.mutation_id, "name": rec.name, "status": rec.status.value,
                "requests": rec.requests, "error_rate": round(rec.error_rate, 4),
                "median_latency_ms": round(rec.median_latency, 2), "rollback_reason": rec.rollback_reason}

    def all_statuses(self) -> List[Dict[str, Any]]:
        return [self.status(mid) for mid in self._mutations]

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

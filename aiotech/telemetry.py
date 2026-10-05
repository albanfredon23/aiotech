"""
Métriques au format texte Prometheus, sans dépendance.

Corrections v3 : une seule ligne `# TYPE` par famille de métriques (v3 en émettait une par
combinaison d'étiquettes, ce que Prometheus rejette), et les distributions sont exposées en
type `summary` (quantiles), ce qu'elles sont réellement. Les échantillons des distributions
sont bornés (fenêtre glissante) pour ne pas croître sans limite.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

LabelKey = Tuple[Tuple[str, str], ...]


def _labels(labels: Optional[Dict[str, str]]) -> LabelKey:
    return tuple(sorted((labels or {}).items()))


def _fmt_labels(lk: LabelKey, extra: Optional[Tuple[str, str]] = None) -> str:
    items = list(lk) + ([extra] if extra else [])
    if not items:
        return ""
    esc = lambda v: str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")  # noqa: E731
    return "{" + ",".join(f'{k}="{esc(v)}"' for k, v in items) + "}"


class MetricsRegistry:
    def __init__(self, window: int = 5000):
        self._lock = threading.Lock()
        self._counters: Dict[str, Dict[LabelKey, float]] = {}
        self._gauges: Dict[str, Dict[LabelKey, float]] = {}
        self._summaries: Dict[str, Dict[LabelKey, Tuple[Deque[float], list]]] = {}
        self._window = window
        self._start = time.time()

    def counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        with self._lock:
            fam = self._counters.setdefault(name, {})
            lk = _labels(labels)
            fam[lk] = fam.get(lk, 0.0) + value

    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        with self._lock:
            self._gauges.setdefault(name, {})[_labels(labels)] = value

    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        with self._lock:
            fam = self._summaries.setdefault(name, {})
            lk = _labels(labels)
            if lk not in fam:
                fam[lk] = (deque(maxlen=self._window), [0, 0.0])  # échantillons, [count, sum]
            samples, agg = fam[lk]
            samples.append(value)
            agg[0] += 1
            agg[1] += value

    @staticmethod
    def _quantile(sorted_vals, q: float) -> float:
        if not sorted_vals:
            return 0.0
        idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
        return sorted_vals[idx]

    def export_prometheus(self) -> str:
        lines = []
        with self._lock:
            for name, fam in self._counters.items():
                lines.append(f"# TYPE {name} counter")
                for lk, v in fam.items():
                    lines.append(f"{name}{_fmt_labels(lk)} {v}")
            for name, fam in self._gauges.items():
                lines.append(f"# TYPE {name} gauge")
                for lk, v in fam.items():
                    lines.append(f"{name}{_fmt_labels(lk)} {v}")
            for name, fam in self._summaries.items():
                lines.append(f"# TYPE {name} summary")
                for lk, (samples, agg) in fam.items():
                    s = sorted(samples)
                    for q in (0.5, 0.9, 0.99):
                        lines.append(f"{name}{_fmt_labels(lk, ('quantile', str(q)))} {self._quantile(s, q):.6g}")
                    lines.append(f"{name}_sum{_fmt_labels(lk)} {agg[1]:.6g}")
                    lines.append(f"{name}_count{_fmt_labels(lk)} {agg[0]}")
            lines.append("# TYPE aiotech_uptime_seconds gauge")
            lines.append(f"aiotech_uptime_seconds {time.time() - self._start:.0f}")
        return "\n".join(lines) + "\n"

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            flat = lambda fams: {n + _fmt_labels(lk): v for n, fam in fams.items() for lk, v in fam.items()}  # noqa: E731
            summaries = {}
            for name, fam in self._summaries.items():
                for lk, (samples, agg) in fam.items():
                    s = sorted(samples)
                    summaries[name + _fmt_labels(lk)] = {
                        "count": agg[0], "sum": agg[1],
                        "p50": self._quantile(s, 0.5), "p99": self._quantile(s, 0.99)}
            return {"counters": flat(self._counters), "gauges": flat(self._gauges),
                    "summaries": summaries, "uptime_s": round(time.time() - self._start, 1)}


METRICS = MetricsRegistry()

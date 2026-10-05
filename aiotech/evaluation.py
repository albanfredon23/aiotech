"""
Évaluation des réponses : correspondance exacte normalisée, tolérance numérique, inclusion.
(Repris de v3, normalisation des accents ajoutée, extraction numérique gérant 1 234,5 et 1,234.5.)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from aiotech.text import strip_accents


def _normalize(text: str) -> str:
    text = strip_accents(text.lower()).strip()
    text = re.sub(r"[^\w\s.\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_number(text: str) -> Optional[float]:
    """Dernier nombre du texte. Gère '1 234,5', '1,234.5', '1234.5', '-3e2'."""
    cands = re.findall(r"[-+]?\d[\d\s  .,]*(?:[eE][-+]?\d+)?", text)
    for raw in reversed(cands):
        s = re.sub(r"[\s  ]", "", raw).rstrip(".,")
        if "," in s and "." in s:
            s = s.replace(",", "") if s.rfind(".") > s.rfind(",") else s.replace(".", "").replace(",", ".")
        elif "," in s:
            parts = s.split(",")
            s = s.replace(",", "") if len(parts[-1]) == 3 and len(parts) > 1 and len(parts[0]) <= 3 else s.replace(",", ".")
        try:
            return float(s)
        except ValueError:
            continue
    return None


class MultiTaskEvaluator:
    def __init__(self, numeric_rtol: float = 0.01, numeric_atol: float = 0.5):
        self.rtol = numeric_rtol
        self.atol = numeric_atol

    def evaluate(self, prediction: str, reference: str, task_type: str = "exact_match") -> Dict[str, Any]:
        if task_type == "numeric":
            p, r = extract_number(prediction), extract_number(reference)
            if p is None or r is None:
                return self.evaluate(prediction, reference, "exact_match")
            err = abs(p - r)
            ok = err <= self.atol or err / max(abs(r), 1e-9) <= self.rtol
            return {"correct": ok, "score": float(ok), "detail": f"pred={p} ref={r} abs_err={err:.4g}"}
        if task_type == "contains":
            ok = _normalize(reference) in _normalize(prediction)
            return {"correct": ok, "score": float(ok), "detail": f"'{reference[:30]}' présent : {ok}"}
        p, r = _normalize(prediction), _normalize(reference)
        return {"correct": p == r, "score": float(p == r), "detail": f"pred='{p[:40]}' ref='{r[:40]}'"}

    def evaluate_batch(self, predictions: List[str], references: List[str],
                       task_type: str = "exact_match") -> Dict[str, Any]:
        if len(predictions) != len(references):
            raise ValueError("predictions et references doivent avoir la même longueur")
        res = [self.evaluate(p, r, task_type) for p, r in zip(predictions, references)]
        n_ok = sum(r["correct"] for r in res)
        return {"accuracy": round(n_ok / len(res), 4) if res else 0.0, "n_correct": n_ok,
                "n_total": len(res), "task_type": task_type, "per_sample": res}

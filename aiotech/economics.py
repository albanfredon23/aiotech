"""
Modèle économique : coût mensuel avec et sans AIOTECH, seuil de rentabilité, économies d'échelle.

Notations (par requête, sauf mention contraire) :
  N        requêtes par mois
  P        tokens de prompt (question + instructions)
  C        tokens de contexte documentaire candidat (RAG)
  O        tokens de sortie
  pin_f, pout_f   prix du modèle phare ($/Mtok entrée, sortie)
  pin_c, pout_c   prix du modèle économique
  h        taux de réponses servies par le cache          (levier 1)
  r        part des requêtes non cachées routées vers le modèle économique (levier 2)
  rho      réduction du contexte par l'ARG                (levier 3)
  F        coût fixe mensuel du middleware (hébergement, maintenance, licence)
  v        coût variable du middleware par requête (calcul, embeddings éventuels)

Sans AIOTECH (référence) :
  cost0(req) = [ (P + C)·pin_f + O·pout_f ] / 1e6
  C0 = N · cost0(req)

Avec AIOTECH :
  P' + C' = P + (1 - rho)·C
  cost_llm(req) = (1 - h) · { r·[(P + C')·pin_c + O·pout_c] + (1 - r)·[(P + C')·pin_f + O·pout_f] } / 1e6
  C1 = F + N·v + N·cost_llm(req)

Économie : S = C0 - C1. Gain unitaire brut g = cost0(req) - cost_llm(req) - v.
Seuil de rentabilité : N* = F / g (si g > 0).
Coût moyen par requête : C1/N = F/N + v + cost_llm(req) -> décroît avec N : c'est
l'économie d'échelle (le coût fixe s'amortit, le gain unitaire reste constant).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence


@dataclass
class Scenario:
    requests_per_month: float = 1_000_000
    prompt_tokens: float = 400
    context_tokens: float = 3_000
    output_tokens: float = 350
    flagship_in: float = 4.0      # Claude Opus 5.5, $/Mtok (2026-09-25)
    flagship_out: float = 20.0
    cheap_in: float = 1.0         # Claude Haiku 4.5
    cheap_out: float = 5.0
    cache_hit_rate: float = 0.15
    cheap_share: float = 0.40
    context_reduction: float = 0.50
    fixed_cost_month: float = 1_500.0
    variable_cost_per_request: float = 0.00002

    def validate(self) -> None:
        for name in ("cache_hit_rate", "cheap_share", "context_reduction"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} doit être dans [0, 1]")
        for name in ("requests_per_month", "prompt_tokens", "context_tokens", "output_tokens",
                     "flagship_in", "flagship_out", "cheap_in", "cheap_out",
                     "fixed_cost_month", "variable_cost_per_request"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} doit être positif")

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _unit_costs(s: Scenario) -> Dict[str, float]:
    p, c, o = s.prompt_tokens, s.context_tokens, s.output_tokens
    base = ((p + c) * s.flagship_in + o * s.flagship_out) / 1e6
    in_after = p + (1.0 - s.context_reduction) * c
    cheap = (in_after * s.cheap_in + o * s.cheap_out) / 1e6
    flag = (in_after * s.flagship_in + o * s.flagship_out) / 1e6
    llm = (1.0 - s.cache_hit_rate) * (s.cheap_share * cheap + (1.0 - s.cheap_share) * flag)
    return {"baseline": base, "llm_after": llm}


def lever_breakdown(s: Scenario) -> Dict[str, float]:
    """Économie unitaire attribuée à chaque levier, appliqués dans l'ordre contexte -> routage -> cache.

    L'ordre change la répartition, pas le total (les leviers se multiplient).
    """
    s.validate()
    p, c, o = s.prompt_tokens, s.context_tokens, s.output_tokens
    base = ((p + c) * s.flagship_in + o * s.flagship_out) / 1e6
    in_after = p + (1.0 - s.context_reduction) * c
    after_ctx = (in_after * s.flagship_in + o * s.flagship_out) / 1e6
    cheap = (in_after * s.cheap_in + o * s.cheap_out) / 1e6
    after_route = s.cheap_share * cheap + (1.0 - s.cheap_share) * after_ctx
    after_cache = (1.0 - s.cache_hit_rate) * after_route
    return {
        "context": base - after_ctx,
        "routing": after_ctx - after_route,
        "cache": after_route - after_cache,
        "middleware_cost": -s.variable_cost_per_request,
    }


def monthly_costs(s: Scenario) -> Dict[str, Optional[float]]:
    s.validate()
    u = _unit_costs(s)
    n = s.requests_per_month
    c0 = n * u["baseline"]
    c1 = s.fixed_cost_month + n * (s.variable_cost_per_request + u["llm_after"])
    gain = u["baseline"] - u["llm_after"] - s.variable_cost_per_request
    breakeven = s.fixed_cost_month / gain if gain > 0 else None
    return {
        "baseline_month": c0,
        "aiotech_month": c1,
        "savings_month": c0 - c1,
        "savings_pct": (c0 - c1) / c0 if c0 > 0 else 0.0,
        "baseline_per_request": u["baseline"],
        "aiotech_per_request": c1 / n if n > 0 else None,
        "gross_gain_per_request": gain,
        "breakeven_requests_month": breakeven,
        "savings_year": 12 * (c0 - c1),
    }


def cost_curve(s: Scenario, volumes: Sequence[float]) -> List[Dict[str, float]]:
    """Coût par requête et économie mensuelle pour une série de volumes mensuels."""
    out = []
    for n in volumes:
        sc = Scenario(**{**s.as_dict(), "requests_per_month": float(n)})
        m = monthly_costs(sc)
        out.append({
            "requests_per_month": float(n),
            "baseline_per_request": m["baseline_per_request"],
            "aiotech_per_request": m["aiotech_per_request"],
            "savings_month": m["savings_month"],
        })
    return out


def log_volumes(start: float = 1e3, stop: float = 1e8, points: int = 26) -> List[float]:
    if points < 2:
        return [start]
    ratio = (stop / start) ** (1.0 / (points - 1))
    return [start * ratio ** i for i in range(points)]

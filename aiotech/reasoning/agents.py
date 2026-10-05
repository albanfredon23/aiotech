"""
Modulation d'agents (successeur de l'AgentModulatorGate v3, sans PyTorch ni sphère).

v3 faisait passer la requête dans des réseaux aléatoires jamais entraînés puis comparait
les sorties par cosinus : le routage était du bruit. Ici :
- chaque agent est décrit par un texte (ses compétences) ;
- l'affinité requête/agent est le critère `reach` de l'ARG (euclidien, sans normalisation) ;
- le prior de chaque agent est son taux de succès estimé en ligne (retours utilisateurs) ;
- un agent est admissible si T_G(affinité, prior) >= tau ;
- les poids de routage sont softmax(affinité / T + log prior) sur les agents admissibles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from aiotech.embeddings import HashingEmbedder
from aiotech.reasoning.arg import ReachabilityGate, godel_tnorm
from aiotech.text import content_words


class OnlineSuccessEstimator:
    """Estimateur RLS scalaire y = beta avec facteur d'oubli lambda.

    Avec une entrée constante x = 1, la récursion RLS de v3 se réduit exactement à une
    moyenne pondérée exponentiellement des scores de succès : c'est ce que l'on calcule,
    directement et sans tenseurs (gain_t = 1 / w_t avec w_t = lambda * w_{t-1} + 1).
    Le prior initial compte pour `prior_strength` observations. `prior` est borné à [floor, 1].
    """

    def __init__(self, prior: float = 0.5, forgetting: float = 0.95, floor: float = 0.05,
                 prior_strength: float = 1.0):
        self.beta = prior
        self.lam = forgetting
        self.floor = floor
        self._weight = prior_strength
        self.updates = 0

    def update(self, success: float) -> None:
        success = min(max(float(success), 0.0), 1.0)
        self._weight = self.lam * self._weight + 1.0
        gain = 1.0 / self._weight
        self.beta += gain * (success - self.beta)
        self.updates += 1

    @property
    def prior(self) -> float:
        return float(min(max(self.beta, self.floor), 1.0))


@dataclass
class _Agent:
    agent_id: str
    description: str
    vector: np.ndarray
    estimator: OnlineSuccessEstimator = field(default_factory=OnlineSuccessEstimator)


class AgentModulator:
    def __init__(self, tau: float = 0.2, temperature: float = 0.15,
                 embedder: Optional[HashingEmbedder] = None):
        self.tau = tau
        self.temperature = temperature
        self.embedder = embedder or HashingEmbedder(bigram_weight=0.0)
        self._agents: Dict[str, _Agent] = {}

    def register(self, agent_id: str, description: str, prior: float = 0.5) -> None:
        vec = self.embedder.embed_one(description)
        self._agents[agent_id] = _Agent(agent_id, description, vec, OnlineSuccessEstimator(prior))

    @property
    def agent_ids(self) -> List[str]:
        return list(self._agents)

    def route(self, query: str) -> Dict[str, Any]:
        if not self._agents:
            raise RuntimeError("Aucun agent enregistré.")
        words = content_words(query)
        q = self.embedder.embed_one(" ".join(words) if words else query)
        rows = []
        for a in self._agents.values():
            affinity = float(np.clip(ReachabilityGate.reach(q, a.vector), 0.0, 1.0))
            prior = a.estimator.prior
            godel = float(godel_tnorm([affinity, prior]))
            rows.append({"id": a.agent_id, "affinity": affinity, "prior": prior,
                         "godel": godel, "admissible": godel >= self.tau})

        candidates = [r for r in rows if r["admissible"]]
        fallback = not candidates
        if fallback:  # repli : l'agent le moins dégradé
            candidates = [max(rows, key=lambda r: r["godel"])]

        logits = np.array([r["affinity"] / self.temperature + math.log(r["prior"]) for r in candidates])
        logits -= logits.max()
        w = np.exp(logits)
        w /= w.sum()
        weights = {r["id"]: 0.0 for r in rows}
        for r, wi in zip(candidates, w):
            weights[r["id"]] = float(wi)

        best = max(weights, key=weights.get)
        return {
            "selected": best,
            "weights": {k: round(v, 4) for k, v in weights.items()},
            "agents": [{**r, "affinity": round(r["affinity"], 4), "godel": round(r["godel"], 4),
                        "prior": round(r["prior"], 4)} for r in rows],
            "fallback": fallback,
        }

    def feedback(self, agent_id: str, success: float) -> None:
        if agent_id not in self._agents:
            raise KeyError(agent_id)
        self._agents[agent_id].estimator.update(success)

    def stats(self) -> Dict[str, Any]:
        return {aid: {"prior": round(a.estimator.prior, 4), "updates": a.estimator.updates}
                for aid, a in self._agents.items()}

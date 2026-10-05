"""
Budget de contexte dynamique (successeur du DifferentialMemoryAllocator d'AIOTECH 44).

Différence essentielle : AIOTECH 44 calculait un k à partir d'un réseau non entraîné, et les
documents n'étaient jamais transmis au LLM. Ici, seuls les segments retenus sont injectés
dans le prompt, et l'économie est comptée en tokens.

Règle de sélection (k dynamique, couverture gloutonne) :
  1. éligibles : segments dont l'admissibilité individuelle dépasse `entry_floor` et qui
     ne violent aucune contrainte (élimine le bruit sans exiger qu'un segment réponde seul) ;
  2. on ajoute à chaque tour le segment qui apporte le plus grand GAIN DE COUVERTURE de la
     requête (mots discriminants pondérés par IDF, puis gain de `reach` en départage) ;
  3. arrêt dès que le meilleur gain < `min_gain` (les segments restants sont redondants :
     ex. les fiches ORION-13, ORION-14 n'apportent rien à une question sur ORION-12),
     ou budget de tokens / max_chunks atteints ;
  4. l'ensemble retenu S est jugé par la conjonction de Gödel A(S) >= tau. Sinon, le
     contexte est transmis quand même mais signalé `fallback=True` (contexte insuffisant
     ou hors sujet), au lieu de répondre en silence avec un contexte inadéquat.

Une première version jugeait chaque segment isolément : sur les questions comparatives
(deux fiches nécessaires), aucune fiche n'était admissible seule et le rappel tombait à
0 %. Le banc benchmarks/bench_context.py mesure les deux cas séparément.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from aiotech.reasoning.arg import ArgConstraints, ReachabilityGate, SegmentScore
from aiotech.text import chunk_text

DocumentLike = Union[str, Dict[str, Any]]


@dataclass
class ContextSelection:
    selected: List[SegmentScore]
    scores: List[SegmentScore]
    tokens_candidates: int
    tokens_selected: int
    fallback: bool
    tau: float
    set_admissibility: float = 0.0

    @property
    def tokens_saved(self) -> int:
        return self.tokens_candidates - self.tokens_selected

    @property
    def reduction(self) -> float:
        if self.tokens_candidates == 0:
            return 0.0
        return self.tokens_saved / self.tokens_candidates

    def render(self, header: str = "Contexte vérifié (AIOTECH ARG)") -> str:
        if not self.selected:
            return ""
        lines = [f"{header} :"]
        for s in self.selected:
            lines.append(f"[{s.source}] {s.text}")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tau": self.tau,
            "set_admissibility": round(self.set_admissibility, 4),
            "candidates": len(self.scores),
            "selected": len(self.selected),
            "tokens_candidates": self.tokens_candidates,
            "tokens_selected": self.tokens_selected,
            "tokens_saved": self.tokens_saved,
            "reduction": round(self.reduction, 4),
            "fallback": self.fallback,
            "segments": [s.as_dict() for s in self.scores],
        }


@dataclass
class ContextBudgeter:
    gate: ReachabilityGate = field(default_factory=ReachabilityGate)
    token_budget: int = 2000
    max_chunks: int = 8
    min_chunks: int = 1
    chunk_tokens: int = 120
    entry_floor: float = 0.10
    min_gain: float = 0.08

    @staticmethod
    def _normalize_docs(documents: Sequence[DocumentLike]) -> List[Dict[str, Any]]:
        out = []
        for i, d in enumerate(documents):
            if isinstance(d, str):
                out.append({"text": d, "source": f"doc{i}", "metadata": {}})
            else:
                out.append({
                    "text": str(d.get("text", "")),
                    "source": str(d.get("source") or f"doc{i}"),
                    "metadata": dict(d.get("metadata") or {}),
                })
        return out

    def segment(self, documents: Sequence[DocumentLike]):
        texts, sources, metas = [], [], []
        for d in self._normalize_docs(documents):
            chunks = chunk_text(d["text"], self.chunk_tokens) or [d["text"]]
            for j, c in enumerate(chunks):
                texts.append(c)
                sources.append(d["source"] if len(chunks) == 1 else f"{d['source']}#{j}")
                metas.append(d["metadata"])
        return texts, sources, metas

    def select(
        self,
        query: str,
        documents: Sequence[DocumentLike],
        constraints: Optional[ArgConstraints] = None,
    ) -> ContextSelection:
        texts, sources, metas = self.segment(documents)
        if not texts:
            return ContextSelection([], [], 0, 0, False, self.gate.tau)
        an = self.gate.analyze(query, texts, sources, metas, constraints)
        scores = an.scores
        total_tokens = sum(s.tokens for s in scores)

        for s in scores:
            s.admitted = s.constraint > 0.0 and s.admissibility >= self.entry_floor
        pool = [s.index for s in scores if s.admitted]

        chosen: List[int] = []
        used = 0
        cov, reach = 0.0, 0.0
        while pool and len(chosen) < self.max_chunks:
            best, best_key = None, None
            for i in pool:
                if chosen and used + scores[i].tokens > self.token_budget:
                    continue
                g_cov = an.set_coverage(chosen + [i]) - cov
                g_reach = an.set_reach(chosen + [i]) - reach
                key = (round(g_cov, 6), g_reach, scores[i].admissibility)
                if best_key is None or key > best_key:
                    best, best_key = i, key
            if best is None:
                break
            gain = best_key[0] if chosen else max(best_key[0], self.min_gain)
            if gain < self.min_gain:
                break
            chosen.append(best)
            pool.remove(best)
            used += scores[best].tokens
            cov, reach = an.set_coverage(chosen), an.set_reach(chosen)

        if len(chosen) < self.min_chunks:
            # Rien d'éligible : on garde le moins dégradé s'il a au moins un lien avec la question.
            # Une question sans aucun rapport avec les documents ne reçoit aucun contexte.
            for s in sorted(scores, key=lambda s: s.admissibility, reverse=True):
                if len(chosen) >= self.min_chunks:
                    break
                if s.index not in chosen and s.constraint > 0.0 and s.admissibility > 0.0:
                    chosen.append(s.index)
                    used += s.tokens

        set_adm = an.set_admissibility(chosen)
        selected = sorted((scores[i] for i in chosen), key=lambda s: s.index)
        for s in selected:
            s.selected = True
        return ContextSelection(selected, scores, total_tokens, used,
                                fallback=set_adm < self.gate.tau, tau=self.gate.tau,
                                set_admissibility=set_adm)

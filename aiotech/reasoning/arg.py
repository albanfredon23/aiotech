"""
ARG – Admissibility & Reachability Gate (remplace le Spherical Constraint Graph d'AIOTECH 44).

Ce qui change par rapport au SCG :
- AIOTECH 44 projetait chaque état sur l'hypersphère unité S^{D-1} (normalisation L2) et
  mesurait des violations par produit scalaire de vecteurs unitaires. La norme (donc
  l'échelle) était jetée, les seuils n'avaient pas d'unité, et les contraintes étaient des
  vecteurs aléatoires.
- L'ARG ne normalise rien. Chaque critère est une grandeur euclidienne ou logique avec une
  unité explicite, dans [0, 1], et l'admissibilité est leur conjonction de Gödel :

      A(x) = T_G(c_1(x), ..., c_m(x)) = min_i c_i(x)        admissible  <=>  A(x) >= tau

  Un seul critère défaillant suffit à rejeter (c'est la sémantique du "ET" logique).

Critères pour un segment de contexte x face à une requête q :
  reach(q, x)    = 1 - || relu(q - x) ||_2 / || q ||_2
                   part de la masse euclidienne de la requête atteinte par le segment,
                   calculée dans le sous-espace des traits de la requête (pas de sphère).
  coverage(q, x) = somme des IDF des mots de q présents dans x / somme des IDF des mots de q
                   l'IDF est calculée sur les candidats : un mot présent partout ne
                   discrimine rien, un identifiant (ex. "X-12") pèse lourd.
  constraint(x)  = degré de vérité des contraintes explicites (termes requis, interdits,
                   métadonnées).

Chaîne de raisonnement s_1 .. s_H (vérification a posteriori d'une réponse) :
  support(s_h)   = reach(s_h, contexte ∪ requête ∪ s_1..s_{h-1})
  T_G(chaîne)    = min_h support(s_h)   ->  le maillon le plus faible décide.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from aiotech.embeddings import HashingEmbedder
from aiotech.text import content_words, estimate_tokens, normalize, split_sentences


# ── T-normes ──────────────────────────────────────────────────────────────────
def godel_tnorm(values: Sequence[float] | np.ndarray, axis: int = -1) -> np.ndarray | float:
    """T-norme de Gödel : T_G(a_1..a_n) = min(a_1..a_n)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 1.0  # conjonction vide = vrai
    out = np.min(arr, axis=axis)
    return float(out) if np.ndim(out) == 0 else out


def soft_godel(values: Sequence[float] | np.ndarray, temperature: float = 0.05) -> float:
    """Min lissé (log-sum-exp tempéré), utile pour l'apprentissage. Converge vers min quand T -> 0."""
    arr = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    if arr.size == 0:
        return 1.0
    t = max(temperature, 1e-6)
    m = arr.min()
    val = m - t * math.log(np.exp(-(arr - m) / t).sum())
    return float(np.clip(val, 0.0, 1.0))


# ── Contraintes explicites ────────────────────────────────────────────────────
@dataclass
class ArgConstraints:
    """Contraintes vérifiables sur un segment.

    required_terms : chaque terme absent fait baisser le degré (degré = fraction présente)
    forbidden_terms: un seul terme présent -> degré 0 (rejet)
    metadata       : égalité stricte sur les métadonnées du document (ex. {"langue": "fr"})
    """
    required_terms: List[str] = field(default_factory=list)
    forbidden_terms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ArgConstraints":
        data = data or {}
        return cls(
            required_terms=list(data.get("required_terms", [])),
            forbidden_terms=list(data.get("forbidden_terms", [])),
            metadata=dict(data.get("metadata", {})),
        )

    def is_empty(self) -> bool:
        return not (self.required_terms or self.forbidden_terms or self.metadata)

    def degree(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        norm = normalize(text)
        metadata = metadata or {}
        for k, v in self.metadata.items():
            if metadata.get(k) != v:
                return 0.0
        for term in self.forbidden_terms:
            if normalize(term) in norm:
                return 0.0
        if self.required_terms:
            present = sum(1 for t in self.required_terms if normalize(t) in norm)
            return present / len(self.required_terms)
        return 1.0


# ── Résultats ─────────────────────────────────────────────────────────────────
@dataclass
class SegmentScore:
    index: int
    text: str
    source: str
    tokens: int
    reach: float
    coverage: float
    constraint: float
    admissibility: float
    admitted: bool = False
    selected: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index, "source": self.source, "tokens": self.tokens,
            "reach": round(self.reach, 4), "coverage": round(self.coverage, 4),
            "constraint": round(self.constraint, 4),
            "admissibility": round(self.admissibility, 4),
            "admitted": self.admitted, "selected": self.selected,
        }


@dataclass
class ChainReport:
    supports: List[float]
    t_godel: float
    admissible: bool
    weakest_step: Optional[int]
    steps: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "supports": [round(s, 4) for s in self.supports],
            "t_godel": round(self.t_godel, 4),
            "admissible": self.admissible,
            "weakest_step": self.weakest_step,
            "weakest_text": self.steps[self.weakest_step] if self.weakest_step is not None else None,
        }


# ── Porte ARG ─────────────────────────────────────────────────────────────────
class ReachabilityGate:
    def __init__(self, tau: float = 0.35, embedder: Optional[HashingEmbedder] = None):
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau doit être dans [0, 1]")
        self.tau = tau
        # Pas de bigrammes : la requête est réduite à ses mots porteurs de sens,
        # ses bigrammes ne correspondraient pas à ceux du texte source.
        self.embedder = embedder or HashingEmbedder(bigram_weight=0.0)

    # -- critère euclidien ---------------------------------------------------
    @staticmethod
    def reach(q: np.ndarray, x: np.ndarray) -> np.ndarray:
        """reach(q, x) pour x de forme (d,) ou (n, d). Résultat dans [0, 1]."""
        qn = float(np.linalg.norm(q))
        if qn == 0.0:
            return np.ones(x.shape[0]) if x.ndim == 2 else np.float64(1.0)
        deficit = np.maximum(q - x, 0.0)
        dn = np.linalg.norm(deficit, axis=-1)
        return 1.0 - dn / qn

    # -- critère lexical discriminant ---------------------------------------
    @staticmethod
    def _idf(query_words: List[str], segment_word_sets: List[set]) -> Dict[str, float]:
        n = len(segment_word_sets)
        idf = {}
        for w in set(query_words):
            df = sum(1 for s in segment_word_sets if w in s)
            idf[w] = math.log((n + 1.0) / (df + 0.5))
        return idf

    @staticmethod
    def coverage(query_words: List[str], segment_words: set, idf: Dict[str, float]) -> float:
        uniq = set(query_words)
        if not uniq:
            return 1.0
        total = sum(idf[w] for w in uniq)
        if total <= 0:
            return 1.0
        return sum(idf[w] for w in uniq if w in segment_words) / total

    # -- évaluation d'un ensemble de segments --------------------------------
    def analyze(
        self,
        query: str,
        segments: Sequence[str],
        sources: Optional[Sequence[str]] = None,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        constraints: Optional[ArgConstraints] = None,
    ) -> "Analysis":
        """Scores individuels + tout ce qu'il faut pour raisonner sur un ENSEMBLE de segments."""
        sources = list(sources) if sources is not None else [f"doc{i}" for i in range(len(segments))]
        metadatas = list(metadatas) if metadatas is not None else [{} for _ in segments]
        constraints = constraints or ArgConstraints()

        q_words = content_words(query)
        q_vec = self.embedder.embed_one(" ".join(q_words) if q_words else query)
        x_mat = self.embedder.embed(list(segments)) if segments else np.zeros((0, self.embedder.dim), np.float32)
        reach = self.reach(q_vec, x_mat) if len(segments) else np.zeros(0)

        seg_sets = [set(content_words(s)) for s in segments]
        idf = self._idf(q_words, seg_sets)

        scores: List[SegmentScore] = []
        for i, seg in enumerate(segments):
            cov = self.coverage(q_words, seg_sets[i], idf)
            con = constraints.degree(seg, metadatas[i])
            r = float(np.clip(reach[i], 0.0, 1.0))
            adm = float(godel_tnorm([r, cov, con]))
            scores.append(SegmentScore(
                index=i, text=seg, source=sources[i], tokens=estimate_tokens(seg),
                reach=r, coverage=cov, constraint=con, admissibility=adm,
                admitted=adm >= self.tau,
            ))
        return Analysis(self, q_words, q_vec, x_mat, seg_sets, idf, scores)

    def score(self, query: str, segments: Sequence[str], sources: Optional[Sequence[str]] = None,
              metadatas: Optional[Sequence[Dict[str, Any]]] = None,
              constraints: Optional[ArgConstraints] = None) -> List[SegmentScore]:
        return self.analyze(query, segments, sources, metadatas, constraints).scores

    # -- chaîne de raisonnement ---------------------------------------------
    def verify_chain(
        self,
        steps: Sequence[str] | str,
        context: Sequence[str] = (),
        query: str = "",
        tau: Optional[float] = None,
    ) -> ChainReport:
        """Vérifie que chaque étape est atteignable depuis le contexte et les étapes précédentes."""
        if isinstance(steps, str):
            steps = split_sentences(steps)
        steps = [s for s in steps if s.strip()]
        tau = self.tau if tau is None else tau
        if not steps:
            return ChainReport([], 1.0, True, None, [])

        known = np.zeros(self.embedder.dim, dtype=np.float32)
        for src in list(context) + ([query] if query else []):
            known = np.maximum(known, self.embedder.embed_one(src))

        supports: List[float] = []
        for step in steps:
            words = content_words(step)
            if not words:
                supports.append(1.0)  # étape de liaison ("Donc :"), rien à vérifier
            else:
                s_vec = self.embedder.embed_one(" ".join(words))
                supports.append(float(np.clip(self.reach(s_vec, known), 0.0, 1.0)))
            known = np.maximum(known, self.embedder.embed_one(step))

        t = float(godel_tnorm(supports))
        weakest = int(np.argmin(supports))
        return ChainReport(supports, t, t >= tau, weakest, list(steps))


@dataclass
class Analysis:
    """Résultat de `ReachabilityGate.analyze` : permet d'évaluer un ensemble S de segments.

    Admissibilité d'ensemble (questions multi-documents) :
        A(S) = T_G( reach(q, ∪S), coverage(q, ∪S), min_{x∈S} constraint(x) )
    où ∪S est le maximum élément par élément des vecteurs (traits atteints par au moins
    un segment) et l'union des mots. Une question comparant deux produits est admissible
    avec les DEUX fiches, alors qu'aucune fiche seule ne l'est.
    """
    gate: ReachabilityGate
    q_words: List[str]
    q_vec: np.ndarray
    x_mat: np.ndarray
    seg_sets: List[set]
    idf: Dict[str, float]
    scores: List[SegmentScore]

    def set_reach(self, indices: Sequence[int]) -> float:
        if not len(indices):
            return 0.0
        union = np.max(self.x_mat[list(indices)], axis=0)
        return float(np.clip(ReachabilityGate.reach(self.q_vec, union), 0.0, 1.0))

    def set_coverage(self, indices: Sequence[int]) -> float:
        words: set = set()
        for i in indices:
            words |= self.seg_sets[i]
        return ReachabilityGate.coverage(self.q_words, words, self.idf)

    def set_admissibility(self, indices: Sequence[int]) -> float:
        if not len(indices):
            return 0.0
        con = min(self.scores[i].constraint for i in indices)
        return float(godel_tnorm([self.set_reach(indices), self.set_coverage(indices), con]))

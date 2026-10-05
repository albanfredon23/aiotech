"""
Embeddings.

`HashingEmbedder` : embedding lexical local, déterministe, sans modèle ni appel réseau
(hachage de mots, bigrammes et trigrammes de caractères, pondération TF sous-linéaire).
Contrairement au cache v3 (vecteur aléatoire tiré d'un SHA-256, donc sans aucune notion
de similarité), deux textes qui partagent du vocabulaire ont ici des vecteurs proches.

Les vecteurs ne sont PAS projetés sur la sphère unité : l'ARG travaille en distance
euclidienne avec une échelle calibrée sur les candidats (voir reasoning/arg.py).

`LiteLLMEmbedder` : embeddings d'un fournisseur via LiteLLM (optionnel).
"""
from __future__ import annotations

import hashlib
import math
from typing import List, Optional, Protocol, Sequence

import numpy as np

from aiotech.text import stem, words


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # (n, dim)
        ...


def _bucket(feature: str, dim: int) -> int:
    h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % dim


class HashingEmbedder:
    def __init__(self, dim: int = 2048, char_ngrams: int = 3, word_weight: float = 1.0,
                 bigram_weight: float = 0.7, char_weight: float = 0.25):
        self.dim = dim
        self.char_ngrams = char_ngrams
        self.word_weight = word_weight
        self.bigram_weight = bigram_weight
        self.char_weight = char_weight

    def _features(self, text: str) -> dict:
        feats: dict = {}
        toks = words(text)
        for w in toks:
            k = "w:" + stem(w)
            feats[k] = feats.get(k, 0.0) + self.word_weight
        for a, b in zip(toks, toks[1:]):
            k = "b:" + a + "_" + b
            feats[k] = feats.get(k, 0.0) + self.bigram_weight
        n = self.char_ngrams
        for w in toks:
            padded = f" {w} "
            for i in range(max(0, len(padded) - n + 1)):
                k = "c:" + padded[i:i + n]
                feats[k] = feats.get(k, 0.0) + self.char_weight
        return feats

    def embed_one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for feat, tf in self._features(text).items():
            # TF sous-linéaire : 1 + log(tf) atténue les répétitions
            v[_bucket(feat, self.dim)] += 1.0 + math.log(tf) if tf >= 1.0 else tf
        return v

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.embed_one(t) for t in texts])


class LiteLLMEmbedder:
    """Embeddings fournisseur (ex. "text-embedding-3-small", "mistral/mistral-embed").

    Attention : chaque appel est facturé par le fournisseur. Le coût est à intégrer
    dans `variable_cost_per_request` du modèle économique.
    """

    def __init__(self, model: str, dim: Optional[int] = None):
        try:
            import litellm  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("litellm n'est pas installé : pip install litellm") from exc
        self.model = model
        self.dim = dim or 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - réseau
        import litellm
        resp = litellm.embedding(model=self.model, input=list(texts))
        vecs = [np.asarray(d["embedding"], dtype=np.float32) for d in resp.data]
        out = np.stack(vecs)
        self.dim = out.shape[1]
        return out


def weighted_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Similarité de Jaccard pondérée sur vecteurs positifs : sum(min) / sum(max), dans [0, 1].

    Mesure de recouvrement lexical, sans géométrie sphérique (pas de cosinus).
    """
    num = float(np.minimum(a, b).sum())
    den = float(np.maximum(a, b).sum())
    return num / den if den > 0 else 0.0


def pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distances euclidiennes au carré entre lignes de x (n,d) et y (m,d) -> (n,m)."""
    x2 = np.sum(x * x, axis=1)[:, None]
    y2 = np.sum(y * y, axis=1)[None, :]
    d = x2 + y2 - 2.0 * (x @ y.T)
    return np.maximum(d, 0.0)


def default_embedder() -> HashingEmbedder:
    return HashingEmbedder()


__all__: List[str] = [
    "Embedder", "HashingEmbedder", "LiteLLMEmbedder", "weighted_jaccard",
    "pairwise_sq_dists", "default_embedder",
]

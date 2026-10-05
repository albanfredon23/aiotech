"""
Tarifs par million de tokens (USD) et calcul de coût.

Les tarifs Claude ci-dessous sont les tarifs publics de l'API Anthropic relevés le
2026-09-25. Les tarifs évoluent : vérifiez-les et surchargez-les via la variable
d'environnement AIOTECH_PRICES (JSON), par exemple :

    AIOTECH_PRICES='{"openai/gpt-x": [1.25, 10.0], "mistral/mistral-large-latest": [2.0, 6.0]}'

Un modèle absent de la table a un coût inconnu (None) : aucun montant n'est inventé.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

PRICES_AS_OF = "2026-09-25"

DEFAULT_PRICES: Dict[str, Tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def _strip_provider(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


class PriceTable:
    def __init__(self, prices: Optional[Dict[str, Tuple[float, float]]] = None, use_env: bool = True):
        self._prices: Dict[str, Tuple[float, float]] = dict(DEFAULT_PRICES)
        if prices:
            self._prices.update({k: (float(v[0]), float(v[1])) for k, v in prices.items()})
        if use_env and os.getenv("AIOTECH_PRICES"):
            try:
                extra = json.loads(os.environ["AIOTECH_PRICES"])
                self._prices.update({k: (float(v[0]), float(v[1])) for k, v in extra.items()})
            except (ValueError, TypeError, IndexError):
                pass

    def get(self, model: str) -> Optional[Tuple[float, float]]:
        return self._prices.get(model) or self._prices.get(_strip_provider(model))

    def cost(self, model: str, tokens_in: int, tokens_out: int) -> Optional[float]:
        p = self.get(model)
        if p is None:
            return None
        return (tokens_in * p[0] + tokens_out * p[1]) / 1_000_000.0

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        return {k: {"input_per_mtok": v[0], "output_per_mtok": v[1]} for k, v in self._prices.items()}

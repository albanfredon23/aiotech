"""
Routage de modèle par complexité estimée : requête simple -> modèle économique,
requête complexe -> modèle phare.

v3 utilisait surtout le nombre de caractères distincts du texte (presque toute phrase de
25 lettres différentes obtenait le même score). Les signaux ci-dessous sont explicables et
renvoyés dans `explain()`. C'est une heuristique : le seuil doit être calibré sur le trafic
réel (part de réponses jugées correctes par modèle).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from aiotech.text import normalize, words

_REASONING_MARKERS = {
    # FR
    "pourquoi", "demontre", "demontrer", "prouve", "prouver", "justifie", "analyse", "analyser",
    "compare", "comparer", "explique", "expliquer", "optimise", "optimiser", "calcule", "calculer",
    "derive", "deriver", "integre", "resous", "resoudre", "strategie", "architecture", "etape",
    "etapes", "hypothese", "implications", "evalue", "evaluer", "concois", "concevoir",
    # EN
    "why", "prove", "proof", "derive", "analyze", "analyse", "compare", "explain", "optimize",
    "calculate", "solve", "strategy", "architecture", "step", "steps", "hypothesis", "evaluate",
    "design", "tradeoff", "tradeoffs", "theorem", "algorithm", "debug", "refactor",
}
_MATH_RE = re.compile(r"[=+\-*/^∑∫√≤≥<>]|\d+\s*%")
_CODE_RE = re.compile(r"```|def |class |function |SELECT |#include|=>|\{\s*\n")


class ModelSelector:
    def __init__(self, cheap_model: str, flagship_model: str, threshold: float = 0.45):
        self.cheap = cheap_model
        self.flagship = flagship_model
        self.threshold = threshold

    @staticmethod
    def signals(text: str) -> Dict[str, float]:
        toks = words(text)
        n = len(toks)
        norm = normalize(text)
        markers = sum(1 for w in toks if w in _REASONING_MARKERS)
        return {
            # longueur : 0 pour 10 mots, 1 au-delà de ~400 mots
            "length": min(max((n - 10) / 390.0, 0.0), 1.0),
            "reasoning_markers": min(markers / 3.0, 1.0),
            "sub_questions": min(max(text.count("?") - 1, 0) / 3.0, 1.0),
            "math": min(len(_MATH_RE.findall(text)) / 6.0, 1.0),
            "code": 1.0 if _CODE_RE.search(text) else 0.0,
            "enumeration": 1.0 if re.search(r"(^|\n)\s*(\d+[.)]|-|\*)\s", text) else 0.0,
            "_n_words": float(n),
            "_has_constraints": 1.0 if any(k in norm for k in ("sans ", "uniquement", "au moins", "at most", "must", "doit")) else 0.0,
        }

    _WEIGHTS = {"length": 0.25, "reasoning_markers": 0.30, "sub_questions": 0.10,
                "math": 0.15, "code": 0.15, "enumeration": 0.05}

    def complexity(self, text: str) -> float:
        s = self.signals(text)
        score = sum(self._WEIGHTS[k] * s[k] for k in self._WEIGHTS)
        return round(min(score + 0.05 * s["_has_constraints"], 1.0), 4)

    def select(self, messages: List[Dict[str, str]], forced_model: Optional[str] = None) -> str:
        if forced_model:
            return forced_model
        return self.flagship if self.complexity(self._user_text(messages)) >= self.threshold else self.cheap

    def explain(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        text = self._user_text(messages)
        c = self.complexity(text)
        return {"complexity": c, "threshold": self.threshold,
                "selected": self.flagship if c >= self.threshold else self.cheap,
                "signals": {k: round(v, 3) for k, v in self.signals(text).items() if not k.startswith("_")}}

    @staticmethod
    def _user_text(messages: List[Dict[str, str]]) -> str:
        # On évalue la demande de l'utilisateur, pas le contexte injecté ni le prompt système.
        users = [m.get("content") or "" for m in messages if m.get("role") == "user"]
        return users[-1] if users else ""

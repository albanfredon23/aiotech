"""
Garde-fous d'entrée (PromptShield) et de sortie (OutputValidator).

Corrections par rapport à v3 :
- motifs FR + EN pour l'injection de prompt ;
- "system prompt" n'est plus bloqué tout seul (faux positifs sur des questions légitimes) :
  seule une demande de révélation/contournement l'est ;
- l'ordre des motifs PII est corrigé (une adresse IP ou une carte n'est plus étiquetée
  "téléphone") et les cartes bancaires sont confirmées par l'algorithme de Luhn ;
- la validation JSON cherche le premier objet JSON réellement décodable au lieu d'une
  expression régulière gloutonne.

Ces filtres sont des heuristiques : ils réduisent les risques évidents, ils ne remplacent
pas une politique de sécurité du fournisseur ni un modèle de modération dédié.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


def _luhn_ok(digits: str) -> bool:
    nums = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total = 0
    for i, n in enumerate(reversed(nums)):
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# Ordre important : du plus spécifique au plus générique.
_PII_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    ("api_key", re.compile(r"\b(?:sk-ant-[A-Za-z0-9\-_]{16,}|sk-[A-Za-z0-9\-_]{16,}|aio-[A-Za-z0-9\-_]{16,})"), "[API_KEY]"),
    ("email", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,3})?\b"), "[IBAN]"),
    ("card", re.compile(r"\b(?:\d[ \-]?){12,18}\d\b"), "[CARD]"),
    ("ip", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), "[IP]"),
    ("phone", re.compile(r"(?<![\w.])\+?\(?\d[\d .\-()]{7,}\d(?![\w.])"), "[PHONE]"),
]

_INJECTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"ignore[sz]?\s+(?:all\s+|toutes?\s+)?(?:the\s+|les\s+)?(?:previous|prior|above|pr[ée]c[ée]dentes?)\s+(?:instructions?|consignes?|r[èe]gles?)",
    r"ignore\s+(?:all\s+)?previous\s+instructions?",
    r"ignore[sz]?\s+(?:toutes?\s+)?(?:les|tes|vos)\s+(?:instructions?|consignes?|r[èe]gles?)\s+(?:pr[ée]c[ée]dentes?|ci-dessus|ant[ée]rieures?)",
    r"forget\s+(?:all\s+)?(?:previous|your)\s+(?:instructions?|rules?)",
    r"oublie[sz]?\s+(?:toutes?\s+)?(?:tes|vos|les)\s+(?:instructions?|consignes?|r[èe]gles?)",
    r"(?:reveal|print|show|display|leak|r[ée]v[èe]le|affiche|montre)\w*\s+(?:me\s+|moi\s+)?(?:your|the|ton|votre|le)\s+(?:system\s+prompt|prompt\s+syst[èe]me|instructions?\s+(?:cach[ée]es|syst[èe]me))",
    r"you\s+are\s+now\s+(?:a\s+|an\s+)?(?:dan|jailbroken|unrestricted|evil)",
    r"disregard\s+(?:your\s+|all\s+)?(?:rules?|guidelines?|restrictions?|safety)",
    r"(?:bypass|override|d[ée]sactive)\w*\s+(?:the\s+|les?\s+)?(?:safety|security|filtres?|s[ée]curit[ée]|garde-fous?)",
]]

_MODERATION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\b(?:make|build|synthesi[sz]e|fabriquer|synth[ée]tiser)\b.{0,40}\b(?:bomb|bombe|explosi\w+|bioweapon|arme\s+(?:chimique|biologique)|nerve\s+agent)\b",
    r"\b(?:bomb|bombe|explosi\w+|bioweapon|arme\s+(?:chimique|biologique))\b.{0,40}\b(?:make|build|synthesi[sz]e|fabriquer|synth[ée]tiser|recette|recipe)\b",
    r"\b(?:write|code|create|generate|[ée]cri[st]|cr[ée]e)\b.{0,40}\b(?:ransomware|keylogger|malware)\b",
]]


class PromptShield:
    def __init__(self, block_injection: bool = True, scrub_pii: bool = True, block_harmful: bool = True):
        self.block_injection = block_injection
        self.scrub_pii = scrub_pii
        self.block_harmful = block_harmful
        self._stats = {"injections_blocked": 0, "pii_scrubbed": 0, "harmful_blocked": 0, "clean": 0}

    def analyze(self, text: str) -> Dict[str, Any]:
        """Retourne {safe, reason, cleaned, pii_found}."""
        if self.block_injection and any(p.search(text) for p in _INJECTION_PATTERNS):
            self._stats["injections_blocked"] += 1
            return {"safe": False, "reason": "prompt_injection", "cleaned": text, "pii_found": []}
        if self.block_harmful and any(p.search(text) for p in _MODERATION_PATTERNS):
            self._stats["harmful_blocked"] += 1
            return {"safe": False, "reason": "harmful_content", "cleaned": text, "pii_found": []}

        cleaned = text
        found: List[Dict[str, Any]] = []
        if self.scrub_pii:
            for kind, pat, repl in _PII_PATTERNS:
                count = 0

                def _sub(m: re.Match, kind=kind, repl=repl) -> str:
                    nonlocal count
                    if kind == "card" and not _luhn_ok(m.group(0)):
                        return m.group(0)
                    if kind == "phone" and sum(c.isdigit() for c in m.group(0)) < 9:
                        return m.group(0)  # trop court pour un numéro : années, montants...
                    count += 1
                    return repl

                cleaned = pat.sub(_sub, cleaned)
                if count:
                    found.append({"type": kind, "count": count})
            if found:
                self._stats["pii_scrubbed"] += sum(f["count"] for f in found)
        self._stats["clean"] += 1
        return {"safe": True, "reason": "ok", "cleaned": cleaned, "pii_found": found}

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)


class OutputValidator:
    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self._stats = {"validated": 0, "failed": 0, "retries": 0}

    @staticmethod
    def _first_json_object(text: str) -> Tuple[Optional[Any], Optional[str]]:
        decoder = json.JSONDecoder()
        idx = text.find("{")
        last_error = "no_json_found"
        while idx != -1:
            try:
                obj, _ = decoder.raw_decode(text, idx)
                if isinstance(obj, dict):
                    return obj, None
            except json.JSONDecodeError as exc:
                last_error = f"invalid_json: {exc.msg}"
            idx = text.find("{", idx + 1)
        return None, last_error

    def validate_json(self, text: str, required_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        data, error = self._first_json_object(text)
        if data is None:
            self._stats["failed"] += 1
            return {"valid": False, "data": None, "error": error}
        missing = [k for k in (required_keys or []) if k not in data]
        if missing:
            self._stats["failed"] += 1
            return {"valid": False, "data": data, "error": f"missing_keys: {missing}"}
        self._stats["validated"] += 1
        return {"valid": True, "data": data, "error": None}

    def build_retry_prompt(self, error: str, schema_hint: str = "") -> str:
        self._stats["retries"] += 1
        hint = f" Schéma attendu : {schema_hint}." if schema_hint else ""
        return (f"[CORRECTION REQUISE] Ta réponse précédente était invalide ({error}).{hint} "
                "Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ni après.")

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

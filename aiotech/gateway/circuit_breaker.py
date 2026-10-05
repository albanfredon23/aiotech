"""
Disjoncteur par fournisseur : CLOSED -> OPEN (trop d'échecs) -> HALF_OPEN (une sonde) -> CLOSED.

Correction v3 : tout modèle sans "/" était classé "openai", y compris les modèles Claude ;
le repli de GPT vers Claude restait donc bloqué sur le même disjoncteur ouvert.
"""
from __future__ import annotations

import time
from typing import Dict


def provider_of(model: str) -> str:
    m = model.lower()
    if "/" in m:
        return m.split("/", 1)[0]
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("gpt", "o1", "o3", "o4", "text-embedding", "chatgpt")):
        return "openai"
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith(("mistral", "codestral", "pixtral")):
        return "mistral"
    return "unknown"


class CircuitBreaker:
    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, failure_threshold: int = 3, recovery_seconds: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self._failures: Dict[str, int] = {}
        self._opened_at: Dict[str, float] = {}
        self._state: Dict[str, str] = {}
        self._probe_in_flight: Dict[str, bool] = {}

    def state(self, provider: str) -> str:
        s = self._state.get(provider, self.CLOSED)
        if s == self.OPEN and time.time() - self._opened_at.get(provider, 0.0) >= self.recovery_seconds:
            self._state[provider] = self.HALF_OPEN
            self._probe_in_flight[provider] = False
            return self.HALF_OPEN
        return s

    def is_available(self, provider: str) -> bool:
        s = self.state(provider)
        if s == self.CLOSED:
            return True
        if s == self.HALF_OPEN and not self._probe_in_flight.get(provider, False):
            return True
        return False

    def acquire(self, provider: str) -> bool:
        """À appeler juste avant l'appel réseau : en HALF_OPEN, une seule sonde passe."""
        if not self.is_available(provider):
            return False
        if self.state(provider) == self.HALF_OPEN:
            self._probe_in_flight[provider] = True
        return True

    def record_success(self, provider: str) -> None:
        self._failures[provider] = 0
        self._state[provider] = self.CLOSED
        self._probe_in_flight[provider] = False

    def record_failure(self, provider: str) -> None:
        if self.state(provider) == self.HALF_OPEN:
            self._state[provider] = self.OPEN
            self._opened_at[provider] = time.time()
            self._probe_in_flight[provider] = False
            return
        self._failures[provider] = self._failures.get(provider, 0) + 1
        self._state.setdefault(provider, self.CLOSED)
        if self._failures[provider] >= self.failure_threshold:
            self._state[provider] = self.OPEN
            self._opened_at[provider] = time.time()

    def all_states(self) -> Dict[str, str]:
        return {p: self.state(p) for p in list(self._state)}

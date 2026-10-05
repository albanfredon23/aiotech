"""
Configuration centralisée, lue depuis les variables d'environnement (voir .env.example).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "oui"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default)
    return [x.strip() for x in raw.split(",") if x.strip()]


@dataclass
class Settings:
    # Routage de modèles (identifiants LiteLLM, préfixe fournisseur recommandé)
    flagship_model: str = "anthropic/claude-opus-5-5"
    cheap_model: str = "anthropic/claude-haiku-4-5"
    fallback_models: List[str] = field(default_factory=lambda: ["anthropic/claude-sonnet-5-5"])
    router_threshold: float = 0.45

    # ARG (Admissibility & Reachability Gate)
    arg_tau: float = 0.35
    arg_max_chunks: int = 8
    arg_min_chunks: int = 1
    context_token_budget: int = 2000
    chunk_tokens: int = 120

    # Cache
    cache_threshold: float = 0.85
    cache_ttl_seconds: float = 24 * 3600
    cache_max_entries: int = 5000

    # Disjoncteur
    cb_failure_threshold: int = 3
    cb_recovery_seconds: float = 30.0

    # Sécurité / API
    admin_token: str = ""
    require_api_key: bool = True
    enable_tools: bool = False
    tools_dir: str = "./sandbox"
    db_path: str = "./aiotech.sqlite3"

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            flagship_model=os.getenv("AIOTECH_FLAGSHIP_MODEL", d.flagship_model),
            cheap_model=os.getenv("AIOTECH_CHEAP_MODEL", d.cheap_model),
            fallback_models=_env_list("AIOTECH_FALLBACK_MODELS", d.fallback_models),
            router_threshold=_env_float("AIOTECH_ROUTER_THRESHOLD", d.router_threshold),
            arg_tau=_env_float("AIOTECH_ARG_TAU", d.arg_tau),
            arg_max_chunks=_env_int("AIOTECH_ARG_MAX_CHUNKS", d.arg_max_chunks),
            arg_min_chunks=_env_int("AIOTECH_ARG_MIN_CHUNKS", d.arg_min_chunks),
            context_token_budget=_env_int("AIOTECH_CONTEXT_TOKEN_BUDGET", d.context_token_budget),
            chunk_tokens=_env_int("AIOTECH_CHUNK_TOKENS", d.chunk_tokens),
            cache_threshold=_env_float("AIOTECH_CACHE_THRESHOLD", d.cache_threshold),
            cache_ttl_seconds=_env_float("AIOTECH_CACHE_TTL_SECONDS", d.cache_ttl_seconds),
            cache_max_entries=_env_int("AIOTECH_CACHE_MAX_ENTRIES", d.cache_max_entries),
            cb_failure_threshold=_env_int("AIOTECH_CB_FAILURE_THRESHOLD", d.cb_failure_threshold),
            cb_recovery_seconds=_env_float("AIOTECH_CB_RECOVERY_SECONDS", d.cb_recovery_seconds),
            admin_token=os.getenv("AIOTECH_ADMIN_TOKEN", d.admin_token),
            require_api_key=_env_bool("AIOTECH_REQUIRE_API_KEY", d.require_api_key),
            enable_tools=_env_bool("AIOTECH_ENABLE_TOOLS", d.enable_tools),
            tools_dir=os.getenv("AIOTECH_TOOLS_DIR", d.tools_dir),
            db_path=os.getenv("AIOTECH_DB_PATH", d.db_path),
        )

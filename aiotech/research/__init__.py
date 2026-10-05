"""
Modules de recherche différentiables (PyTorch requis : pip install torch).

    from aiotech.research.arg_core import ARGCore
    from aiotech.research.compiler import search
"""
try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - dépend de l'environnement
    TORCH_AVAILABLE = False

__all__ = ["TORCH_AVAILABLE"]

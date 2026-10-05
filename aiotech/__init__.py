"""
AIOTECH 45 – middleware de raisonnement contraint et d'économie de tokens pour LLM.

Le paquet fonctionne avec numpy seul. FastAPI, LiteLLM et PyTorch sont optionnels :
- fastapi/uvicorn  -> serveur HTTP (aiotech.api)
- litellm          -> appels réels aux fournisseurs LLM (sinon client "stub" déterministe)
- torch            -> module de recherche différentiable (aiotech.research)
"""

__version__ = "45.0.0"

from aiotech.reasoning.arg import ReachabilityGate, ArgConstraints, godel_tnorm
from aiotech.reasoning.context_budget import ContextBudgeter
from aiotech.economics import Scenario, monthly_costs, cost_curve

__all__ = [
    "__version__",
    "ReachabilityGate",
    "ArgConstraints",
    "godel_tnorm",
    "ContextBudgeter",
    "Scenario",
    "monthly_costs",
    "cost_curve",
]

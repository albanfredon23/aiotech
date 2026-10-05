"""
Démonstration hors ligne (aucune clé API nécessaire) :
    python examples/quickstart.py

Avec une vraie clé : définir ANTHROPIC_API_KEY (ou autre) et AIOTECH_OFFLINE=0.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AIOTECH_OFFLINE", "1")

from aiotech.config import Settings  # noqa: E402
from aiotech.economics import Scenario, monthly_costs  # noqa: E402
from aiotech.gateway import AiotechClient  # noqa: E402

DOCS = [
    {"source": "fiche-orion-12", "text": "Fiche du module ORION-12. Poids : 4,2 kg. Fabriqué à Lyon. Garantie : 3 ans."},
    {"source": "fiche-orion-13", "text": "Fiche du module ORION-13. Poids : 7,9 kg. Fabriqué à Lille. Garantie : 2 ans."},
    {"source": "fiche-vega-20", "text": "Fiche du module VEGA-20. Poids : 5,1 kg. Fabriqué à Nantes. Garantie : 5 ans."},
    {"source": "retours", "text": "Politique de retours : les commandes sont retournables sous 30 jours."},
]

client = AiotechClient(Settings(db_path=":memory:"))
for question in ["Quel est le poids du module ORION-12 ?",
                 "Lequel est le plus lourd : ORION-12 ou VEGA-20 ?",
                 "Quel est le poids du module ORION-12 ?"]:
    r = client.completion_sync([{"role": "user", "content": question}], documents=DOCS)
    ctx = r["context"]
    print(f"\n> {question}")
    print(f"  modèle : {r['model']} | cache : {r['cache_hit']}")
    print(f"  contexte : {ctx['selected']}/{ctx['candidates']} segments, "
          f"{ctx['tokens_selected']} tokens au lieu de {ctx['tokens_candidates']} (-{ctx['reduction']:.0%})")
    print(f"  coût : {r['cost_usd']} $ | référence : {r['baseline_cost_usd']} $ | économie : {r['saved_usd']} $")

print("\nMétriques cumulées :")
m = client.metrics
print(json.dumps({k: m[k] for k in ("requests", "cache_hits", "context_tokens_saved", "cost_usd",
                                     "baseline_cost_usd", "saved_usd", "saved_pct")}, indent=2))

print("\nProjection mensuelle (scénario par défaut, à adapter) :")
print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in monthly_costs(Scenario()).items()}, indent=2))

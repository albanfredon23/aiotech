"""
Banc 2 – cache d'équivalence lexicale contextuelle : taux de réponses réutilisées et réponses ERRONÉES servies.

Flux de 3 000 requêtes tirées selon une loi de Zipf (s = 1,1) sur les questions du corpus,
chacune sous une de 5 formulations de surface. Une réponse servie depuis le cache est
"correcte" si la question d'origine est la même, "erronée" sinon (ex. poids de ORION-12
servi pour ORION-13). On compare avec et sans la garde sur les identifiants.

Exécution :  python -m benchmarks.bench_cache
"""
from __future__ import annotations

import json
import os
import random
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiotech.gateway.cache import SemanticCache  # noqa: E402
from benchmarks.corpus import build_corpus, paraphrases  # noqa: E402


def run(n_requests: int = 3000, threshold: float = 0.85, zipf_s: float = 1.1, seed: int = 3,
        identifier_guard: bool = True) -> Dict[str, float]:
    rng = random.Random(seed)
    questions = build_corpus()["questions"]
    weights = [1.0 / (i + 1) ** zipf_s for i in range(len(questions))]
    cache = SemanticCache(threshold=threshold, max_entries=100_000, identifier_guard=identifier_guard)
    hits = wrong = 0
    for _ in range(n_requests):
        qi = rng.choices(range(len(questions)), weights=weights)[0]
        text = rng.choice(paraphrases(questions[qi]))
        hit = cache.lookup("m", text, "")
        if hit:
            hits += 1
            wrong += hit["content"] != f"ANS::{qi}"
        else:
            cache.store("m", text, f"ANS::{qi}", 100, 50)
    return {"requests": n_requests, "threshold": threshold, "identifier_guard": identifier_guard,
            "hit_rate": hits / n_requests, "wrong_answer_rate": wrong / n_requests,
            "wrong_among_hits": wrong / hits if hits else 0.0}


def run_all() -> List[Dict[str, float]]:
    return [run(identifier_guard=True), run(identifier_guard=False)]


def main() -> None:
    results = run_all()
    for r in results:
        print(f"garde identifiants={'oui' if r['identifier_guard'] else 'non'} | seuil {r['threshold']} | "
              f"taux de cache {r['hit_rate']:.1%} | réponses erronées {r['wrong_answer_rate']:.2%} "
              f"({r['wrong_among_hits']:.1%} des hits)")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_cache.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

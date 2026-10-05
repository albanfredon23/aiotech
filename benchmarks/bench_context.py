"""
Banc 1 – réduction du contexte par l'ARG, à information utile conservée.

Protocole :
  1. un premier étage de recherche (TF-IDF) ramène les N=8 meilleurs documents par
     question : c'est ce qu'un RAG standard envoie au LLM (référence) ;
  2. l'ARG filtre ces 8 candidats (conjonction de Gödel, k dynamique) ;
  3. on mesure les tokens envoyés et si le document qui contient la réponse ("gold") est
     encore présent (rappel), pour plusieurs seuils tau.

Exécution :  python -m benchmarks.bench_context   (depuis la racine du projet)
Sortie     :  tableau + benchmarks/results_context.json
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiotech.reasoning.arg import ReachabilityGate  # noqa: E402
from aiotech.reasoning.context_budget import ContextBudgeter  # noqa: E402
from aiotech.text import content_words, estimate_tokens  # noqa: E402
from benchmarks.corpus import build_corpus  # noqa: E402

TOP_N = 8


class TfIdfRetriever:
    """Premier étage de recherche, volontairement standard (TF-IDF, produit scalaire)."""

    def __init__(self, docs: List[Dict[str, object]]):
        self.docs = docs
        self.tfs = [Counter(content_words(str(d["text"]))) for d in docs]
        df = Counter(w for tf in self.tfs for w in tf)
        n = len(docs)
        self.idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}

    def top(self, query: str, n: int) -> List[Dict[str, object]]:
        q = Counter(content_words(query))
        scores = []
        for i, tf in enumerate(self.tfs):
            s = sum(qc * (1 + math.log(tf[w])) * self.idf.get(w, 0.0) ** 2 for w, qc in q.items() if tf.get(w))
            scores.append((s, i))
        scores.sort(reverse=True)
        return [self.docs[i] for _, i in scores[:n]]


def run(taus=(0.2, 0.3, 0.35, 0.4, 0.5), min_gain: float = 0.08, entry_floor: float = 0.10) -> Dict[str, object]:
    corpus = build_corpus()
    docs, questions = corpus["documents"], corpus["questions"]
    retriever = TfIdfRetriever(docs)

    candidates = [retriever.top(q.text, TOP_N) for q in questions]
    base_tokens = [sum(estimate_tokens(str(d["text"])) for d in c) for c in candidates]
    base_recall = [set(q.gold) <= {d["source"] for d in c} for q, c in zip(questions, candidates)]

    def summarize(idx: List[int], sel_tokens, recall, ks, fallbacks) -> Dict[str, float]:
        n = len(idx)
        bt = sum(base_tokens[i] for i in idx)
        st = sum(sel_tokens[i] for i in idx)
        return {
            "n": n,
            "tokens_baseline_avg": bt / n,
            "tokens_arg_avg": st / n,
            "reduction": 1 - st / bt,
            "recall_baseline": sum(base_recall[i] for i in idx) / n,
            "recall_arg": sum(recall[i] for i in idx) / n,
            "k_avg": sum(ks[i] for i in idx) / n,
            "fallback_rate": sum(fallbacks[i] for i in idx) / n,
        }

    single = [i for i, q in enumerate(questions) if not q.multi_doc]
    multi = [i for i, q in enumerate(questions) if q.multi_doc]
    rows = []
    for tau in taus:
        budgeter = ContextBudgeter(gate=ReachabilityGate(tau=tau), token_budget=10_000, max_chunks=TOP_N,
                                   chunk_tokens=400, min_gain=min_gain, entry_floor=entry_floor)
        sel_tokens, recall, ks, fallbacks = [], [], [], []
        for q, cands in zip(questions, candidates):
            sel = budgeter.select(q.text, cands)
            sel_tokens.append(sel.tokens_selected)
            kept = {s.source.split("#")[0] for s in sel.selected}
            recall.append(set(q.gold) <= kept)
            ks.append(len(sel.selected))
            fallbacks.append(int(sel.fallback))
        everything = list(range(len(questions)))
        rows.append({
            "tau": tau,
            "all": summarize(everything, sel_tokens, recall, ks, fallbacks),
            "single_doc": summarize(single, sel_tokens, recall, ks, fallbacks),
            "two_docs": summarize(multi, sel_tokens, recall, ks, fallbacks),
        })
    return {"n_questions": len(questions), "n_single": len(single), "n_two_docs": len(multi),
            "n_documents": len(docs), "top_n": TOP_N, "min_gain": min_gain, "entry_floor": entry_floor,
            "rows": rows}


def main() -> None:
    res = run()
    print(f"Corpus : {res['n_documents']} documents, {res['n_questions']} questions "
          f"({res['n_single']} à 1 document, {res['n_two_docs']} à 2 documents), top-{res['top_n']} candidats\n")
    print(f"{'tau':>5} | {'groupe':>10} | {'tokens réf.':>11} | {'tokens ARG':>10} | {'réduction':>9} | "
          f"{'rappel réf.':>11} | {'rappel ARG':>10} | {'k moyen':>7} | {'repli':>6}")
    print("-" * 106)
    for r in res["rows"]:
        for group in ("all", "single_doc", "two_docs"):
            g = r[group]
            print(f"{r['tau']:>5.2f} | {group:>10} | {g['tokens_baseline_avg']:>11.1f} | {g['tokens_arg_avg']:>10.1f} | "
                  f"{g['reduction']:>8.1%} | {g['recall_baseline']:>10.1%} | {g['recall_arg']:>9.1%} | "
                  f"{g['k_avg']:>7.2f} | {g['fallback_rate']:>5.1%}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_context.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\nRésultats écrits dans {out}")


if __name__ == "__main__":
    main()

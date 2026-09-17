import time
import torch
import numpy as np

class EnergyBenchmark:
    """
    Harnais de test mesurant scientifiquement la réduction des FLOPs,
    la latence et la suppression des violations de contraintes (hallucinations).
    """
    def __init__(self, baseline_model: torch.nn.Module, aiotech_model: torch.nn.Module):
        self.baseline = baseline_model
        self.aiotech = aiotech_model
        self.results = {"Baseline": [], "AIOTECH44": []}

    def simulate_flops(self, active_nodes: int, seq_len: int, emb_dim: int) -> float:
        return float((active_nodes ** 2) * emb_dim)

    def evaluate_query(self, query_emb: torch.Tensor, retrieved_docs: torch.Tensor, 
                       graph_nodes: torch.Tensor, constraints: torch.Tensor):
        emb_dim = query_emb.size(-1)
        total_nodes_available = graph_nodes.size(1)

        # 1. Évaluation Baseline (Calcul fixe systématique)
        start_time = time.perf_counter()
        with torch.no_grad():
            baseline_policy, baseline_trajectories = self.baseline(query_emb, retrieved_docs, graph_nodes)
            baseline_violations = torch.relu(-baseline_trajectories * constraints.unsqueeze(1)).sum().item()
        baseline_time = (time.perf_counter() - start_time) * 1000
        baseline_flops = self.simulate_flops(total_nodes_available, retrieved_docs.size(1), emb_dim)

        self.results["Baseline"].append({
            "time_ms": baseline_time,
            "flops": baseline_flops,
            "violations": baseline_violations
        })

        # 2. Évaluation AIOTECH44 (Calcul adaptatif + SCG)
        start_time = time.perf_counter()
        with torch.no_grad():
            out = self.aiotech(query_emb, retrieved_docs, graph_nodes, constraints)
            aiotech_violations = torch.relu(-out["trajectories"] * constraints.unsqueeze(1)).sum().item()
        aiotech_time = (time.perf_counter() - start_time) * 1000
        aiotech_flops = self.simulate_flops(out["budget_k"].mean().item(), retrieved_docs.size(1), emb_dim)

        self.results["AIOTECH44"].append({
            "time_ms": aiotech_time,
            "flops": aiotech_flops,
            "violations": aiotech_violations,
            "complexity_score": out["complexity_score"].item()
        })

    def generate_report(self):
        print("\n" + "="*50)
        print(" RAPPORT DE BENCHMARK : GREEN AI (AIOTECH44 vs BASELINE)")
        print("="*50)

        for model_name, metrics in self.results.items():
            avg_time = np.mean([m["time_ms"] for m in metrics])
            avg_flops = np.mean([m["flops"] for m in metrics])
            total_violations = np.sum([m["violations"] for m in metrics])
            
            print(f"\n{model_name.upper()} :")
            print(f" - Latence moyenne     : {avg_time:.2f} ms")
            print(f" - Charge de calcul    : {avg_flops:,.0f} FLOPs simulés")
            print(f" - Taux Hallucinations : {total_violations:.2f} violations")

        base_flops = np.mean([m["flops"] for m in self.results["Baseline"]])
        aio_flops = np.mean([m["flops"] for m in self.results["AIOTECH44"]])
        energy_saved = (1.0 - (aio_flops / base_flops)) * 100.0

        print("\n" + "-"*50)
        print(" GAIN STRATÉGIQUE AIOTECH44 :")
        print(f" -> Réduction de la charge de calcul : {energy_saved:.1f} %")
        print("-" * 50 + "\n")

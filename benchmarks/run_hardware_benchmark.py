import time
import torch
import torch.nn as nn
from core.aiotech44_core import AIOTECH44_EnergyCore


class StaticBaseline(nn.Module):
    """Modèle de référence à calcul dense sans élagage ni allocation adaptative."""
    def __init__(self, emb_dim: int, num_nodes: int):
        super().__init__()
        self.policy = nn.Linear(emb_dim * 2, num_nodes)

    def forward(self, query_emb, docs, graph_nodes):
        fused = query_emb + docs.mean(dim=1)
        return self.policy(torch.cat([graph_nodes.mean(dim=1), fused], dim=-1))


def run_benchmark(num_iterations: int = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exécution du benchmark matériel sur : {device}")

    emb_dim = 256
    num_nodes = 100
    seq_len = 30
    batch_size = 4

    baseline = StaticBaseline(emb_dim, num_nodes).to(device)
    aiotech = AIOTECH44_EnergyCore(emb_dim=emb_dim, num_nodes=num_nodes).to(device)
    baseline.eval()
    aiotech.eval()

    # Données fixes pour le test
    query = torch.randn(batch_size, emb_dim, device=device)
    docs = torch.randn(batch_size, seq_len, emb_dim, device=device)
    graph = torch.randn(batch_size, num_nodes, emb_dim, device=device)
    constraints = torch.randn(batch_size, emb_dim, device=device)

    # Mesure Baseline
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = baseline(query, docs, graph)
    t_baseline = (time.perf_counter() - t0) * 1000 / num_iterations

    # Mesure AIOTECH44
    t1 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            out_aio = aiotech(query, docs, graph, constraints)
    t_aiotech = (time.perf_counter() - t1) * 1000 / num_iterations

    tokens_initial = seq_len * batch_size
    tokens_conserves = out_aio["allocated_tokens"]
    reduction_pct = (1.0 - (tokens_conserves / tokens_initial)) * 100

    print("\n" + "=" * 55)
    print(" RÉSULTATS DU BENCHMARK MATÉRIEL")
    print("=" * 55)
    print(f"Latence moyenne Baseline  : {t_baseline:.2f} ms")
    print(f"Latence moyenne AIOTECH44 : {t_aiotech:.2f} ms")
    print(f"Contexte initial          : {tokens_initial} tokens")
    print(f"Contexte alloué effectif  : {tokens_conserves} tokens")
    print(f"Volume contextuel réduit  : -{reduction_pct:.1f} %")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    run_benchmark()

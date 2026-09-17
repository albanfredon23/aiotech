import torch
import torch.nn as nn
from core.aiotech44_core import AIOTECH44_EnergyCore
from scg.metrics import EnergyBenchmark

class StandardLLMBaseline(nn.Module):
    """Simule un LLM/RAG classique à calcul fixe sans élagage géométrique."""
    def __init__(self, emb_dim: int, num_nodes: int):
        super().__init__()
        self.policy = nn.Linear(emb_dim * 2, num_nodes)

    def forward(self, query_emb, retrieved_docs, graph_nodes):
        fused = query_emb + retrieved_docs.mean(dim=1)
        trajectories = graph_nodes[:, :4, :]
        policy = self.policy(torch.cat([graph_nodes.mean(dim=1), fused], dim=-1))
        return policy, trajectories

def main():
    emb_dim = 256
    num_nodes = 100
    num_queries = 20

    baseline = StandardLLMBaseline(emb_dim, num_nodes)
    aiotech = AIOTECH44_EnergyCore(emb_dim=emb_dim, num_nodes=num_nodes)
    benchmark = EnergyBenchmark(baseline, aiotech)

    print(f"Démarrage du benchmark comparatif sur {num_queries} requêtes...")
    for _ in range(num_queries):
        query = torch.randn(1, emb_dim)
        docs = torch.randn(1, 10, emb_dim)
        graph = torch.randn(1, num_nodes, emb_dim)
        constraints = torch.randn(1, emb_dim)
        benchmark.evaluate_query(query, docs, graph, constraints)

    benchmark.generate_report()

if __name__ == "__main__":
    main()

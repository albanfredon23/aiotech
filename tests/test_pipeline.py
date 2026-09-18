import torch
from core.aiotech44_core import AIOTECH44_EnergyCore


def test_full_pipeline_backward():
    """Valide les dimensions, le flux de tenseurs et la rétropropagation de bout en bout."""
    batch_size, emb_dim, num_nodes = 2, 64, 20
    core = AIOTECH44_EnergyCore(emb_dim=emb_dim, num_nodes=num_nodes, num_agents=4, num_rules=8)
    core.train()

    query = torch.randn(batch_size, emb_dim, requires_grad=True)
    docs = torch.randn(batch_size, 15, emb_dim, requires_grad=True)
    graph = torch.randn(batch_size, num_nodes, emb_dim, requires_grad=True)
    constraints = torch.randn(batch_size, emb_dim)

    outputs = core(query, docs, graph, constraints)

    # 1. Vérification des dimensions et des clés requises
    assert outputs["policy"].shape == (batch_size, num_nodes), "Dimension de policy incorrecte"
    assert "allocated_tokens" in outputs, "Clé allocated_tokens manquante"
    assert "allocated_memory_ratio" in outputs, "Clé allocated_memory_ratio manquante"
    assert outputs["allocated_tokens"] <= 15 * batch_size, "Budget alloué supérieur au contexte fourni"

    # 2. Vérification de l'autograd de bout en bout
    loss = outputs["policy"].sum() + outputs["scg_loss"]
    loss.backward()

    assert query.grad is not None, "Rupture de gradient sur query_emb"
    assert docs.grad is not None, "Rupture de gradient sur docs"
    assert core.policy_head.weight.grad is not None, "Rupture de gradient sur policy_head"


if __name__ == "__main__":
    test_full_pipeline_backward()
    print("✓ tests/test_pipeline.py exécuté et validé sans erreur.")

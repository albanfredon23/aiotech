import torch
from core.aiotech44_core import AIOTECH44_EnergyCore


def test_pipeline_integration_and_autograd():
    batch_size, emb_dim, num_nodes, num_docs = 2, 64, 20, 15
    core = AIOTECH44_EnergyCore(emb_dim=emb_dim, num_nodes=num_nodes, num_agents=4, num_rules=8)
    core.train()

    query = torch.randn(batch_size, emb_dim, requires_grad=True)
    docs = torch.randn(batch_size, num_docs, emb_dim, requires_grad=True)
    graph = torch.randn(batch_size, num_nodes, emb_dim, requires_grad=True)
    constraints = torch.randn(batch_size, emb_dim)

    outputs = core(query, docs, graph, constraints)

    # 1. Dimensions et clés
    assert outputs["policy"].shape == (batch_size, num_nodes)
    assert "allocated_tokens" in outputs
    assert "allocated_tokens_total" in outputs
    assert "allocated_memory_ratio" in outputs
    assert "padding_mask" in outputs

    # 2. Validation scalaire et bornes
    assert outputs["allocated_tokens"].item() <= num_docs
    assert outputs["allocated_tokens"].item() >= 2
    assert 0.0 <= outputs["allocated_memory_ratio"].item() <= 100.0

    # 3. Autograd complet et gradients réels
    loss = outputs["policy"].sum() + outputs["scg_loss"]
    loss.backward()

    for name, tensor in [("query", query), ("docs", docs)]:
        assert tensor.grad is not None, f"Gradient absent pour {name}"
        assert torch.isfinite(tensor.grad).all(), f"Gradient non fini (NaN/Inf) pour {name}"
        assert tensor.grad.abs().sum() > 0, f"Gradient nul pour {name}"

    assert core.policy_head.weight.grad is not None
    assert torch.isfinite(core.policy_head.weight.grad).all()
    assert core.policy_head.weight.grad.abs().sum() > 0


if __name__ == "__main__":
    test_pipeline_integration_and_autograd()
    print("test_pipeline.py validé avec succès.")

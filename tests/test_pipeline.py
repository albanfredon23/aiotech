import torch
import pytest
from core.aiotech44_core import AIOTECH44_EnergyCore

def test_full_pipeline_backward():
    """Valide les dimensions, le flux de tenseurs et la rétropropagation de bout en bout."""
    batch_size, emb_dim, num_nodes = 2, 64, 10
    core = AIOTECH44_EnergyCore(emb_dim=emb_dim, num_nodes=num_nodes, num_agents=4, num_rules=8)
    core.train()

    query = torch.randn(batch_size, emb_dim, requires_grad=True)
    docs = torch.randn(batch_size, 15, emb_dim, requires_grad=True)
    graph = torch.randn(batch_size, num_nodes, emb_dim, requires_grad=True)
    constraints = torch.randn(batch_size, emb_dim)

    outputs = core(query, docs, graph, constraints)
    
    # 1. Vérification des dimensions
    assert outputs["policy"].shape == (batch_size, num_nodes)
    assert outputs["allocated_tokens"] <= 15

    # 2. Vérification de l'autograd de bout en bout (Point 10)
    loss = outputs["policy"].sum()
    loss.backward()

    assert query.grad is not None
    assert docs.grad is not None
    assert core.policy_head.weight.grad is not None

import pytest

torch = pytest.importorskip("torch")

from aiotech.research.arg_core import ARGCore, EuclideanAdmissibility, soft_godel  # noqa: E402
from aiotech.research.compiler import InvariantRegistry, MutationCompiler  # noqa: E402

B, D, NODES, DOCS = 2, 64, 20, 15


def inputs(requires_grad=False):
    torch.manual_seed(0)
    q = torch.randn(B, D, requires_grad=requires_grad)
    docs = torch.randn(B, DOCS, D, requires_grad=requires_grad)
    nodes = torch.randn(B, NODES, D, requires_grad=requires_grad)
    cons = torch.randn(B, D)
    return q, docs, nodes, cons


def test_smoke_inference_shapes_and_fallback():
    core = ARGCore(emb_dim=D, num_nodes=NODES, num_rules=8).eval()
    with torch.no_grad():
        out = core(*inputs())
    assert out["policy"].shape == (B, NODES)
    assert out["trajectories"].shape == (B, 4, 4, D)       # K=4, H+1=4
    assert out["admissibility"].shape == (B, 4)
    assert (out["mask"].sum(-1) >= 1).all()                  # jamais zéro trajectoire (repli)
    assert 0 < out["docs_kept_ratio"] <= 1


def test_gradients_flow_end_to_end():
    core = ARGCore(emb_dim=D, num_nodes=NODES, num_rules=8).train()
    q, docs, nodes, cons = inputs(requires_grad=True)
    out = core(q, docs, nodes, cons)
    (out["policy"].sum() + out["aux_loss"]).backward()
    for name, t in [("query", q), ("docs", docs), ("nodes", nodes)]:
        assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0, name
    assert core.budget.gate[0].weight.grad is not None      # le budget k reçoit un gradient
    assert core.admissibility.radius_raw.grad is not None


def test_admissibility_uses_scale_not_sphere():
    adm = EuclideanAdmissibility(D, tau=0.5).eval()
    anchor = torch.zeros(1, D)
    near = torch.full((1, 1, 2, D), 0.01)
    far = near * 1000                                        # même direction, autre échelle
    a_near = adm(near, anchor)["admissibility"].item()
    a_far = adm(far, anchor)["admissibility"].item()
    assert a_near > a_far  # une projection sphérique rendrait ces deux cas identiques


def test_soft_godel_matches_min():
    x = torch.tensor([[0.9, 0.2, 0.7]])
    assert soft_godel(x, 0.001).item() == pytest.approx(0.2, abs=0.01)


def test_fuse_linear_is_exact_and_skips_activations():
    torch.manual_seed(1)
    seq = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    fused = MutationCompiler.fuse_linear(seq)
    assert len(fused) == 3                                   # seules les 2 premières fusionnées
    x = torch.randn(5, 8)
    ok, _ = InvariantRegistry.numerical(fused, seq, x, atol=1e-5)
    assert ok
    with_act = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))
    assert MutationCompiler.fuse_linear(with_act) is None    # v3 les aurait fusionnées à tort

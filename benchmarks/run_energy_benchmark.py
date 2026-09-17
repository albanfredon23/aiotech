import os
import sys
import gc
import torch
import torch.nn as nn
import numpy as np

# Permet d'exécuter le script directement depuis la racine ou le dossier benchmarks
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.aiotech44_core import AIOTECH44_EnergyCore


class StandardBaseline(nn.Module):
    """
    Baseline standard sans DifferentialMemoryAllocator ni élagage géométrique.
    Conserve 100% des documents et résout la politique avec calcul quadratique fixe.
    """
    def __init__(self, emb_dim: int, num_nodes: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_nodes = num_nodes
        # Projection dense sur la totalité des documents
        self.doc_projection = nn.Linear(emb_dim, emb_dim)
        # La tête de politique concatène graph_context + trajectory_context + docs_context
        self.policy_head = nn.Linear(emb_dim * 3, num_nodes)

    def forward(
        self, 
        query_emb: torch.Tensor, 
        retrieved_docs_emb: torch.Tensor, 
        graph_nodes: torch.Tensor, 
        constraints: torch.Tensor
    ):
        # 1. Traitement systématique de l'intégralité du contexte (sans budget_k)
        proj_docs = self.doc_projection(retrieved_docs_emb)
        docs_context = proj_docs.mean(dim=1)  # (batch_size, emb_dim)

        # 2. Pas d'élagage géométrique : conserve les 4 premières trajectoires par défaut
        trajectories = graph_nodes[:, :4, :]
        graph_context = graph_nodes.mean(dim=1)
        trajectory_context = trajectories.mean(dim=1)

        # 3. Calcul de politique sur dimensions fixes
        combined = torch.cat([graph_context, trajectory_context, docs_context], dim=-1)
        policy = self.policy_head(combined)

        return {
            "policy": policy,
            "trajectories": trajectories,
            "allocated_tokens": retrieved_docs_emb.size(1)  # 100% des tokens consommés
        }


def measure_inference(model: nn.Module, inputs: tuple, device: torch.device):
    """
    Mesure avec précision la latence CUDA, le pic VRAM et les FLOPs réels d'une inférence.
    """
    query, docs, graph, constraints = inputs
    is_cuda = device.type == "cuda"

    # Nettoyage mémoire pré-inférence
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # 1. Mesure de la Latence
    if is_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            outputs = model(query, docs, graph, constraints)
        end_event.record()
        torch.cuda.synchronize(device)
        latency_ms = start_event.elapsed_time(end_event)
    else:
        import time
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(query, docs, graph, constraints)
        latency_ms = (time.perf_counter() - start) * 1000.0

    # 2. Mesure du Pic VRAM alloué (en KiB)
    if is_cuda:
        peak_vram_kb = torch.cuda.max_memory_allocated(device) / 1024.0
    else:
        peak_vram_kb = 0.0

    # 3. Mesure des FLOPs réels via le Torch Profiler
    activities = [torch.profiler.ProfilerActivity.CPU]
    if is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        with_flops=True
    ) as prof:
        with torch.no_grad():
            _ = model(query, docs, graph, constraints)

    # Somme des FLOPs instrumentés par les kernels aten
    real_flops = sum(getattr(evt, "flops", 0) for evt in prof.key_averages())

    # Fallback d'approximation si les kernels natifs ne renvoient pas les compteurs
    if real_flops == 0:
        total_params = sum(p.numel() for p in model.parameters())
        tokens = outputs["allocated_tokens"]
        real_flops = 2 * total_params + (tokens ** 2) * query.size(-1)

    return {
        "latency_ms": latency_ms,
        "peak_vram_kb": peak_vram_kb,
        "flops": float(real_flops),
        "allocated_tokens": outputs["allocated_tokens"]
    }


def run_benchmark(num_queries: int = 50, emb_dim: int = 256, num_nodes: int = 50, max_docs: int = 30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Exécution sur l'appareil : {device}")
    if device.type == "cuda":
        print(f"[*] GPU détecté : {torch.cuda.get_device_name(0)}")

    # Instanciation
    baseline_model = StandardBaseline(emb_dim, num_nodes).to(device).eval()
    aiotech_model = AIOTECH44_EnergyCore(emb_dim=emb_dim, num_nodes=num_nodes).to(device).eval()

    # Warm-up (essentiel pour charger les DLL cuBLAS et stabiliser les mesures)
    print("[*] Préchauffage du hardware (warmup)...")
    dummy_query = torch.randn(1, emb_dim, device=device)
    dummy_docs = torch.randn(1, max_docs, emb_dim, device=device)
    dummy_graph = torch.randn(1, num_nodes, emb_dim, device=device)
    dummy_constraints = torch.randn(1, emb_dim, device=device)
    for _ in range(5):
        _ = baseline_model(dummy_query, dummy_docs, dummy_graph, dummy_constraints)
        _ = aiotech_model(dummy_query, dummy_docs, dummy_graph, dummy_constraints)

    baseline_metrics = []
    aiotech_metrics = []

    print(f"[*] Lancement de l'évaluation comparative sur {num_queries} requêtes...")
    for q_idx in range(num_queries):
        # Requêtes synthétiques variant en complexité et taille contextuelle
        q = torch.randn(1, emb_dim, device=device)
        docs = torch.randn(1, max_docs, emb_dim, device=device)
        graph = torch.randn(1, num_nodes, emb_dim, device=device)
        constraints = torch.randn(1, emb_dim, device=device)
        inputs = (q, docs, graph, constraints)

        # Baseline
        m_base = measure_inference(baseline_model, inputs, device)
        baseline_metrics.append(m_base)

        # AIOTECH44
        m_aio = measure_inference(aiotech_model, inputs, device)
        aiotech_metrics.append(m_aio)

    # Synthèse statistique
    avg_base_lat = np.mean([m["latency_ms"] for m in baseline_metrics])
    avg_aio_lat = np.mean([m["latency_ms"] for m in aiotech_metrics])

    avg_base_vram = np.mean([m["peak_vram_kb"] for m in baseline_metrics])
    avg_aio_vram = np.mean([m["peak_vram_kb"] for m in aiotech_metrics])

    avg_base_flops = np.mean([m["flops"] for m in baseline_metrics])
    avg_aio_flops = np.mean([m["flops"] for m in aiotech_metrics])

    avg_base_tokens = np.mean([m["allocated_tokens"] for m in baseline_metrics])
    avg_aio_tokens = np.mean([m["allocated_tokens"] for m in aiotech_metrics])

    print("\n" + "=" * 65)
    print("        RÉSULTATS DU BENCHMARK MATÉRIEL (GREEN AI)")
    print("=" * 65)
    print(f"{'Métrique':<25} | {'Baseline':<16} | {'AIOTECH44':<16} | {'Gain / Économie'}")
    print("-" * 65)
    
    token_gain = ((avg_base_tokens - avg_aio_tokens) / avg_base_tokens) * 100.0
    print(f"{'Tokens/Docs alloués':<25} | {avg_base_tokens:<16.1f} | {avg_aio_tokens:<16.1f} | {token_gain:+.1f}%")

    if device.type == "cuda":
        vram_gain = ((avg_base_vram - avg_aio_vram) / avg_base_vram) * 100.0
        print(f"{'Pic VRAM alloué':<25} | {avg_base_vram:<13.1f} KiB | {avg_aio_vram:<13.1f} KiB | {vram_gain:+.1f}%")

    lat_diff = ((avg_base_lat - avg_aio_lat) / avg_base_lat) * 100.0
    print(f"{'Latence moyenne':<25} | {avg_base_lat:<14.2f} ms | {avg_aio_lat:<14.2f} ms | {lat_diff:+.1f}%")

    flops_gain = ((avg_base_flops - avg_aio_flops) / avg_base_flops) * 100.0
    print(f"{'Charge calcul (FLOPs)':<25} | {avg_base_flops:<16,.0f} | {avg_aio_flops:<16,.0f} | {flops_gain:+.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark(num_queries=30, emb_dim=256, num_nodes=50, max_docs=20)

from typing import Dict, Any
import torch
import torch.nn as nn

from .adaptive_gate import AdaptiveComputeGate
from .dynamic_allocator import DynamicAgentAllocator
from .symbolic_engine import NeuroSymbolicEngine
from .beam_planner import DifferentiableBeamSearch
from .memory_allocator import DifferentialMemoryAllocator
from scg.pruner import SCGEnergyPruner


class AIOTECH44_EnergyCore(nn.Module):
    """
    Orchestrateur central unifiant calcul adaptatif, agents spécialisés,
    logique formelle différentiable, élagage SCG et mémoire différentielle.
    """
    def __init__(
        self,
        emb_dim: int = 256,
        num_nodes: int = 50,
        num_agents: int = 4,
        num_rules: int = 16
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_nodes = num_nodes
        self.num_agents = num_agents
        self.num_rules = num_rules

        # Modules constitutifs
        self.compute_gate = AdaptiveComputeGate(emb_dim=emb_dim)
        self.agent_allocator = DynamicAgentAllocator(emb_dim=emb_dim, num_agents=num_agents)
        self.neuro_symbolic_engine = NeuroSymbolicEngine(emb_dim=emb_dim, num_rules=num_rules)
        self.beam_planner = DifferentiableBeamSearch(emb_dim=emb_dim)
        self.scg_pruner = SCGEnergyPruner()
        self.memory_allocator = DifferentialMemoryAllocator(base_k=2, max_k=min(20, num_nodes))

        # Tête de décision finale (Graphe + Trajectoires + Résumé Mémoire)
        self.policy_head = nn.Linear(emb_dim * 3, num_nodes)

    def forward(
        self,
        query_emb: torch.Tensor,
        retrieved_docs_emb: torch.Tensor,
        graph_nodes: torch.Tensor,
        constraints: torch.Tensor
    ) -> Dict[str, Any]:
        total_docs = retrieved_docs_emb.size(1)

        # 1. Évaluation et porte adaptative
        gated_nodes, complexity_score = self.compute_gate(query_emb, graph_nodes)

        # 2. Allocation dynamique des agents
        weighted_context, agent_weights = self.agent_allocator(query_emb)

        # 3. Logique formelle neuro-symbolique
        logic_activation = self.neuro_symbolic_engine(weighted_context)
        logic_context = torch.matmul(
            logic_activation, self.neuro_symbolic_engine.rule_embeddings
        )
        fused_query = query_emb + logic_context

        # 4. Planificateur de trajectoires
        raw_trajectories = self.beam_planner(fused_query, gated_nodes)

        # 5. Élagage SCG (4 sorties synchronisées)
        surviving_trajectories, active_mask, scg_scores, scg_loss = self.scg_pruner(
            raw_trajectories, constraints
        )

        # 6. Allocation mémoire différentielle (3 sorties déballées)
        allocated_memory, budget_k, padding_mask = self.memory_allocator(
            scg_scores, retrieved_docs_emb
        )

        # Calcul propre des métriques scalaires
        allocated_tokens = budget_k.detach().mean()
        allocated_memory_ratio = (allocated_tokens / max(1, total_docs)) * 100.0
        allocated_tokens_total = padding_mask.sum().detach()

        # 7. Synthèse décisionnelle causale
        graph_context = gated_nodes.mean(dim=1)
        trajectory_context = surviving_trajectories.mean(dim=1)
        memory_summary = allocated_memory.mean(dim=1)

        fused_features = torch.cat(
            [graph_context, trajectory_context, memory_summary], dim=-1
        )
        policy_logits = self.policy_head(fused_features)

        return {
            "policy": policy_logits,
            "trajectories": surviving_trajectories,
            "complexity_score": complexity_score,
            "active_agents": agent_weights,
            "budget_k": budget_k,
            "allocated_tokens": allocated_tokens,
            "allocated_tokens_total": allocated_tokens_total,
            "allocated_memory_ratio": allocated_memory_ratio,
            "padding_mask": padding_mask,
            "scg_loss": scg_loss
        }

    def get_summary(self) -> Dict[str, Any]:
        return {
            "emb_dim": self.emb_dim,
            "num_nodes": self.num_nodes,
            "num_agents": self.num_agents,
            "num_rules": self.num_rules,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

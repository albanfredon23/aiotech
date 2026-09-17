import logging
import torch
import torch.nn as nn
from typing import Dict, Any

from .adaptive_gate import AdaptiveComputeGate
from .dynamic_allocator import DynamicAgentAllocator
from .symbolic_engine import NeuroSymbolicEngine
from .beam_planner import DifferentiableBeamSearch
from .memory_allocator import DifferentialMemoryAllocator
from scg.pruner import SCGEnergyPruner

logger = logging.getLogger(__name__)

class AIOTECH44_EnergyCore(nn.Module):
    """
    Orchestrateur central connectant le budget mémoire et l'élagage
    au pipeline réel de décision.
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
        
        self.compute_gate = AdaptiveComputeGate(emb_dim)
        self.agent_allocator = DynamicAgentAllocator(emb_dim, num_agents)
        self.neuro_symbolic_engine = NeuroSymbolicEngine(emb_dim, num_rules)
        self.beam_planner = DifferentiableBeamSearch(emb_dim, beam_width=4)
        self.scg_pruner = SCGEnergyPruner(lam=0.2, energy_threshold=0.5)
        self.memory_allocator = DifferentialMemoryAllocator(base_k=2, max_k=min(20, num_nodes))
        
        # La tête de politique combine :
        # - Le contexte du graphe (emb_dim)
        # - Le contexte des trajectoires survivantes (emb_dim)
        # - Le contexte mémoire réellement alloué par budget_k (emb_dim)
        self.policy_head = nn.Linear(emb_dim * 3, num_nodes)
        nn.init.xavier_uniform_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)

    def forward(
        self, 
        query_emb: torch.Tensor, 
        retrieved_docs_emb: torch.Tensor,
        graph_nodes: torch.Tensor, 
        constraints: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Forward pass avec réduction réelle du calcul et propagation de gradient.
        """
        # 1. Évaluation et réduction de la mémoire contextuelle dès l'entrée
        # Estimation initiale de cohérence requête/contraintes pour dimensionner le RAG
        initial_violation = torch.relu(-query_emb * constraints).sum(dim=-1, keepdim=True)
        
        # Contexte tronqué physiquement à k_max_batch (réduction VRAM et FLOPs)
        sliced_memory, budget_k, mem_padding_mask = self.memory_allocator(
            initial_violation, retrieved_docs_emb
        )
        
        # Pooling pondéré par le masque de validité réel
        valid_counts = mem_padding_mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)
        docs_context = sliced_memory.sum(dim=1, keepdim=True) / valid_counts  # (batch_size, 1, emb_dim)
        fused_query = query_emb + docs_context.squeeze(1)

        # 2. Gate adaptatif sur les nœuds
        gated_nodes, complexity = self.compute_gate(fused_query, graph_nodes)

        # 3. Allocation conditionnelle des agents
        weighted_context, agent_weights = self.agent_allocator(fused_query)

        # 4. Inférence neuro-symbolique (Gödel)
        logic_activation = self.neuro_symbolic_engine(weighted_context)
        logic_context = torch.matmul(logic_activation, self.neuro_symbolic_engine.rule_embeddings)
        fused_query = fused_query + logic_context

        # 5. Recherche de trajectoires et élagage SCG
        raw_trajectories = self.beam_planner(fused_query, gated_nodes)
        surviving_trajectories, active_mask = self.scg_pruner.prune_trajectories(
            raw_trajectories, constraints
        )

        # 6. Synthèse finale : TOUS les composants conditionnels pilotent la décision
        graph_context = gated_nodes.mean(dim=1)
        trajectory_context = surviving_trajectories.mean(dim=1)
        memory_summary = docs_context.squeeze(1)
        
        combined_repr = torch.cat([graph_context, trajectory_context, memory_summary], dim=-1)
        policy = self.policy_head(combined_repr)

        return {
            "policy": policy,
            "trajectories": surviving_trajectories,
            "complexity_score": complexity,
            "active_agents": agent_weights,
            "budget_k": budget_k,
            "allocated_tokens": sliced_memory.size(1),  # Preuve physique de réduction
            "allocated_memory_ratio": self.memory_allocator.get_budget_efficiency(
                budget_k, retrieved_docs_emb.size(1)
            )
        }

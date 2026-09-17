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
    Orchestrateur central unifiant calcul adaptatif, agents, logique formelle,
    planificateur différentiable, élagage SCG et mémoire différentielle.
    """
    def __init__(self, emb_dim: int = 256, num_nodes: int = 50, 
                 num_agents: int = 4, num_rules: int = 16):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_nodes = num_nodes
        self.num_agents = num_agents
        self.num_rules = num_rules
        
        # Modules fonctionnels AIOTECH44
        self.compute_gate = AdaptiveComputeGate(emb_dim)
        self.agent_allocator = DynamicAgentAllocator(emb_dim, num_agents)
        self.neuro_symbolic_engine = NeuroSymbolicEngine(emb_dim, num_rules)
        self.beam_planner = DifferentiableBeamSearch(emb_dim, beam_width=4)
        self.scg_pruner = SCGEnergyPruner(lam=0.2, energy_threshold=0.5)
        self.memory_allocator = DifferentialMemoryAllocator(base_k=2, max_k=min(20, num_nodes))
        
        # Couche de décision finale
        self.policy_head = nn.Linear(emb_dim * 2, num_nodes)
        nn.init.xavier_uniform_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)

    def forward(self, query_emb: torch.Tensor, retrieved_docs_emb: torch.Tensor,
                graph_nodes: torch.Tensor, constraints: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            query_emb:          (batch_size, emb_dim)
            retrieved_docs_emb: (batch_size, seq_len, emb_dim)
            graph_nodes:        (batch_size, num_nodes, emb_dim)
            constraints:        (batch_size, emb_dim)
        """
        try:
            # 1. Fusion RAG sécurisée
            docs_mean = retrieved_docs_emb.mean(dim=1)
            fused_query = query_emb + docs_mean
            
            # 2. Évaluation de la complexité & Gate adaptatif
            gated_nodes, complexity = self.compute_gate(fused_query, graph_nodes)
            
            # 3. Allocation différentielle des agents (20%)
            weighted_context, agent_weights = self.agent_allocator(fused_query)
            
            # 4. Moteur neuro-symbolique (Gödel)
            logic_activation = self.neuro_symbolic_engine(weighted_context)
            logic_context = self.neuro_symbolic_engine.aggregate_rules(logic_activation)
            fused_query = fused_query + logic_context
            
            # 5. Recherche différentiable de trajectoires
            raw_trajectories = self.beam_planner(fused_query, gated_nodes)
            
            # 6. Élagage géométrique préventif SCG
            surviving_trajectories, active_mask = self.scg_pruner.prune_trajectories(
                raw_trajectories, constraints
            )
            
            # 7. Allocation différentielle de mémoire VRAM (Context Budgeting)
            allocated_memory, budget_k = self.memory_allocator(active_mask, retrieved_docs_emb)
            memory_ratio = self.memory_allocator.get_budget_efficiency(
                budget_k, retrieved_docs_emb.size(1)
            )
            
            # 8. Tête de politique finale (Policy)
            graph_context = gated_nodes.mean(dim=1)
            trajectory_context = surviving_trajectories.mean(dim=1)
            policy = self.policy_head(torch.cat([graph_context, trajectory_context], dim=-1))
            
            return {
                "policy": policy,
                "trajectories": surviving_trajectories,
                "complexity_score": complexity,
                "active_agents": agent_weights,
                "budget_k": budget_k,
                "allocated_memory_ratio": memory_ratio
            }
            
        except Exception as e:
            logger.error(f"Erreur d'inférence dans AIOTECH44_EnergyCore: {str(e)}")
            raise

    def get_summary(self) -> Dict[str, Any]:
        """Retourne les spécifications et paramètres du moteur."""
        return {
            "emb_dim": self.emb_dim,
            "num_nodes": self.num_nodes,
            "num_agents": self.num_agents,
            "num_rules": self.num_rules,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

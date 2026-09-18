from typing import Dict, Any, Tuple
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
    logique formelle différentiable, élagage SCG et allocation différentielle de mémoire.
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

        # 1. Modulation adaptative de l'effort de calcul
        self.compute_gate = AdaptiveComputeGate(emb_dim=emb_dim)

        # 2. Allocation dynamique des agents d'expertise
        self.agent_allocator = DynamicAgentAllocator(emb_dim=emb_dim, num_agents=num_agents)

        # 3. Moteur logique neuro-symbolique (Soft-Gödel)
        self.neuro_symbolic_engine = NeuroSymbolicEngine(emb_dim=emb_dim, num_rules=num_rules)

        # 4. Planificateur de trajectoires (Gumbel-Softmax STE / Top-k)
        self.beam_planner = DifferentiableBeamSearch(emb_dim=emb_dim)

        # 5. Élagage géométrique et régularisation (SCG)
        self.scg_pruner = SCGEnergyPruner()

        # 6. Allocation différentielle de mémoire contextuelle
        self.memory_allocator = DifferentialMemoryAllocator(base_k=2, max_k=min(20, num_nodes))

        # 7. Tête de décision finale (Fusion : Graphe + Trajectoires + Résumé Mémoire)
        self.policy_head = nn.Linear(emb_dim * 3, num_nodes)

    def forward(
        self,
        query_emb: torch.Tensor,
        retrieved_docs_emb: torch.Tensor,
        graph_nodes: torch.Tensor,
        constraints: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Passe avant complète du pipeline adaptatif AIOTECH44.
        """
        total_docs = retrieved_docs_emb.size(1)

        # Étape 1 : Modulation adaptative de l'effort de calcul
        gated_nodes, complexity_score = self.compute_gate(query_emb, graph_nodes)

        # Étape 2 : Routage et pondération des agents spécialisés
        weighted_context, agent_weights = self.agent_allocator(query_emb)

        # Étape 3 : Évaluation logique neuro-symbolique
        logic_activation = self.neuro_symbolic_engine(weighted_context)
        logic_context = torch.matmul(
            logic_activation, self.neuro_symbolic_engine.rule_embeddings
        )

        # Fusion de l'intention avec le contexte symbolique
        fused_query = query_emb + logic_context

        # Étape 4 : Génération des trajectoires candidates
        raw_trajectories = self.beam_planner(fused_query, gated_nodes)

        # Étape 5 : Élagage géométrique SCG
        surviving_trajectories, active_mask, scg_scores, scg_loss = self.scg_pruner(
            raw_trajectories, constraints
        )

        # Étape 6 : Allocation dynamique de la mémoire (3 retours synchronisés)
        allocated_memory, budget_k, padding_mask = self.memory_allocator(
            scg_scores, retrieved_docs_emb
        )

        # Calcul des grandeurs attendues par les tests et main.py
        allocated_tokens = int(budget_k.sum().item())
        allocated_memory_ratio = float((budget_k.mean().item() / max(1, total_docs)) * 100.0)

        # Étape 7 : Synthèse et décision finale (Policy reliée au contexte mémoire)
        graph_context = gated_nodes.mean(dim=1)
        trajectory_context = surviving_trajectories.mean(dim=1)
        memory_summary = allocated_memory.mean(dim=1)

        fused_decision_features = torch.cat(
            [graph_context, trajectory_context, memory_summary], dim=-1
        )
        policy_logits = self.policy_head(fused_decision_features)

        return {
            "policy": policy_logits,
            "trajectories": surviving_trajectories,
            "complexity_score": complexity_score,
            "active_agents": agent_weights,
            "budget_k": budget_k,
            "allocated_tokens": allocated_tokens,
            "allocated_memory_ratio": allocated_memory_ratio,
            "padding_mask": padding_mask,
            "scg_loss": scg_loss
        }

    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de l'empreinte paramétrique du modèle."""
        return {
            "emb_dim": self.emb_dim,
            "num_nodes": self.num_nodes,
            "num_agents": self.num_agents,
            "num_rules": self.num_rules,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }from typing import Dict, Any, Tuple
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

        # 1. Modulation adaptative de l'effort de calcul
        self.compute_gate = AdaptiveComputeGate(emb_dim=emb_dim)

        # 2. Allocation dynamique des agents d'expertise
        self.agent_allocator = DynamicAgentAllocator(emb_dim=emb_dim, num_agents=num_agents)

        # 3. Moteur logique neuro-symbolique (Soft-Gödel)
        self.neuro_symbolic_engine = NeuroSymbolicEngine(emb_dim=emb_dim, num_rules=num_rules)

        # 4. Planificateur de trajectoires (Gumbel-Softmax STE / Top-k)
        self.beam_planner = DifferentiableBeamSearch(emb_dim=emb_dim)

        # 5. Élagage géométrique et régularisation (SCG)
        self.scg_pruner = SCGEnergyPruner()

        # 6. Allocation différentielle de mémoire contextuelle
        self.memory_allocator = DifferentialMemoryAllocator(base_k=2, max_k=min(20, num_nodes))

        # 7. Tête de décision finale (Graphe + Trajectoires + Résumé Mémoire)
        self.policy_head = nn.Linear(emb_dim * 3, num_nodes)

    def forward(
        self,
        query_emb: torch.Tensor,
        retrieved_docs_emb: torch.Tensor,
        graph_nodes: torch.Tensor,
        constraints: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Passe avant complète du pipeline adaptatif AIOTECH44.
        """
        total_docs = retrieved_docs_emb.size(1)

        # Étape 1 : Modulation adaptative de l'effort de calcul
        gated_nodes, complexity_score = self.compute_gate(query_emb, graph_nodes)

        # Étape 2 : Routage et pondération des agents spécialisés
        weighted_context, agent_weights = self.agent_allocator(query_emb)

        # Étape 3 : Évaluation logique neuro-symbolique
        logic_activation = self.neuro_symbolic_engine(weighted_context)
        logic_context = torch.matmul(
            logic_activation, self.neuro_symbolic_engine.rule_embeddings
        )

        # Fusion de l'intention avec le contexte symbolique
        fused_query = query_emb + logic_context

        # Étape 4 : Génération des trajectoires candidates
        raw_trajectories = self.beam_planner(fused_query, gated_nodes)

        # Étape 5 : Élagage géométrique SCG (récupération des scores réels)
        surviving_trajectories, active_mask, scg_scores, scg_loss = self.scg_pruner(
            raw_trajectories, constraints
        )

        # Étape 6 : Allocation dynamique de la mémoire (3 sorties synchronisées)
        allocated_memory, budget_k, padding_mask = self.memory_allocator(
            scg_scores, retrieved_docs_emb
        )

        # Métriques attendues par main.py et tests/test_pipeline.py
        allocated_tokens = int(budget_k.sum().item())
        allocated_memory_ratio = float(budget_k.mean().item() / max(1, total_docs))

        # Étape 7 : Synthèse et décision finale
        graph_context = gated_nodes.mean(dim=1)
        trajectory_context = surviving_trajectories.mean(dim=1)
        memory_summary = allocated_memory.mean(dim=1)

        fused_decision_features = torch.cat(
            [graph_context, trajectory_context, memory_summary], dim=-1
        )
        policy_logits = self.policy_head(fused_decision_features)

        return {
            "policy": policy_logits,
            "trajectories": surviving_trajectories,
            "complexity_score": complexity_score,
            "active_agents": agent_weights,
            "budget_k": budget_k,
            "allocated_tokens": allocated_tokens,
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

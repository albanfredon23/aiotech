import torch
import torch.nn as nn
from typing import List, Tuple

class DynamicAgentAllocator(nn.Module):
    """
    Ajuste en temps réel le budget et le poids de confiance de chaque agent spécialisé
    sans réentraînement complet (régression continue 20%).
    
    ✓ Optimisation Green AI : Court-circuit (early exit) pour les agents inactifs.
    ✓ Architecture robuste : MLP + LayerNorm par agent.
    ✓ Vectorisation PyTorch sans torch.stack superflu.
    """
    def __init__(self, emb_dim: int, num_agents: int = 4, activation_threshold: float = 0.05):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_agents = num_agents
        self.activation_threshold = activation_threshold
        
        # Agents spécialisés (Raisonnement, Causalité, Code, Critique...)
        self.agents = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim)
            )
            for _ in range(num_agents)
        ])
        
        # Logits des poids d'agents (ajustables en ligne)
        self.agent_logits = nn.Parameter(torch.zeros(num_agents))

    def forward(self, fused_query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Alloue les requêtes aux agents avec poids adaptatifs et court-circuit énergétique.
        
        Args:
            fused_query: (batch_size, emb_dim)
            
        Returns:
            weighted_context: (batch_size, emb_dim)
            normalized_weights: (num_agents,)
        """
        normalized_weights = torch.softmax(self.agent_logits, dim=0)
        weighted_context = torch.zeros_like(fused_query)
        
        # Green AI : calcul conditionnel strict pour économiser les FLOPs
        for i, agent in enumerate(self.agents):
            weight = normalized_weights[i]
            if weight > self.activation_threshold:
                # Seuls les agents retenus exécutent leur réseau de neurones
                agent_out = agent(fused_query)
                weighted_context = weighted_context + agent_out * weight

        return weighted_context, normalized_weights

    def get_active_agents(self) -> List[int]:
        """
        Retourne la liste des indices des agents actuellement mobilisés.
        """
        with torch.no_grad():
            weights = torch.softmax(self.agent_logits, dim=0)
            active = (weights > self.activation_threshold).nonzero(as_tuple=True)[0].tolist()
            return active if active else [int(torch.argmax(weights).item())]

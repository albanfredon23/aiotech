import torch
import torch.nn as nn
from typing import Tuple, Dict

class SpecializedAgent(nn.Module):
    """Agent expert possédant une signature structurelle propre."""
    def __init__(self, emb_dim: int, role: str):
        super().__init__()
        self.role = role
        if role == "math":
            self.net = nn.Sequential(nn.Linear(emb_dim, emb_dim * 2), nn.GELU(), nn.Linear(emb_dim * 2, emb_dim))
        elif role == "logic":
            self.net = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.Tanh(), nn.Linear(emb_dim, emb_dim))
        elif role == "critic":
            self.net = nn.Sequential(nn.Linear(emb_dim, emb_dim // 2), nn.ReLU(), nn.Linear(emb_dim // 2, emb_dim))
        else:
            self.net = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.LayerNorm(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DynamicAgentAllocator(nn.Module):
    """
    Allocation dynamique des agents et mise à jour des logits 
    via une règle de régression continue basée sur le signal d'erreur SCG.
    """
    def __init__(self, emb_dim: int, lr_online: float = 0.05):
        super().__init__()
        self.roles = ["reasoning", "math", "logic", "critic"]
        self.agents = nn.ModuleList([SpecializedAgent(emb_dim, r) for r in self.roles])
        self.num_agents = len(self.roles)
        
        # Logits d'experts
        self.agent_logits = nn.Parameter(torch.zeros(self.num_agents))
        self.lr_online = lr_online

    def forward(self, fused_query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.agent_logits, dim=0)
        weighted_context = torch.zeros_like(fused_query)

        # Calcul conditionnel réel : n'exécute que les agents retenus
        for i, agent in enumerate(self.agents):
            if weights[i] > 0.05 or self.training:
                out = agent(fused_query)
                weighted_context = weighted_context + (out * weights[i])

        return weighted_context, weights

    def update_continuous_weights(self, agent_losses: torch.Tensor):
        """
        Régression continue en ligne (Point 15 & 16) :
        Ajuste les logits selon la performance observée sans passe d'optimiseur complet.
        agent_losses: (num_agents,) gradients d'erreur ou violations attribuées.
        """
        with torch.no_grad():
            grad_update = agent_losses - agent_losses.mean()
            self.agent_logits.data -= self.lr_online * grad_update

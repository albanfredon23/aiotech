
import torch
import torch.nn as nn
from typing import Tuple, List

class SpecializedAgent(nn.Module):
    def __init__(self, emb_dim: int, role: str):
        super().__init__()
        self.role = role
        if role == "math":
            self.net = nn.Sequential(
                nn.Linear(emb_dim, emb_dim * 2),
                nn.GELU(),
                nn.Linear(emb_dim * 2, emb_dim),
                nn.LayerNorm(emb_dim)
            )
        elif role == "logic":
            self.net = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.Tanh(),
                nn.Linear(emb_dim, emb_dim),
                nn.LayerNorm(emb_dim)
            )
        elif role == "critic":
            self.net = nn.Sequential(
                nn.Linear(emb_dim, emb_dim // 2),
                nn.ReLU(),
                nn.Linear(emb_dim // 2, emb_dim),
                nn.LayerNorm(emb_dim)
            )
        else:  # reasoning
            self.net = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicAgentAllocator(nn.Module):
    """
    Constructeur explicite restaurant la compatibilité avec AIOTECH44_EnergyCore.
    """
    def __init__(
        self, 
        emb_dim: int, 
        num_agents: int = 4, 
        lr_online: float = 0.05,
        activation_threshold: float = 0.05
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_agents = num_agents
        self.lr_online = lr_online
        self.activation_threshold = activation_threshold

        roles_base = ["reasoning", "math", "logic", "critic"]
        self.roles = roles_base[:num_agents]
        while len(self.roles) < num_agents:
            self.roles.append(f"expert_{len(self.roles)}")

        self.agents = nn.ModuleList([
            SpecializedAgent(emb_dim, role) for role in self.roles
        ])
        
        self.agent_logits = nn.Parameter(torch.zeros(num_agents))

    def forward(self, fused_query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized_weights = torch.softmax(self.agent_logits, dim=0)
        weighted_context = torch.zeros_like(fused_query)

        # Calcul conditionnel strict
        for i, agent in enumerate(self.agents):
            weight = normalized_weights[i]
            if weight > self.activation_threshold or self.training:
                weighted_context = weighted_context + agent(fused_query) * weight

        return weighted_context, normalized_weights

    def update_continuous_weights(self, agent_losses: torch.Tensor):
        """Mise à jour en ligne des logits via le signal d'erreur."""
        with torch.no_grad():
            grad_update = agent_losses - agent_losses.mean()
            self.agent_logits.data -= self.lr_online * grad_update

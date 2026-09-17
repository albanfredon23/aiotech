

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any


class SpecializedAgent(nn.Module):
    """
    Agent expert possédant une architecture et un rôle propres.
    """
    def __init__(self, emb_dim: int, role: str):
        super().__init__()
        self.role = role
        if role == "math":
            # Expansion de dimension pour calcul symbolique dense
            self.net = nn.Sequential(
                nn.Linear(emb_dim, emb_dim * 2),
                nn.GELU(),
                nn.Linear(emb_dim * 2, emb_dim),
                nn.LayerNorm(emb_dim)
            )
        elif role == "logic":
            # Traitement non-linéaire borné pour inférence formelle
            self.net = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.Tanh(),
                nn.Linear(emb_dim, emb_dim),
                nn.LayerNorm(emb_dim)
            )
        elif role == "critic":
            # Bottleneck pour extraction de contradictions
            self.net = nn.Sequential(
                nn.Linear(emb_dim, emb_dim // 2),
                nn.ReLU(),
                nn.Linear(emb_dim // 2, emb_dim),
                nn.LayerNorm(emb_dim)
            )
        else: # "reasoning" / généraliste
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
    Routeur d'agents conditionné à la requête avec sélection clairsemée 
    et apprentissage continu en ligne piloté par l'erreur SCG.
    """
    def __init__(
        self, 
        emb_dim: int, 
        num_agents: int = 4, 
        activation_threshold: float = 0.05,
        lr_online: float = 0.02
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_agents = num_agents
        self.activation_threshold = activation_threshold
        self.lr_online = lr_online
        
        # 1. Spécialisation effective des agents (Point 11)
        self.roles = ["reasoning", "math", "logic", "critic"][:num_agents]
        while len(self.roles) < num_agents:
            self.roles.append(f"expert_{len(self.roles)}")
            
        self.agents = nn.ModuleList([
            SpecializedAgent(emb_dim, role) for role in self.roles
        ])
        
        # 2. Routeur dépendant de l'entrée (Point 12)
        self.router = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, num_agents)
        )
        
        # Prior persistant ajusté par la régression continue en ligne (Point 15)
        self.global_prior = nn.Parameter(torch.zeros(num_agents), requires_grad=False)

    def forward(self, fused_query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcule les poids spécifiques à chaque requête du batch.
        
        Args:
            fused_query: (batch_size, emb_dim)
            
        Returns:
            weighted_context: (batch_size, emb_dim)
            normalized_weights: (batch_size, num_agents)
        """
        batch_size = fused_query.size(0)
        
        # 1. Logits dynamiques dépendant de la requête + prior historique
        instance_logits = self.router(fused_query) # (batch_size, num_agents)
        total_logits = instance_logits + self.global_prior.unsqueeze(0)
        
        normalized_weights = torch.softmax(total_logits, dim=-1) # (batch_size, num_agents)
        weighted_context = torch.zeros_like(fused_query)
        
        # 2. Calcul conditionnel strict (Point 13)
        # On n'exécute le forward de l'agent i que pour les requêtes du batch où son poids > seuil
        for i, agent in enumerate(self.agents):
            agent_weights = normalized_weights[:, i] # (batch_size,)
            active_mask = agent_weights > self.activation_threshold
            
            if active_mask.any():
                # Sous-sélection des requêtes nécessitant cet agent
                sub_inputs = fused_query[active_mask]
                sub_outputs = agent(sub_inputs)
                
                # Injection pondérée
                weighted_context[active_mask] += sub_outputs * agent_weights[active_mask].unsqueeze(-1)

        return weighted_context, normalized_weights

    def update_online(
        self, 
        fused_query: torch.Tensor, 
        agent_weights: torch.Tensor, 
        scg_scores: torch.Tensor, 
        task_success: torch.Tensor
    ):
        """
        Régression continue en ligne (Points 15 & 16) :
        Ajuste les poids de confiance sans passe complète d'optimiseur standard.
        
        Règle : 
        Δw_i = lr * [ (succès - scg_violation) * poids_utilisé_i ]
        
        Args:
            fused_query: (batch_size, emb_dim)
            agent_weights: (batch_size, num_agents)
            scg_scores: (batch_size,) score d'admissibilité dans [0, 1]
            task_success: (batch_size,) booléen ou float [0, 1]
        """
        with torch.no_grad():
            # Signal de renforcement : positif si succès et fort score SCG, négatif sinon
            feedback = (task_success.float() + scg_scores - 1.0) # Centré autour de 0
            
            # Attribution du crédit proportionnelle à l'implication de chaque agent
            # (batch_size, 1) * (batch_size, num_agents) -> (num_agents,)
            delta_prior = (feedback.unsqueeze(-1) * agent_weights).mean(dim=0)
            
            # 1. Mise à jour du prior global (plasticité synaptique lente)
            self.global_prior.add_(self.lr_online * delta_prior)
            # Centrage pour stabiliser le softmax
            self.global_prior.sub_(self.global_prior.mean())
            
            # 2. Ajustement direct de la dernière couche du routeur (adaptation rapide)
            router_last_layer = self.router[-1]
            # Gradient synthétique : erreur propagée vers la dernière couche linéaire
            grad_synthetic = -delta_prior.unsqueeze(0).expand(fused_query.size(0), -1)
            # Mise à jour SGD locale
            router_last_layer.weight.sub_(self.lr_online * torch.matmul(grad_synthetic.t(), fused_query) / fused_query.size(0))
            router_last_layer.bias.sub_(self.lr_online * grad_synthetic.mean(dim=0))

    def get_agent_stats(self) -> Dict[str, Any]:
        """Retourne les priors de confiance stabilisés par agent."""
        priors = torch.softmax(self.global_prior, dim=0).detach().cpu().tolist()
        return {role: round(p, 4) for role, p in zip(self.roles, priors)}


import torch
import torch.nn as nn
from typing import Tuple

class DifferentialMemoryAllocator(nn.Module):
    """
    Alloue dynamiquement la taille du contexte de travail (budget k)
    en fonction du score d'admissibilité SCG pour briser physiquement 
    le coût de calcul en aval.
    """
    def __init__(self, base_k: int = 2, max_k: int = 20):
        super().__init__()
        self.base_k = max(1, int(base_k))
        self.max_k = max(self.base_k + 1, int(max_k))
        
        # Réseau évaluant le ratio contextuel à allouer
        self.budget_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(
        self, 
        trajectory_scg_scores: torch.Tensor, 
        retrieved_context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            trajectory_scg_scores: (batch_size, ...) scores d'admissibilité SCG
            retrieved_context:     (batch_size, seq_len, emb_dim)
            
        Returns:
            sliced_context: (batch_size, max_budget_k, emb_dim) contexte tronqué
            budget_k:       (batch_size,) budget alloué par requête
            padding_mask:   (batch_size, max_budget_k) masque booléen de validité
        """
        batch_size, seq_len, emb_dim = retrieved_context.size()
        
        # 1. Réduction robuste du score SCG -> (batch_size, 1)
        if trajectory_scg_scores.dim() == 1:
            scg_input = trajectory_scg_scores.unsqueeze(-1)
        else:
            scg_input = trajectory_scg_scores.view(batch_size, -1).mean(dim=-1, keepdim=True)
            
        # 2. Ratio continu [0, 1]
        budget_ratio = self.budget_gate(scg_input)  # (batch_size, 1)
        
        # 3. Calcul de dynamic_k avec Straight-Through Estimator (STE) pour conserver le gradient
        effective_max_k = min(self.max_k, seq_len)
        k_spread = max(1, effective_max_k - self.base_k)
        
        dynamic_k_continuous = self.base_k + (budget_ratio.squeeze(-1) * k_spread)
        dynamic_k_discrete = torch.clamp(dynamic_k_continuous.round(), min=self.base_k, max=effective_max_k).long()
        
        # dynamic_k conserve les gradients de budget_ratio grâce au détachement résiduel
        dynamic_k = dynamic_k_discrete.float() + (dynamic_k_continuous - dynamic_k_continuous.detach())
        
        # 4. Tronquage physique : on ne conserve que max(dynamic_k_discrete) tokens
        k_max_batch = int(dynamic_k_discrete.max().item())
        truncated_context = retrieved_context[:, :k_max_batch, :]  # (batch_size, k_max_batch, emb_dim)
        
        # 5. Masque de padding pour les éléments du batch ayant un k < k_max_batch
        indices = torch.arange(k_max_batch, device=retrieved_context.device).unsqueeze(0).expand(batch_size, -1)
        padding_mask = indices < dynamic_k_discrete.unsqueeze(-1)  # (batch_size, k_max_batch)
        
        # Application du masque sur les dimensions non utilisées du batch
        sliced_context = truncated_context * padding_mask.unsqueeze(-1).float()
        
        return sliced_context, dynamic_k, padding_mask

    def get_budget_efficiency(self, dynamic_k: torch.Tensor, max_seq_len: int) -> float:
        """Pourcentage effectif de tokens conservés."""
        if max_seq_len <= 0:
            return 0.0
        return float((dynamic_k.mean().item() / max_seq_len) * 100.0)

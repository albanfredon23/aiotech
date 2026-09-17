import torch
import torch.nn as nn
from typing import Tuple

class DifferentialMemoryAllocator(nn.Module):
    """
    Alloue dynamiquement la taille du contexte de travail (budget k)
    en fonction du score d'admissibilité SCG pour optimiser la mémoire VRAM
    et briser le coût quadratique de l'attention (Green AI).
    """
    def __init__(self, base_k: int = 2, max_k: int = 20):
        super().__init__()
        self.base_k = max(1, int(base_k))
        self.max_k = max(self.base_k + 1, int(max_k))
        
        # Réseau évaluant le besoin contextuel
        self.budget_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, trajectory_scg_scores: torch.Tensor, 
                retrieved_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Alloue le contexte de façon adaptative.
        
        Args:
            trajectory_scg_scores: (batch_size, 1) ou (batch_size, num_trajectories) ou (batch_size,)
            retrieved_context: (batch_size, seq_len, emb_dim)
            
        Returns:
            allocated_memory: (batch_size, seq_len, emb_dim) - contexte masqué
            dynamic_k: (batch_size,) - budget de nœuds alloué
        """
        batch_size, seq_len, _ = retrieved_context.size()
        
        # 1. Normalisation robuste des dimensions du score SCG vers (batch_size, 1)
        if trajectory_scg_scores.dim() == 1:
            scg_input = trajectory_scg_scores.unsqueeze(-1)
        elif trajectory_scg_scores.dim() > 2:
            scg_input = trajectory_scg_scores.view(batch_size, -1).mean(dim=-1, keepdim=True)
        else:
            scg_input = trajectory_scg_scores.mean(dim=-1, keepdim=True)
            
        # 2. Calcul du ratio d'allocation [0, 1]
        budget_ratio = self.budget_gate(scg_input)  # (batch_size, 1)
        
        # 3. Calcul du budget k discret borné entre base_k et min(max_k, seq_len)
        effective_max_k = min(self.max_k, seq_len)
        k_spread = max(1, effective_max_k - self.base_k)
        
        dynamic_k_float = self.base_k + (budget_ratio.squeeze(-1) * k_spread)
        dynamic_k = torch.clamp(dynamic_k_float, min=self.base_k, max=effective_max_k).round().long()
        
        # 4. Construction vectorisée du masque de contexte
        indices = torch.arange(seq_len, device=retrieved_context.device).unsqueeze(0).expand(batch_size, -1)
        memory_mask = (indices < dynamic_k.unsqueeze(-1)).float().unsqueeze(-1)
        
        # 5. Application du masquage
        allocated_memory = retrieved_context * memory_mask
        
        return allocated_memory, dynamic_k.float()

    def get_budget_efficiency(self, dynamic_k: torch.Tensor, max_seq_len: int) -> float:
        """
        Calcule l'efficacité d'utilisation contextuelle (pourcentage de tokens conservés).
        """
        if max_seq_len <= 0:
            return 0.0
        return float((dynamic_k.mean().item() / max_seq_len) * 100.0)

import torch
import torch.nn as nn
from typing import Tuple

class SCGEnergyPruner(nn.Module):
    """
    Régularisateur géométrique SCG.
    - Entraînement : pondération continue différentiable + calcul du terme de loss.
    - Inférence : masque dur booléen (pruning franc).
    """
    def __init__(self, lam: float = 0.2, energy_threshold: float = 0.5, temp: float = 0.1):
        super().__init__()
        self.lam = lam
        self.energy_threshold = energy_threshold
        self.temp = temp

    def forward(
        self, trajectories: torch.Tensor, constraints: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if constraints.dim() == 1:
            c = constraints.unsqueeze(0).unsqueeze(0)
        elif constraints.dim() == 2:
            c = constraints.unsqueeze(1)
        else:
            c = constraints

        # Violation continue
        violations = torch.relu(-trajectories * c).sum(dim=-1) # (B, beam_width)
        scg_scores = self.lam * violations
        scg_loss = scg_scores.mean()

        if self.training:
            # Pénalisation douce différentiable via sigmoïde inverse
            soft_mask = torch.sigmoid((self.energy_threshold - scg_scores) / self.temp)
            surviving_trajectories = trajectories * soft_mask.unsqueeze(-1)
            return surviving_trajectories, soft_mask, scg_loss
        else:
            # Pruning dur pour l'inférence (Green AI)
            active_mask = (scg_scores < self.energy_threshold).float()
            surviving_trajectories = trajectories * active_mask.unsqueeze(-1)
            return surviving_trajectories, active_mask, scg_loss

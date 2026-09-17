import logging
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class SCGEnergyPruner(nn.Module):
    """
    Régularisateur géométrique du Spherical Constraint Graph (SCG).
    - Entraînement : pondération continue différentiable.
    - Inférence : élagage binaire dur.
    """
    def __init__(
        self, 
        lam: float = 0.2, 
        energy_threshold: float = 0.5, 
        temp: float = 0.1,
        max_history_len: int = 1000
    ):
        super().__init__()
        self.lam = lam
        self.energy_threshold = energy_threshold
        self.temp = max(temp, 1e-5)
        self.max_history_len = max_history_len
        self.violation_history = []

    def forward(
        self, 
        trajectories: torch.Tensor, 
        constraints: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Gestion du format des contraintes
        if constraints.dim() == 1:
            c = constraints.unsqueeze(0).unsqueeze(0)
        elif constraints.dim() == 2:
            c = constraints.unsqueeze(1)
        else:
            c = constraints

        violations = torch.relu(-trajectories * c).sum(dim=-1)
        scg_scores = self.lam * violations
        scg_loss = scg_scores.mean()

        if self.training:
            soft_mask = torch.sigmoid((self.energy_threshold - scg_scores) / self.temp)
            surviving_trajectories = trajectories * soft_mask.unsqueeze(-1)
            active_mask = soft_mask
        else:
            hard_mask = (scg_scores < self.energy_threshold).float()
            surviving_trajectories = trajectories * hard_mask.unsqueeze(-1)
            active_mask = hard_mask

        # Suivi borné de l'historique
        total_violations = float(violations.sum().item())
        self.violation_history.append(total_violations)
        if len(self.violation_history) > self.max_history_len:
            self.violation_history.pop(0)

        return surviving_trajectories, active_mask, scg_loss

    def prune_trajectories(
        self, 
        trajectories: torch.Tensor, 
        constraints: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Méthode explicite appelée par AIOTECH44_EnergyCore."""
        surviving, mask, _ = self.forward(trajectories, constraints)
        return surviving, mask

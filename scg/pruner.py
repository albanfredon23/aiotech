from typing import Tuple, Dict, Any
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SCGEnergyPruner(nn.Module):
    """
    Régularisateur géométrique du Spherical Constraint Graph (SCG).
    - Entraînement : pondération continue différentiable + calcul de la scg_loss.
    - Inférence : élagage dur binaire (pruning franc pour libérer des FLOPs).
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            trajectories: (batch_size, beam_width, emb_dim)
            constraints:  (batch_size, emb_dim), (emb_dim,) ou (batch_size, num_constraints, emb_dim)

        Returns:
            surviving_trajectories: (batch_size, beam_width, emb_dim)
            active_mask:            (batch_size, beam_width)
            scg_scores:             (batch_size, beam_width)
            scg_loss:               scalaire PyTorch différentiable
        """
        # 1. Alignement et diffusion (broadcasting) sécurisés des contraintes
        if constraints.dim() == 1:
            c = constraints.unsqueeze(0).unsqueeze(0)
        elif constraints.dim() == 2:
            c = constraints.unsqueeze(1)
        else:
            c = constraints

        # Normalisation unitaire
        norm_trajectories = torch.nn.functional.normalize(trajectories, p=2, dim=-1)
        norm_constraints = torch.nn.functional.normalize(c, p=2, dim=-1)

        # 2. Calcul vectorisé des violations géométriques
        violations = torch.relu(-norm_trajectories * norm_constraints).sum(dim=-1)
        scg_scores = self.lam * violations
        scg_loss = scg_scores.mean()

        # 3. Filtrage différentiable vs élagage dur
        if self.training:
            active_mask = torch.sigmoid((self.energy_threshold - scg_scores) / self.temp)
            surviving_trajectories = trajectories * active_mask.unsqueeze(-1)
        else:
            active_mask = (scg_scores < self.energy_threshold).float()
            surviving_trajectories = trajectories * active_mask.unsqueeze(-1)

        # 4. Historique borné pour éviter toute fuite de mémoire RAM
        total_violations = float(violations.sum().item())
        self.violation_history.append(total_violations)
        if len(self.violation_history) > self.max_history_len:
            self.violation_history.pop(0)

        return surviving_trajectories, active_mask, scg_scores, scg_loss

    def prune_trajectories(
        self,
        trajectories: torch.Tensor,
        constraints: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias rétrocompatible avec l'ancienne signature à 2 sorties."""
        surviving, mask, _, _ = self.forward(trajectories, constraints)
        return surviving, mask

    def get_pruning_stats(self) -> Dict[str, Any]:
        if not self.violation_history:
            return {"message": "Aucun historique d'élagage disponible"}
        return {
            "total_pruning_events": len(self.violation_history),
            "avg_violations": sum(self.violation_history) / len(self.violation_history),
            "max_violations": max(self.violation_history),
            "min_violations": min(self.violation_history)
        }


class SCGScore:
    """Calculateur de score géométrique SCG."""
    def __init__(self, lam: float = 0.2):
        self.lam = lam

    def compute_penalty(
        self,
        trajectory_state: torch.Tensor,
        constraints: torch.Tensor
    ) -> torch.Tensor:
        norm_state = torch.nn.functional.normalize(trajectory_state, p=2, dim=-1)
        norm_constraints = torch.nn.functional.normalize(constraints, p=2, dim=-1)
        violation = torch.relu(-norm_state * norm_constraints).sum(dim=-1)
        return self.lam * violation

    def is_admissible(
        self,
        trajectory_state: torch.Tensor,
        constraints: torch.Tensor,
        threshold: float = 0.5
    ) -> bool:
        penalty = self.compute_penalty(trajectory_state, constraints)
        return bool(penalty.mean().item() < threshold)

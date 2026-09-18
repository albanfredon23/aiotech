from typing import Tuple
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SCGEnergyPruner(nn.Module):
    """
    Garde-fou géométrique du Spherical Constraint Graph (SCG).
    - Entraînement : pondération continue différentiable + calcul du terme de loss.
    - Inférence : élagage binaire dur (pruning franc).
    """
    def __init__(
        self,
        lam: float = 0.2,
        energy_threshold: float = 0.5,
        temperature: float = 0.1,
        max_history_len: int = 1000
    ):
        super().__init__()
        self.lam = lam
        self.energy_threshold = energy_threshold
        self.temperature = max(temperature, 1e-5)
        self.max_history_len = max_history_len
        self.violation_history = []

    def forward(
        self,
        trajectories: torch.Tensor,
        constraints: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Évalue et élague les trajectoires selon les contraintes géométriques.

        Args:
            trajectories: (batch_size, beam_width, emb_dim)
            constraints:  (batch_size, emb_dim), (emb_dim,) ou (batch_size, num_constraints, emb_dim)

        Returns:
            surviving_trajectories: Trajectoires pondérées ou élaguées
            active_mask:            Masque d'admissibilité (continu ou binaire)
            scg_scores:             Scores scalaires de violation par trajectoire
            scg_loss:               Terme de régularisation différentiable
        """
        # Normalisation unitaire sur la sphère S^(D-1)
        norm_trajectories = torch.nn.functional.normalize(trajectories, p=2, dim=-1)

        # 1. Alignement dimensionnel des contraintes
        if constraints.dim() == 1:
            c = constraints.unsqueeze(0).unsqueeze(0)
        elif constraints.dim() == 2:
            c = constraints.unsqueeze(1)
        else:
            c = constraints

        norm_constraints = torch.nn.functional.normalize(c, p=2, dim=-1)

        # 2. Calcul vectorisé des violations géométriques
        violations = torch.relu(-norm_trajectories * norm_constraints).sum(dim=-1)
        scg_scores = self.lam * violations

        # 3. Filtrage différentiable vs élagage dur
        if self.training:
            # Mode entraînement : sigmoïde tempérée pour propager le gradient
            active_mask = torch.sigmoid((self.energy_threshold - scg_scores) / self.temperature)
            surviving_trajectories = trajectories * active_mask.unsqueeze(-1)
            scg_loss = scg_scores.mean()
        else:
            # Mode inférence : seuil binaire dur pour économiser le calcul
            active_mask = (scg_scores < self.energy_threshold).float()
            surviving_trajectories = trajectories * active_mask.unsqueeze(-1)
            scg_loss = torch.tensor(0.0, device=trajectories.device)

        # 4. Historique borné pour éviter toute fuite mémoire
        total_violations = float(violations.sum().item())
        self.violation_history.append(total_violations)
        if len(self.violation_history) > self.max_history_len:
            self.violation_history.pop(0)

        return surviving_trajectories, active_mask, scg_scores, scg_loss

    # Alias rétrocompatible
    prune_trajectories = forward


class SCGScore:
    """
    Calculateur unitaire ou par lot de score géométrique SCG.
    """
    def __init__(self, lam: float = 0.2):
        self.lam = lam

    def compute_penalty(
        self,
        trajectory_state: torch.Tensor,
        constraints: torch.Tensor
    ) -> torch.Tensor:
        """Calcule la pénalité géométrique sans détruire la dimension de batch."""
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
        """Indique si l'état respecte le seuil d'admissibilité."""
        penalty = self.compute_penalty(trajectory_state, constraints)
        return bool(penalty.mean().item() < threshold)

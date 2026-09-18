from typing import Tuple
import torch
import torch.nn as nn


class SCGEnergyPruner(nn.Module):
    """
    Garde-fou géométrique sphérique (SCG).
    Pénalise continûment les violations de contraintes le long des trajectoires à l'entraînement,
    et applique un élagage franc (pruning) lors de l'inférence.
    """
    def __init__(
        self,
        lam: float = 0.2,
        energy_threshold: float = 0.5,
        temperature: float = 0.1
    ):
        super().__init__()
        self.lam = lam
        self.energy_threshold = energy_threshold
        self.temperature = temperature

    def forward(
        self,
        trajectories: torch.Tensor,
        constraints: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            trajectories: (batch_size, num_trajectories, emb_dim)
            constraints:  (batch_size, emb_dim)

        Returns:
            surviving_trajectories: Trajectoires pondérées ou élaguées
            active_mask: Masque d'activation (continu en train, binaire en eval)
            scg_loss: Terme scalaire de régularisation différentiable
        """
        # Normalisation unitaire sur la sphère S^(D-1)
        norm_trajectories = torch.nn.functional.normalize(trajectories, p=2, dim=-1)
        norm_constraints = torch.nn.functional.normalize(constraints, p=2, dim=-1).unsqueeze(1)

        # Violation angulaire le long de la trajectoire
        # Une projection négative traduit une opposition géométrique à la contrainte
        violations = torch.relu(-norm_trajectories * norm_constraints).sum(dim=-1)
        scg_scores = self.lam * violations

        if self.training:
            # Mode entraînement : masque doux différentiable via sigmoïde tempérée
            # Garantit la rétropropagation vers le planificateur de trajectoires
            active_mask = torch.sigmoid((self.energy_threshold - scg_scores) / self.temperature)
            surviving_trajectories = trajectories * active_mask.unsqueeze(-1)
            scg_loss = scg_scores.mean()
        else:
            # Mode inférence : seuil dur binaire pour économiser le calcul
            active_mask = (scg_scores < self.energy_threshold).float()
            surviving_trajectories = trajectories * active_mask.unsqueeze(-1)
            scg_loss = torch.tensor(0.0, device=trajectories.device)

        return surviving_trajectories, active_mask, scg_loss

    # Alias pour préserver la rétrocompatibilité des appels
    prune_trajectories = forward


class SCGScore(nn.Module):
    """
    Module utilitaire pour l'évaluation ponctuelle de la pénalité géométrique.
    """
    def __init__(self, lam: float = 0.2):
        super().__init__()
        self.lam = lam

    def compute_penalty(
        self,
        trajectory_state: torch.Tensor,
        constraints: torch.Tensor
    ) -> torch.Tensor:
        """Calcule la pénalité géométrique pour un état donné."""
        norm_state = torch.nn.functional.normalize(trajectory_state, p=2, dim=-1)
        norm_constraints = torch.nn.functional.normalize(constraints, p=2, dim=-1)
        violation = torch.relu(-norm_state * norm_constraints).sum(dim=-1)
        return self.lam * violation

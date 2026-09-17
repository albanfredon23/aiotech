
import logging
from typing import Tuple, Dict, Any
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Évalue et élague les trajectoires violant les contraintes sphériques.

        Args:
            trajectories: (batch_size, beam_width, emb_dim)
            constraints:  (batch_size, emb_dim), (emb_dim,) ou (batch_size, num_constraints, emb_dim)

        Returns:
            surviving_trajectories: (batch_size, beam_width, emb_dim)
            mask:                   (batch_size, beam_width)
            scg_loss:               scalaire PyTorch différentiable
        """
        # 1. Alignement et diffusion (broadcasting) sécurisés des contraintes
        if constraints.dim() == 1:
            # (emb_dim,) -> (1, 1, emb_dim)
            c = constraints.unsqueeze(0).unsqueeze(0)
        elif constraints.dim() == 2:
            # (batch_size, emb_dim) -> (batch_size, 1, emb_dim)
            c = constraints.unsqueeze(1)
        elif constraints.dim() == 3:
            # Si multi-contraintes : projection adaptée
            c = constraints
        else:
            c = constraints

        # 2. Calcul vectorisé des violations géométriques
        # Violation = max(0, -trajectoire_i * contrainte_i)
        if c.dim() == 3 and c.size(1) > 1 and trajectories.dim() == 3:
            # Produit matriciel pour cas multi-contraintes
            projections = torch.matmul(trajectories, c.transpose(-1, -2))
            violations = torch.relu(-projections).sum(dim=-1)
        else:
            violations = torch.relu(-trajectories * c).sum(dim=-1)  # (batch_size, beam_width)

        scg_scores = self.lam * violations
        scg_loss = scg_scores.mean()

        # 3. Filtrage différentiable vs élagage dur
        if self.training:
            # Mode entraînement : relaxation continue sigmoïde tempérée
            # Garantit la rétropropagation du gradient vers le planificateur
            soft_mask = torch.sigmoid((self.energy_threshold - scg_scores) / self.temp)
            surviving_trajectories = trajectories * soft_mask.unsqueeze(-1)
            active_mask = soft_mask
        else:
            # Mode inférence : coupure binaire stricte (Green AI)
            hard_mask = (scg_scores < self.energy_threshold).float()
            surviving_trajectories = trajectories * hard_mask.unsqueeze(-1)
            active_mask = hard_mask

        # 4. Historique borné pour éviter toute fuite de mémoire RAM
        total_violations = float(violations.sum().item())
        self.violation_history.append(total_violations)
        if len(self.violation_history) > self.max_history_len:
            self.violation_history.pop(0)

        logger.debug(
            f"SCG Pruner: {active_mask.sum().item():.1f}/{active_mask.numel()} "
            f"trajectoires retenues (violations: {total_violations:.2f})"
        )

        return surviving_trajectories, active_mask, scg_loss

    def prune_trajectories(
        self, 
        trajectories: torch.Tensor, 
        constraints: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Alias rétrocompatible avec l'ancienne signature utilisée dans le code.
        Renvoie directement le couple attendu (surviving_trajectories, active_mask).
        """
        surviving, mask, _ = self.forward(trajectories, constraints)
        return surviving, mask

    def get_pruning_stats(self) -> Dict[str, Any]:
        """Retourne les métriques d'élagage pour les benchmarks."""
        if not self.violation_history:
            return {"message": "Aucun historique d'élagage disponible"}

        return {
            "total_pruning_events": len(self.violation_history),
            "avg_violations": sum(self.violation_history) / len(self.violation_history),
            "max_violations": max(self.violation_history),
            "min_violations": min(self.violation_history)
        }


class SCGScore:
    """
    Calculateur unitaire ou vectorisé de score géométrique SCG.
    """
    def __init__(self, lam: float = 0.2):
        self.lam = lam

    def compute_penalty(
        self, 
        trajectory_state: torch.Tensor, 
        constraints: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la pénalité géométrique sans écraser la dimension de batch.
        
        Args:
            trajectory_state: (..., emb_dim)
            constraints:      (..., emb_dim) ou (emb_dim,)
            
        Returns:
            penalty: tenseur de pénalité de forme (...)
        """
        violation = torch.relu(-trajectory_state * constraints).sum(dim=-1)
        return self.lam * violation

    def is_admissible(
        self, 
        trajectory_state: torch.Tensor, 
        constraints: torch.Tensor, 
        threshold: float = 0.5
    ) -> bool:
        """Vérifie si l'état respecte le seuil d'admissibilité."""
        penalty = self.compute_penalty(trajectory_state, constraints)
        return bool(penalty.mean().item() < threshold)


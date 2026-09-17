import logging
from typing import Dict, Any, Tuple
import torch

logger = logging.getLogger(__name__)

class SCGEnergyPruner:
    """
    Garde-fou géométrique : filtre et élague les branches incohérentes
    avant d'engager des calculs matriciels coûteux.
    """
    def __init__(self, lam: float = 0.2, energy_threshold: float = 0.5, max_history_len: int = 1000):
        """
        Args:
            lam: Coefficient de pénalité pour les violations géométriques (default 0.2)
            energy_threshold: Seuil d'énergie au-delà duquel la trajectoire est coupée (default 0.5)
            max_history_len: Nombre maximal d'événements conservés en mémoire pour éviter les fuites RAM
        """
        self.lam = lam
        self.energy_threshold = energy_threshold
        self.max_history_len = max_history_len
        self.violation_history = []

    def prune_trajectories(self, trajectories: torch.Tensor, 
                           constraints: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Évalue et élague les trajectoires violant les contraintes sphériques.
        
        Args:
            trajectories: (batch_size, beam_width, emb_dim)
            constraints:  (batch_size, emb_dim) ou (emb_dim,)
            
        Returns:
            surviving_trajectories: (batch_size, beam_width, emb_dim)
            active_mask:            (batch_size, beam_width)
        """
        # 1. Alignement dimensionnel automatique via unsqueeze
        if constraints.dim() == 1:
            # (emb_dim,) -> (1, 1, emb_dim) pour broadcast direct sur le batch et le beam
            c = constraints.unsqueeze(0).unsqueeze(0)
        elif constraints.dim() == 2:
            # (batch_size, emb_dim) -> (batch_size, 1, emb_dim)
            c = constraints.unsqueeze(1)
        else:
            c = constraints

        # 2. Calcul vectorisé des violations géométriques
        # Violation = max(0, -trajectoire_i * contrainte_i)
        violations = torch.relu(-trajectories * c).sum(dim=-1)  # (batch_size, beam_width)
        scg_scores = self.lam * violations

        # 3. Élagage "Hard" : coupure nette des branches inadmissibles
        active_mask = (scg_scores < self.energy_threshold).float()
        surviving_trajectories = trajectories * active_mask.unsqueeze(-1)

        # 4. Enregistrement contrôlé pour métriques et logging
        total_violations = float(violations.sum().item())
        self.violation_history.append(total_violations)
        if len(self.violation_history) > self.max_history_len:
            self.violation_history.pop(0)

        logger.debug(
            f"SCG Pruning: {int(active_mask.sum().item())}/{active_mask.numel()} "
            f"trajectoires admises (violations: {total_violations:.2f})"
        )

        return surviving_trajectories, active_mask

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
    Calculateur de score SCG différentiable pour l'évaluation unitaire ou par lot.
    """
    def __init__(self, lam: float = 0.2):
        self.lam = lam

    def compute_penalty(self, trajectory_state: torch.Tensor, 
                        constraints: torch.Tensor) -> torch.Tensor:
        """
        Calcule la pénalité géométrique sans détruire la dimension de batch.
        
        Args:
            trajectory_state: (..., emb_dim)
            constraints:      (..., emb_dim) ou (emb_dim,)
            
        Returns:
            penalty: Tenseur de forme (...) contenant la pénalité scalaire par trajectoire
        """
        violation = torch.relu(-trajectory_state * constraints).sum(dim=-1)
        return self.lam * violation

    def is_admissible(self, trajectory_state: torch.Tensor, 
                      constraints: torch.Tensor, 
                      threshold: float = 0.5) -> bool:
        """Indique si l'état respecte le seuil d'admissibilité géométrique."""
        penalty = self.compute_penalty(trajectory_state, constraints)
        return bool(penalty.mean().item() < threshold)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

class TrajectoryValidator(nn.Module):
    """
    Vérificateur neuro-symbolique de trajectoires de raisonnement.
    Contrôle la cohérence des transitions et le respect des invariants géométriques SCG.
    """
    def __init__(
        self,
        emb_dim: int,
        scg_lambda: float = 0.2,
        scg_threshold: float = 0.60,
        consistency_threshold: float = 0.50
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.scg_lambda = scg_lambda
        self.scg_threshold = scg_threshold
        self.consistency_threshold = consistency_threshold

    def forward(
        self, 
        trajectories: torch.Tensor, 
        constraints: torch.Tensor,
        logic_activations: torch.Tensor = None
    ) -> Dict[str, Any]:
        """
        Args:
            trajectories: (batch_size, beam_width, seq_len, emb_dim) ou (batch_size, beam_width, emb_dim)
            constraints:  (batch_size, num_constraints, emb_dim) ou (batch_size, emb_dim)
            logic_activations: (batch_size, beam_width) ou None
            
        Returns:
            Dictionnaire avec scores SCG, cohérence, masques d'admissibilité et meilleur index.
        """
        # Normalisation en séquence 4D si le tenseur d'entrée est statique 3D
        if trajectories.dim() == 3:
            trajectories = trajectories.unsqueeze(2)  # seq_len = 1
            
        batch_size, beam_width, seq_len, dim = trajectories.size()
        
        # 1. Projection sur l'hypersphère unité S^{D-1}
        norm_traj = F.normalize(trajectories, p=2, dim=-1)
        
        if constraints.dim() == 2:
            norm_constraints = F.normalize(constraints.unsqueeze(1).unsqueeze(2), p=2, dim=-1)
        elif constraints.dim() == 3:
            norm_constraints = F.normalize(constraints.unsqueeze(1), p=2, dim=-1)
        else:
            norm_constraints = F.normalize(constraints, p=2, dim=-1)

        # 2. Calcul vectorisé des violations SCG
        # Projection : <s_t, c>
        projections = torch.matmul(norm_traj, norm_constraints.transpose(-1, -2))
        violations = torch.relu(-projections).sum(dim=-1).mean(dim=-1)  # (batch_size, beam_width)
        
        scg_score = torch.exp(-self.scg_lambda * violations)
        
        # 3. Vérification de la cohérence interne des transitions (si seq_len > 1)
        if seq_len > 1:
            transitions = (norm_traj[:, :, :-1, :] * norm_traj[:, :, 1:, :]).sum(dim=-1)
            consistency_score = torch.clamp((1.0 + transitions.mean(dim=-1)) / 2.0, 0.0, 1.0)
        else:
            consistency_score = torch.ones((batch_size, beam_width), device=trajectories.device)

        # 4. Score logique (Soft-Gödel)
        if logic_activations is not None:
            logic_score = torch.clamp(logic_activations, 0.0, 1.0)
        else:
            logic_score = torch.ones_like(scg_score)

        # 5. Score global d'admissibilité TAP-NN
        # Pondération standard : 30% logique, 35% SCG, 25% cohérence, 10% simplicité
        simplicity_score = torch.exp(torch.tensor(-0.05 * seq_len, device=trajectories.device))
        
        composite_score = (
            0.30 * logic_score +
            0.35 * scg_score +
            0.25 * consistency_score +
            0.10 * simplicity_score
        )

        # 6. Masque booléen d'admissibilité stricte
        is_scg_valid = scg_score >= self.scg_threshold
        is_consistent = consistency_score >= self.consistency_threshold
        admissible_mask = is_scg_valid & is_consistent  # (batch_size, beam_width)

        # Sélection de la trajectoire optimale admissible
        # On masque les non-admissibles par une valeur très basse
        masked_scores = composite_score.clone()
        masked_scores[~admissible_mask] = -1e9
        best_trajectory_idx = torch.argmax(masked_scores, dim=-1)  # (batch_size,)

        return {
            "scg_score": scg_score,
            "consistency_score": consistency_score,
            "composite_score": composite_score,
            "admissible_mask": admissible_mask,
            "best_trajectory_idx": best_trajectory_idx
        }

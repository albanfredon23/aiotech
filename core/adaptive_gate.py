from typing import Tuple
import torch
import torch.nn as nn

class AdaptiveComputeGate(nn.Module):
    """
    Évalue la complexité intrinsèque de la requête pour moduler
    l'effort de calcul et élaguer dynamiquement les nœuds inactifs.
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        self.complexity_evaluator = nn.Linear(emb_dim, 1)
        self.graph_gate = nn.Linear(emb_dim, emb_dim)

    def forward(
        self, 
        query_emb: torch.Tensor, 
        graph_nodes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_emb: Tenseur de la requête (batch_size, emb_dim)
            graph_nodes: Nœuds du graphe (batch_size, num_nodes, emb_dim)

        Returns:
            gated_nodes: Nœuds modulés (batch_size, num_nodes, emb_dim)
            complexity_score: Score de complexité scalaire (batch_size, 1)
        """
        # Score de complexité normalisé entre 0 et 1
        complexity_score = torch.sigmoid(self.complexity_evaluator(query_emb))
        
        # Activation sélective et continue par dimension d'embedding (feature gating)
        gate_activation = torch.sigmoid(self.graph_gate(query_emb))
        gated_nodes = graph_nodes * gate_activation.unsqueeze(1)
        
        return gated_nodes, complexity_score

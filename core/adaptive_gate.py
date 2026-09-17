import torch
import torch.nn as nn

class AdaptiveComputeGate(nn.Module):
    """
    Évalue la complexité intrinsèque de la requête pour moduler
    l'effort de calcul et filtre dynamiquement les dimensions sémantiques.
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        # 1. Scalaire global de complexité (effort de calcul / budget)
        self.complexity_evaluator = nn.Linear(emb_dim, 1)
        # 2. Masque sémantique vectoriel (sélection de caractéristiques)
        self.graph_gate = nn.Linear(emb_dim, emb_dim)

    def forward(self, query_emb: torch.Tensor, graph_nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_emb:   (batch_size, emb_dim)
            graph_nodes: (batch_size, num_nodes, emb_dim)
            
        Returns:
            gated_nodes:      (batch_size, num_nodes, emb_dim)
            complexity_score: (batch_size, 1)
        """
        # Complexité scalaire : [batch_size, 1]
        complexity_score = torch.sigmoid(self.complexity_evaluator(query_emb))
        
        # Activation par dimension : [batch_size, emb_dim] -> [batch_size, 1, emb_dim]
        feature_gate = torch.sigmoid(self.graph_gate(query_emb)).unsqueeze(1)
        
        # Modulation combinée : filtrage sémantique + pondération par complexité
        gated_nodes = graph_nodes * feature_gate * complexity_score.unsqueeze(-1)
        
        return gated_nodes, complexity_score

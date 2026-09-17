import torch
import torch.nn as nn

class DifferentiableBeamSearch(nn.Module):
    """
    Explore l'espace des trajectoires de manière différentiable,
    en sélectionnant les transitions les plus prometteuses.
    """
    def __init__(self, emb_dim: int, beam_width: int = 4):
        super().__init__()
        self.emb_dim = emb_dim
        self.beam_width = beam_width
        self.transition = nn.Linear(emb_dim * 2, emb_dim)
        self.score_head = nn.Linear(emb_dim, 1)

        # Initialisation Xavier pour stabiliser les gradients
        nn.init.xavier_uniform_(self.transition.weight)
        nn.init.zeros_(self.transition.bias)
        nn.init.xavier_uniform_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)

    def forward(self, query_state: torch.Tensor, candidate_nodes: torch.Tensor) -> torch.Tensor:
        """
        Planifie les trajectoires via recherche en faisceau différentiable.

        Args:
            query_state:     (batch_size, emb_dim)
            candidate_nodes: (batch_size, num_nodes, emb_dim)

        Returns:
            selected_trajectories: (batch_size, top_k, emb_dim) où top_k = min(beam_width, num_nodes)
        """
        batch_size, num_nodes, _ = candidate_nodes.size()

        # 1. Expansion de la requête pour croisement avec chaque nœud candidat
        query_expanded = query_state.unsqueeze(1).expand(batch_size, num_nodes, self.emb_dim)

        # 2. Concaténation et calcul des transitions d'états
        pairs = torch.cat([query_expanded, candidate_nodes], dim=-1)
        trajectories = torch.tanh(self.transition(pairs))  # (batch_size, num_nodes, emb_dim)

        # 3. Scoring différentiable des trajectoires candidates
        scores = self.score_head(trajectories).squeeze(-1)  # (batch_size, num_nodes)

        # 4. Détermination sécurisée du top-k
        top_k = min(self.beam_width, num_nodes)
        _, top_indices = torch.topk(scores, k=top_k, dim=-1)  # (batch_size, top_k)

        # 5. Extraction propre avec torch.gather
        # Expansion des indices pour correspondre à (batch_size, top_k, emb_dim)
        gather_indices = top_indices.unsqueeze(-1).expand(-1, -1, self.emb_dim)
        selected_trajectories = torch.gather(trajectories, dim=1, index=gather_indices)

        return selected_trajectories

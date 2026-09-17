import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableBeamSearch(nn.Module):
    """
    Explore l'espace des trajectoires.
    - Entraînement : sélection douce relaxée par Gumbel-Softmax (différentiable).
    - Inférence : top-k dur sans surcoût.
    """
    def __init__(self, emb_dim: int, beam_width: int = 4, tau: float = 1.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.beam_width = beam_width
        self.tau = tau
        self.transition = nn.Linear(emb_dim * 2, emb_dim)
        self.score_head = nn.Linear(emb_dim, 1)

        nn.init.xavier_uniform_(self.transition.weight)
        nn.init.xavier_uniform_(self.score_head.weight)

    def forward(self, query_state: torch.Tensor, candidate_nodes: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = candidate_nodes.size()
        top_k = min(self.beam_width, num_nodes)

        query_expanded = query_state.unsqueeze(1).expand(batch_size, num_nodes, self.emb_dim)
        pairs = torch.cat([query_expanded, candidate_nodes], dim=-1)
        trajectories = torch.tanh(self.transition(pairs))  # (B, N, D)
        scores = self.score_head(trajectories).squeeze(-1)  # (B, N)

        if self.training:
            # Gumbel-Softmax itératif pour simuler un top-k différentiable
            logits = scores.clone()
            soft_selections = []
            for _ in range(top_k):
                weights = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1) # STE
                selected_step = torch.bmm(weights.unsqueeze(1), trajectories).squeeze(1)
                soft_selections.append(selected_step)
                # Masquage doux pour pénaliser les nœuds déjà retenus
                logits = logits - (weights * 1e9)
            return torch.stack(soft_selections, dim=1) # (B, top_k, D)
        else:
            # Inférence classique dure
            _, top_indices = torch.topk(scores, k=top_k, dim=-1)
            gather_indices = top_indices.unsqueeze(-1).expand(-1, -1, self.emb_dim)
            return torch.gather(trajectories, dim=1, index=gather_indices)

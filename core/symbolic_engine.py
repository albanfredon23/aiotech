import torch
import torch.nn as nn
from typing import Optional

class GodelLogic(nn.Module):
    """
    Implémente la t-norme différentiable de Gödel pour le calcul de conjonction (MIN doux).
    Approximation différentiable : min(a, b) ≈ -τ * log(exp(-a/τ) + exp(-b/τ))
    """
    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = max(tau, 1e-6)  # Évite toute division instable

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Conjonction logique différentiable entre deux prédicats ou règles.
        
        Args:
            a, b: Tenseurs de degrés de vérité dans [0, 1] de même forme.
        Returns:
            Tenseur résultant clampé strictement dans [0, 1].
        """
        # Clamping préventif des entrées
        a_clamped = torch.clamp(a, 0.0, 1.0)
        b_clamped = torch.clamp(b, 0.0, 1.0)

        # Soft-min via logsumexp
        log_terms = torch.stack([-a_clamped / self.tau, -b_clamped / self.tau], dim=0)
        result = -self.tau * torch.logsumexp(log_terms, dim=0)

        # Clamping de sortie : compense le décalage négatif du soft-min en (0, 0)
        return torch.clamp(result, min=0.0, max=1.0)


class NeuroSymbolicEngine(nn.Module):
    """
    Évalue l'admissibilité logique des trajectoires via des règles formelles latentes
    et combine logique continue (Gödel) et représentations vectorielles.
    """
    def __init__(self, emb_dim: int, num_rules: int = 16, tau: float = 0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_rules = num_rules

        # Embeddings des règles et projection
        self.rule_embeddings = nn.Parameter(torch.randn(num_rules, emb_dim) / (emb_dim ** 0.5))
        self.rule_evaluator = nn.Linear(emb_dim, num_rules)
        self.godel = GodelLogic(tau=tau)

        # Initialisation Xavier pour assurer la stabilité des gradients au démarrage
        nn.init.xavier_uniform_(self.rule_evaluator.weight)
        nn.init.zeros_(self.rule_evaluator.bias)

    def forward(self, context_emb: torch.Tensor) -> torch.Tensor:
        """
        Calcule les scores de satisfaction de chaque règle logique.

        Args:
            context_emb: (batch_size, emb_dim) ou (batch_size, seq_len, emb_dim)
        Returns:
            activations: Tenseur de probabilités dans [0, 1] de même préfixe de batch
        """
        rule_logits = self.rule_evaluator(context_emb)
        return torch.sigmoid(rule_logits)

    def conjunction(self, rule_acts_a: torch.Tensor, rule_acts_b: torch.Tensor) -> torch.Tensor:
        """
        Applique la t-norme de Gödel pour évaluer la conjonction de deux ensembles de règles.
        """
        return self.godel(rule_acts_a, rule_acts_b)

    def aggregate_rules(self, activations: torch.Tensor, rule_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Projette les activations scalaires de règles dans l'espace des embeddings.

        Args:
            activations: (batch_size, num_rules) ou (batch_size, seq_len, num_rules)
            rule_states: (num_rules, emb_dim), utilise self.rule_embeddings par défaut.
        Returns:
            aggregated: (batch_size, emb_dim) ou (batch_size, seq_len, emb_dim)
        """
        states = self.rule_embeddings if rule_states is None else rule_states
        return torch.matmul(activations, states)

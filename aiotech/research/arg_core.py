"""
ARGCore – version différentiable (PyTorch) du raisonnement contraint, successeur de
AIOTECH44_EnergyCore. Module de RECHERCHE : il doit être entraîné sur des données de la
tâche visée avant que ses sorties aient un sens. Non entraîné, il ne prouve rien.

Changements par rapport à AIOTECH 44 :
  • plus de projection sur l'hypersphère : les contraintes sont des points d'ancrage avec
    un rayon appris, la violation est un dépassement de distance euclidienne,
        v = relu(||s - c||_2 - r) / sigma,   sigma = sqrt(D)   (échelle naturelle en dim D)
  • les trajectoires sont déroulées pas à pas (s_0 -> s_1 -> ... -> s_H) et la cohérence de
    chaque transition est un critère :  coh_h = exp(-||s_{h+1} - s_h||^2 / (2 sigma_s^2))
  • admissibilité d'une trajectoire = T_G lissée (log-sum-exp tempéré) de TOUS les critères
    (contraintes à chaque pas, transitions, satisfaction des règles logiques) ;
  • bug corrigé : le cœur v44 dépaquetait 3 sorties du pruner qui en renvoyait 4 (plantage) ;
  • les documents retenus sont les k plus PROCHES de la requête (v44 gardait les k premiers
    de la liste, quel que soit leur contenu) ; résumé mémoire = moyenne pondérée masquée ;
  • le budget k reçoit un vrai gradient via une pénalité de coût (STE), au lieu d'un
    gradient "conservé" qui ne servait à rien ;
  • masquage Gumbel des nœuds déjà choisis par -inf (v44 soustrayait 1e9 sur un poids
    à gradient, ce qui injectait des gradients de l'ordre de 1e9).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_godel(x: torch.Tensor, temperature: float = 0.05, dim: int = -1) -> torch.Tensor:
    """min lissé : -T * logsumexp(-x / T). Différentiable, tend vers min(x) quand T -> 0."""
    t = max(float(temperature), 1e-6)
    return -t * torch.logsumexp(-x / t, dim=dim)


class AdaptiveComputeGate(nn.Module):
    """Complexité de la requête et modulation des nœuds latents (déterministe)."""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.complexity = nn.Linear(emb_dim, 1)
        self.feature_gate = nn.Linear(emb_dim, emb_dim)

    def forward(self, query: torch.Tensor, nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        complexity = torch.sigmoid(self.complexity(query))            # (B, 1)
        gate = torch.sigmoid(self.feature_gate(query)).unsqueeze(1)   # (B, 1, D)
        return nodes * gate, complexity


class SpecializedAgents(nn.Module):
    """Experts pondérés ; en inférence, les experts sous le seuil ne sont pas calculés."""

    def __init__(self, emb_dim: int, num_agents: int = 4, threshold: float = 0.05, lr_online: float = 0.05):
        super().__init__()
        self.threshold = threshold
        self.lr_online = lr_online
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim), nn.LayerNorm(emb_dim))
            for _ in range(num_agents)
        ])
        self.agent_logits = nn.Parameter(torch.zeros(num_agents))

    def forward(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.agent_logits, dim=0)
        out = torch.zeros_like(query)
        for i, expert in enumerate(self.experts):
            if self.training or float(weights[i]) > self.threshold:
                out = out + weights[i] * expert(query)
        return out, weights

    @torch.no_grad()
    def online_update(self, agent_losses: torch.Tensor) -> None:
        """Mise à jour sans rétropropagation : les agents à perte élevée perdent du poids."""
        self.agent_logits -= self.lr_online * (agent_losses - agent_losses.mean())


class NeuroSymbolicEngine(nn.Module):
    """Règles latentes : degrés de satisfaction dans [0,1] et leur conjonction de Gödel."""

    def __init__(self, emb_dim: int, num_rules: int = 16, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.rule_embeddings = nn.Parameter(torch.randn(num_rules, emb_dim) / math.sqrt(emb_dim))
        self.evaluator = nn.Linear(emb_dim, num_rules)
        nn.init.xavier_uniform_(self.evaluator.weight)
        nn.init.zeros_(self.evaluator.bias)

    def forward(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        acts = torch.sigmoid(self.evaluator(ctx))                          # (B, R)
        logic_context = acts @ self.rule_embeddings                        # (B, D)
        conj = soft_godel(acts, self.temperature, dim=-1).clamp(0.0, 1.0)  # (B,)
        return acts, logic_context, conj


class StepwisePlanner(nn.Module):
    """Choisit K points de départ (Gumbel-STE à l'entraînement, top-k en inférence) puis
    déroule H transitions s_h -> s_{h+1} avec attention sur les nœuds du graphe."""

    def __init__(self, emb_dim: int, beam_width: int = 4, horizon: int = 3, tau_gumbel: float = 1.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.beam_width = beam_width
        self.horizon = horizon
        self.tau_gumbel = tau_gumbel
        self.start = nn.Linear(2 * emb_dim, emb_dim)
        self.score = nn.Linear(emb_dim, 1)
        self.query_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.step = nn.Linear(2 * emb_dim, emb_dim)
        for m in (self.start, self.score, self.step):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, query: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        b, n, d = nodes.shape
        k = min(self.beam_width, n)
        pairs = torch.cat([query.unsqueeze(1).expand(b, n, d), nodes], dim=-1)
        starts = torch.tanh(self.start(pairs))                 # (B, N, D)
        logits = self.score(starts).squeeze(-1)                # (B, N)

        if self.training:
            picks = []
            masked = logits
            for _ in range(k):
                w = F.gumbel_softmax(masked, tau=self.tau_gumbel, hard=True, dim=-1)  # one-hot STE
                picks.append(torch.bmm(w.unsqueeze(1), starts).squeeze(1))
                masked = masked.masked_fill(w.detach() > 0.5, float("-inf"))
            s = torch.stack(picks, dim=1)                       # (B, K, D)
        else:
            idx = logits.topk(k, dim=-1).indices
            s = torch.gather(starts, 1, idx.unsqueeze(-1).expand(b, k, d))

        states = [s]
        keys = nodes                                           # (B, N, D)
        for _ in range(self.horizon):
            att = torch.softmax(self.query_proj(s) @ keys.transpose(1, 2) / math.sqrt(d), dim=-1)  # (B, K, N)
            ctx = att @ keys                                   # (B, K, D)
            s = torch.tanh(self.step(torch.cat([s, ctx], dim=-1)))
            states.append(s)
        return torch.stack(states, dim=2)                      # (B, K, H+1, D)


class EuclideanAdmissibility(nn.Module):
    """Remplace SCGEnergyPruner. Aucune normalisation sur la sphère."""

    def __init__(self, emb_dim: int, tau: float = 0.5, lam: float = 1.0, temperature: float = 0.05,
                 mask_temperature: float = 0.05, margin: float = 0.05):
        super().__init__()
        self.tau = tau
        self.lam = lam
        self.temperature = temperature
        self.mask_temperature = mask_temperature
        self.margin = margin
        self.register_buffer("sigma", torch.tensor(math.sqrt(emb_dim)))
        self.register_buffer("sigma_step", torch.tensor(math.sqrt(emb_dim) / 2.0))
        # rayon admissible autour de chaque ancre, appris (softplus pour rester > 0)
        self.radius_raw = nn.Parameter(torch.tensor(math.log(math.expm1(math.sqrt(emb_dim)))))

    @staticmethod
    def _as_anchors(constraints: torch.Tensor, batch: int) -> torch.Tensor:
        if constraints.dim() == 1:
            return constraints.view(1, 1, -1).expand(batch, 1, -1)
        if constraints.dim() == 2:
            return constraints.unsqueeze(1)
        return constraints                                      # (B, C, D)

    def forward(self, traj: torch.Tensor, constraints: torch.Tensor,
                rule_conj: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        b, k, t, d = traj.shape
        anchors = self._as_anchors(constraints, b)              # (B, C, D)
        c = anchors.shape[1]
        diff = traj.unsqueeze(3) - anchors.view(b, 1, 1, c, d)  # (B, K, T, C, D)
        dist = torch.sqrt((diff * diff).sum(-1) + 1e-8)         # (B, K, T, C)
        radius = F.softplus(self.radius_raw)
        viol = F.relu(dist - radius) / self.sigma
        con = torch.exp(-self.lam * viol).flatten(2)            # (B, K, T*C)

        delta = traj[:, :, 1:] - traj[:, :, :-1]                # (B, K, H, D)
        coh = torch.exp(-(delta * delta).sum(-1) / (2 * self.sigma_step ** 2))  # (B, K, H)

        parts = [con, coh]
        if rule_conj is not None:
            parts.append(rule_conj.view(b, 1, 1).expand(b, k, 1))
        adm = soft_godel(torch.cat(parts, dim=-1), self.temperature, dim=-1).clamp(0.0, 1.0)  # (B, K)

        if self.training:
            mask = torch.sigmoid((adm - self.tau) / self.mask_temperature)
            fallback = torch.zeros(b, dtype=torch.bool, device=traj.device)
        else:
            mask = (adm >= self.tau).float()
            fallback = mask.sum(-1) == 0
            if fallback.any():  # repli : la trajectoire la moins dégradée
                rows = fallback.nonzero(as_tuple=True)[0]
                mask = mask.clone()
                mask[rows, adm.argmax(-1)[rows]] = 1.0
        loss = F.relu(self.tau + self.margin - adm).mean()
        return {"admissibility": adm, "mask": mask, "fallback": fallback, "loss": loss,
                "violation_mean": viol.mean().detach(), "coherence_mean": coh.mean().detach()}


class ContextBudget(nn.Module):
    """k dynamique : les k documents les plus proches (distance euclidienne) de la requête."""

    def __init__(self, emb_dim: int, base_k: int = 2, max_k: int = 20):
        super().__init__()
        self.base_k = max(1, base_k)
        self.max_k = max(self.base_k + 1, max_k)
        self.register_buffer("sigma", torch.tensor(math.sqrt(emb_dim)))
        self.gate = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, query: torch.Tensor, docs: torch.Tensor, admissibility: torch.Tensor,
                complexity: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, n, d = docs.shape
        relevance = -((docs - query.unsqueeze(1)) ** 2).sum(-1) / (2 * self.sigma ** 2)   # (B, N)
        ratio = self.gate(torch.stack([admissibility, complexity.view(b)], dim=-1)).squeeze(-1)  # (B,)
        k_max = min(self.max_k, n)
        base = min(self.base_k, k_max)
        k_cont = base + ratio * max(k_max - base, 0)
        k_disc = torch.clamp(k_cont.round(), min=base, max=k_max).long()
        k_ste = k_disc.float() + (k_cont - k_cont.detach())      # gradient vers la porte

        kb = int(k_disc.max().item())
        top = relevance.topk(kb, dim=-1)
        picked = torch.gather(docs, 1, top.indices.unsqueeze(-1).expand(b, kb, d))      # (B, kb, D)
        valid = torch.arange(kb, device=docs.device).unsqueeze(0) < k_disc.unsqueeze(-1)  # (B, kb)
        w = torch.softmax(top.values.masked_fill(~valid, float("-inf")), dim=-1)
        summary = (w.unsqueeze(-1) * picked).sum(1)                                      # (B, D)
        return {"summary": summary, "k": k_ste, "k_discrete": k_disc, "valid": valid,
                "indices": top.indices, "budget_loss": (k_ste / k_max).mean()}


class ARGCore(nn.Module):
    def __init__(self, emb_dim: int = 256, num_nodes: int = 50, num_agents: int = 4, num_rules: int = 16,
                 beam_width: int = 4, horizon: int = 3, tau: float = 0.5, base_k: int = 2, max_k: int = 20,
                 budget_weight: float = 0.05):
        super().__init__()
        self.emb_dim, self.num_nodes = emb_dim, num_nodes
        self.budget_weight = budget_weight
        self.gate = AdaptiveComputeGate(emb_dim)
        self.agents = SpecializedAgents(emb_dim, num_agents)
        self.logic = NeuroSymbolicEngine(emb_dim, num_rules)
        self.planner = StepwisePlanner(emb_dim, beam_width, horizon)
        self.admissibility = EuclideanAdmissibility(emb_dim, tau=tau)
        self.budget = ContextBudget(emb_dim, base_k, max_k)
        self.policy_head = nn.Linear(3 * emb_dim, num_nodes)

    def forward(self, query: torch.Tensor, docs: torch.Tensor, nodes: torch.Tensor,
                constraints: torch.Tensor) -> Dict[str, Any]:
        gated, complexity = self.gate(query, nodes)
        agent_ctx, agent_w = self.agents(query)
        _, logic_ctx, rule_conj = self.logic(agent_ctx)
        fused = query + logic_ctx

        traj = self.planner(fused, gated)                                  # (B, K, H+1, D)
        adm = self.admissibility(traj, constraints, rule_conj)
        mask = adm["mask"]                                                 # (B, K)
        last = traj[:, :, -1, :]
        traj_ctx = (mask.unsqueeze(-1) * last).sum(1) / mask.sum(1, keepdim=True).clamp_min(1e-6)

        best_adm = adm["admissibility"].max(dim=-1).values                 # (B,)
        mem = self.budget(fused, docs, best_adm, complexity)

        features = torch.cat([gated.mean(1), traj_ctx, mem["summary"]], dim=-1)
        policy = self.policy_head(features)
        aux = adm["loss"] + self.budget_weight * mem["budget_loss"]
        n_docs = docs.shape[1]
        return {
            "policy": policy,
            "trajectories": traj,
            "admissibility": adm["admissibility"],
            "mask": mask,
            "fallback": adm["fallback"],
            "complexity": complexity,
            "agent_weights": agent_w,
            "budget_k": mem["k"],
            "doc_indices": mem["indices"],
            "docs_kept_ratio": float(mem["k_discrete"].float().mean().item() / max(n_docs, 1)),
            "aux_loss": aux,
        }

    def summary(self) -> Dict[str, int]:
        return {"emb_dim": self.emb_dim, "num_nodes": self.num_nodes,
                "params": sum(p.numel() for p in self.parameters())}

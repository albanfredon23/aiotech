"""
Compilateur de mutations + invariants + porte de benchmark (repris de v3, corrigé).

Le nom "Gödel Machine" de v3 est retiré : une machine de Gödel (Schmidhuber) exige une
preuve formelle d'amélioration avant toute réécriture. Ce module fait autre chose, de plus
modeste et vérifiable : il propose des transformations, les rejette si un invariant
exécutable échoue (forme, équivalence numérique, gradients, déterminisme, finitude,
bornes), puis exige un gain de vitesse statistiquement significatif (ABBA + bootstrap).

Correction M1 : v3 fusionnait deux couches linéaires même séparées par une activation
(Linear -> ReLU -> Linear), ce qui change la fonction calculée. La fusion n'est appliquée
qu'à deux Linear strictement consécutives, seul cas où elle est exacte.
"""
from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class MutationCompiler:
    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []

    def generate(self, module: nn.Module) -> List[Tuple[str, nn.Module]]:
        out = []
        for name, fn in [("M1_fuse_linear", self.fuse_linear), ("M2_prune_l1", self.prune_l1),
                         ("M3_low_rank", self.low_rank), ("M4_fold_batchnorm", self.fold_batchnorm)]:
            try:
                cand = fn(module)
                if cand is not None:
                    out.append((name, cand))
                    self.history.append({"mutation": name, "status": "generated"})
            except Exception as exc:  # une mutation qui échoue est simplement écartée
                self.history.append({"mutation": name, "status": "skipped", "reason": str(exc)})
        return out

    @staticmethod
    def fuse_linear(module: nn.Module) -> Optional[nn.Module]:
        if not isinstance(module, nn.Sequential):
            return None
        layers = list(module.children())
        out: List[nn.Module] = []
        i, applied = 0, False
        while i < len(layers):
            a = layers[i]
            b = layers[i + 1] if i + 1 < len(layers) else None
            if isinstance(a, nn.Linear) and isinstance(b, nn.Linear) and a.out_features == b.in_features:
                fused = nn.Linear(a.in_features, b.out_features, bias=True)
                with torch.no_grad():
                    fused.weight.copy_(b.weight @ a.weight)
                    ba = a.bias if a.bias is not None else torch.zeros(a.out_features)
                    bb = b.bias if b.bias is not None else torch.zeros(b.out_features)
                    fused.bias.copy_(b.weight @ ba + bb)
                out.append(fused)
                i += 2
                applied = True
            else:
                out.append(copy.deepcopy(a))
                i += 1
        return nn.Sequential(*out) if applied else None

    @staticmethod
    def prune_l1(module: nn.Module, amount: float = 0.2) -> Optional[nn.Module]:
        cand = copy.deepcopy(module)
        done = False
        for sub in cand.modules():
            if isinstance(sub, nn.Linear):
                prune.l1_unstructured(sub, name="weight", amount=amount)
                prune.remove(sub, "weight")
                done = True
        return cand if done else None

    @staticmethod
    def low_rank(module: nn.Module, rank_frac: float = 0.5) -> Optional[nn.Module]:
        cand = copy.deepcopy(module)
        done = False
        for sub in cand.modules():
            if isinstance(sub, nn.Linear):
                w = sub.weight.data
                r = max(1, int(min(w.shape) * rank_frac))
                u, s, vh = torch.linalg.svd(w, full_matrices=False)
                sub.weight.data.copy_((u[:, :r] * s[:r]) @ vh[:r, :])
                done = True
        return cand if done else None

    @staticmethod
    def fold_batchnorm(module: nn.Module) -> Optional[nn.Module]:
        if not isinstance(module, nn.Sequential):
            return None
        layers = list(module.children())
        out: List[nn.Module] = []
        i, applied = 0, False
        while i < len(layers):
            lin = layers[i]
            bn = layers[i + 1] if i + 1 < len(layers) else None
            if isinstance(lin, nn.Linear) and isinstance(bn, nn.BatchNorm1d) and bn.running_var is not None:
                std = torch.sqrt(bn.running_var + bn.eps)
                w = lin.weight.data / std.unsqueeze(1)
                b = ((lin.bias.data if lin.bias is not None else torch.zeros(lin.out_features)) - bn.running_mean) / std
                if bn.affine:
                    w = w * bn.weight.data.unsqueeze(1)
                    b = b * bn.weight.data + bn.bias.data
                new = nn.Linear(lin.in_features, lin.out_features, bias=True)
                with torch.no_grad():
                    new.weight.copy_(w)
                    new.bias.copy_(b)
                out.append(new)
                i += 2
                applied = True
            else:
                out.append(copy.deepcopy(lin))
                i += 1
        return nn.Sequential(*out) if applied else None


class InvariantRegistry:
    """Invariants exécutables. I1 (structure) n'est exigé que si demandé : une fusion change
    légitimement la structure tout en préservant la fonction (I2)."""

    @staticmethod
    def numerical(cand: nn.Module, ref: nn.Module, x: torch.Tensor, atol: float = 1e-4) -> Tuple[bool, str]:
        cand.eval(); ref.eval()
        with torch.no_grad():
            diff = float((cand(x) - ref(x)).abs().max())
        return diff <= atol, f"max_diff={diff:.2e} (tol {atol:.0e})"

    @staticmethod
    def input_gradient(cand: nn.Module, ref: nn.Module, x: torch.Tensor, atol: float = 1e-3) -> Tuple[bool, str]:
        xc = x.clone().requires_grad_(True)
        xr = x.clone().requires_grad_(True)
        cand(xc).sum().backward()
        ref(xr).sum().backward()
        diff = float((xc.grad - xr.grad).abs().max())
        return diff <= atol, f"max_grad_diff={diff:.2e}"

    @staticmethod
    def determinism(m: nn.Module, x: torch.Tensor) -> Tuple[bool, str]:
        m.eval()
        with torch.no_grad():
            a, b = m(x), m(x)
        return bool(torch.equal(a, b)), "ok" if torch.equal(a, b) else "sorties non déterministes"

    @staticmethod
    def finite_and_bounded(m: nn.Module, x: torch.Tensor, bound: float = 1e4) -> Tuple[bool, str]:
        with torch.no_grad():
            y = m(x)
        if not torch.isfinite(y).all():
            return False, "NaN/Inf"
        return bool(y.abs().max() <= bound), f"max|y|={float(y.abs().max()):.3g}"

    @classmethod
    def validate_all(cls, cand: nn.Module, ref: nn.Module, x: torch.Tensor,
                     atol: float = 1e-4) -> Dict[str, Tuple[bool, str]]:
        return {
            "numerical": cls.numerical(cand, ref, x, atol),
            "input_gradient": cls.input_gradient(cand, ref, x),
            "determinism": cls.determinism(cand, x),
            "finite_and_bounded": cls.finite_and_bounded(cand, x),
        }


class BenchmarkGate:
    """Admet un candidat si la borne haute de l'IC bootstrap de médiane(t_cand - t_ref)
    est sous le gain requis. Mesures appariées alternées A-B / B-A."""

    def __init__(self, pairs: int = 30, bootstrap: int = 2000, alpha: float = 0.05, warmup: int = 5, seed: int = 0):
        self.pairs, self.bootstrap, self.alpha, self.warmup = pairs, bootstrap, alpha, warmup
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _t(m: nn.Module, x: torch.Tensor) -> float:
        t0 = time.perf_counter()
        m(x)
        return (time.perf_counter() - t0) * 1e3

    def validate(self, cand: nn.Module, ref: nn.Module, x: torch.Tensor, target_speedup: float = 1.05) -> Dict[str, Any]:
        cand.eval(); ref.eval()
        with torch.no_grad():
            for _ in range(self.warmup):
                cand(x); ref(x)
            deltas, refs = [], []
            for i in range(self.pairs):
                if i % 2 == 0:
                    tr, tc = self._t(ref, x), self._t(cand, x)
                else:
                    tc, tr = self._t(cand, x), self._t(ref, x)
                deltas.append(tc - tr)
                refs.append(tr)
        d = np.array(deltas)
        boots = np.median(self.rng.choice(d, size=(self.bootstrap, d.size), replace=True), axis=1)
        lo, hi = np.percentile(boots, [100 * self.alpha / 2, 100 * (1 - self.alpha / 2)])
        med_ref = float(np.median(refs))
        required = med_ref * (1.0 / target_speedup - 1.0)
        return {"is_valid": bool(hi < required), "median_ref_ms": round(med_ref, 4),
                "median_delta_ms": round(float(np.median(d)), 4), "ci": (round(float(lo), 4), round(float(hi), 4)),
                "required_delta_ms": round(required, 4)}


def search(module: nn.Module, x: torch.Tensor, atol: float = 1e-4, target_speedup: float = 1.05) -> Dict[str, Any]:
    """Génère les mutations, garde celles qui passent les invariants ET la porte de vitesse."""
    compiler, gate = MutationCompiler(), BenchmarkGate()
    report = []
    for name, cand in compiler.generate(module):
        inv = InvariantRegistry.validate_all(cand, module, x, atol)
        ok = all(v[0] for v in inv.values())
        bench = gate.validate(cand, module, x, target_speedup) if ok else None
        report.append({"mutation": name, "invariants": {k: v[1] for k, v in inv.items()},
                       "invariants_ok": ok, "benchmark": bench,
                       "accepted": bool(ok and bench and bench["is_valid"])})
    return {"candidates": report, "accepted": [r["mutation"] for r in report if r["accepted"]]}

"""
Outils appelables par le LLM (désactivés par défaut : AIOTECH_ENABLE_TOOLS=false).

Corrections de sécurité par rapport à v3 :
- `run_python` exécutait du code fourni par le modèle avec `exec`. Restreindre
  `__builtins__` ne constitue pas un bac à sable (évasion classique via
  `().__class__.__base__.__subclasses__()`). Un prompt injecté dans un document pouvait
  donc exécuter du code sur le serveur. Il est remplacé par `calculator`, un évaluateur
  arithmétique sur l'arbre syntaxique, sans exécution de code.
- `read_file` lisait n'importe quel chemin (y compris .env et ses clés API). Il est
  désormais confiné à un répertoire dédié (AIOTECH_TOOLS_DIR).
"""
from __future__ import annotations

import ast
import math
import operator
import os
from typing import Any, Callable, Dict, List

TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Évalue une expression arithmétique (+ - * / // % **, parenthèses, sqrt, log, exp, sin, cos, round, abs, min, max).",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                           "required": ["expression"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lit un fichier texte du répertoire documentaire autorisé.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                           "required": ["path"]},
        },
    },
]

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: Dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "round": round, "abs": abs,
    "min": min, "max": max,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval(expression: str) -> float:
    if len(expression) > 500:
        raise ValueError("expression trop longue")
    tree = ast.parse(expression, mode="eval")

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
                raise ValueError("puissance trop grande")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _FUNCS and not node.keywords):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"élément non autorisé : {type(node).__name__}")

    return ev(tree)


def _read_file(path: str, root: str) -> str:
    root_real = os.path.realpath(root)
    target = os.path.realpath(os.path.join(root_real, path))
    if os.path.commonpath([root_real, target]) != root_real:
        return "[ERROR] accès refusé : chemin hors du répertoire autorisé"
    try:
        with open(target, "r", encoding="utf-8") as f:
            return f.read(4000)
    except OSError as exc:
        return f"[ERROR] {exc.strerror or exc}"


def dispatch_tool(name: str, args: Dict[str, Any], tools_dir: str = "./sandbox") -> str:
    if name == "calculator":
        try:
            value = safe_eval(str(args.get("expression", "")))
            return repr(round(value, 12)) if isinstance(value, float) else repr(value)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError) as exc:
            return f"[ERROR] {exc}"
    if name == "read_file":
        return _read_file(str(args.get("path", "")), tools_dir)
    return f"[UNKNOWN TOOL] {name}"

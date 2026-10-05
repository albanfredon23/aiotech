"""
Client AIOTECH : le pipeline complet devant n'importe quel LLM (via LiteLLM).

    requête
      │ 1. PromptShield ........ injection / contenu dangereux bloqués, PII masquées
      │ 2. ARG + budget ........ seuls les segments admissibles des documents sont injectés
      │ 3. Cache ............... réponse réutilisée si question équivalente, même contexte
      │ 4. Routage ............. modèle économique si la demande est simple
      │ 5. Disjoncteur ......... repli vers un autre fournisseur si le premier tombe
      │ 6. Appel LLM ........... (outils optionnels, confinés)
      │ 7. Validation JSON ..... relance si la sortie ne respecte pas le schéma
      │ 8. Chaîne de Gödel ..... chaque étape de la réponse est-elle atteignable depuis le contexte ?
      ▼ 9. Comptabilité ........ tokens et coût réels vs coût de référence (modèle phare + tout le contexte)

Sans LiteLLM (ou avec AIOTECH_OFFLINE=1), un LLM "stub" déterministe répond : tout le
pipeline reste testable sans clé API, et les tokens sont estimés sur les vrais messages.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Sequence

from aiotech.config import Settings
from aiotech.gateway.cache import SemanticCache, context_fingerprint
from aiotech.gateway.circuit_breaker import CircuitBreaker, provider_of
from aiotech.gateway.guardrails import OutputValidator, PromptShield
from aiotech.gateway.pricing import PriceTable
from aiotech.gateway.router import ModelSelector
from aiotech.gateway.tools import TOOL_SPECS, dispatch_tool
from aiotech.reasoning.arg import ArgConstraints, ReachabilityGate
from aiotech.reasoning.context_budget import ContextBudgeter, ContextSelection, DocumentLike
from aiotech.text import estimate_tokens

try:  # LiteLLM est optionnel
    import litellm  # type: ignore
    _HAS_LITELLM = True
except ImportError:  # pragma: no cover - dépend de l'environnement
    litellm = None
    _HAS_LITELLM = False


@dataclass
class LLMResult:
    content: str
    tokens_in: int
    tokens_out: int
    exact_usage: bool
    error: Optional[str] = None


LLMCall = Callable[[str, List[Dict[str, Any]], int, Optional[float], bool], Awaitable[LLMResult]]


def _messages_tokens(messages: Sequence[Dict[str, Any]]) -> int:
    # ~4 tokens de structure par message, en plus du contenu
    return sum(estimate_tokens(str(m.get("content") or "")) + 4 for m in messages)


async def stub_llm(model: str, messages: List[Dict[str, Any]], max_tokens: int,
                   temperature: Optional[float], tools: bool) -> LLMResult:
    """LLM factice déterministe pour les tests et le mode hors ligne."""
    user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    content = f"[STUB {model}] Réponse simulée à : {user[:160]}"
    return LLMResult(content, _messages_tokens(messages), estimate_tokens(content), exact_usage=False)


def _litellm_factory(settings: Settings) -> LLMCall:
    async def call(model: str, messages: List[Dict[str, Any]], max_tokens: int,
                   temperature: Optional[float], tools: bool) -> LLMResult:
        kwargs: Dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = TOOL_SPECS
        msgs = list(messages)
        tok_in = tok_out = 0
        for _ in range(4):  # au plus 3 tours d'outils
            resp = await litellm.acompletion(**{**kwargs, "messages": msgs})
            usage = getattr(resp, "usage", None)
            if usage:
                tok_in += int(getattr(usage, "prompt_tokens", 0) or 0)
                tok_out += int(getattr(usage, "completion_tokens", 0) or 0)
            msg = resp.choices[0].message
            calls = getattr(msg, "tool_calls", None)
            if not (tools and calls):
                content = msg.content or ""
                exact = usage is not None
                if not exact:
                    tok_in, tok_out = _messages_tokens(messages), estimate_tokens(content)
                return LLMResult(content, tok_in, tok_out, exact)
            msgs.append({"role": "assistant", "content": msg.content, "tool_calls": calls})
            for tc in calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                msgs.append({"role": "tool", "tool_call_id": tc.id,
                             "content": dispatch_tool(tc.function.name, args, settings.tools_dir)})
        return LLMResult("", tok_in, tok_out, True, error="tool_loop_limit")

    return call


class AiotechClient:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        cache: Optional[SemanticCache] = None,
        shield: Optional[PromptShield] = None,
        validator: Optional[OutputValidator] = None,
        breaker: Optional[CircuitBreaker] = None,
        selector: Optional[ModelSelector] = None,
        budgeter: Optional[ContextBudgeter] = None,
        prices: Optional[PriceTable] = None,
        llm_call: Optional[LLMCall] = None,
    ):
        s = self.settings = settings or Settings.from_env()
        self.cache = cache or SemanticCache(threshold=s.cache_threshold, max_entries=s.cache_max_entries,
                                            ttl_seconds=s.cache_ttl_seconds)
        self.shield = shield or PromptShield()
        self.validator = validator or OutputValidator()
        self.breaker = breaker or CircuitBreaker(s.cb_failure_threshold, s.cb_recovery_seconds)
        self.selector = selector or ModelSelector(s.cheap_model, s.flagship_model, s.router_threshold)
        self.budgeter = budgeter or ContextBudgeter(
            gate=ReachabilityGate(tau=s.arg_tau), token_budget=s.context_token_budget,
            max_chunks=s.arg_max_chunks, min_chunks=s.arg_min_chunks, chunk_tokens=s.chunk_tokens)
        self.prices = prices or PriceTable()
        self._native_stream = False
        if llm_call is not None:
            self._llm = llm_call
        elif _HAS_LITELLM and os.getenv("AIOTECH_OFFLINE", "0") != "1":
            self._llm = _litellm_factory(s)
            self._native_stream = True
        else:
            self._llm = stub_llm
        self.offline = self._llm is stub_llm
        self._m: Dict[str, float] = {
            "requests": 0, "blocked": 0, "cache_hits": 0, "llm_calls": 0, "errors": 0,
            "tokens_in": 0, "tokens_out": 0, "context_tokens_saved": 0,
            "cost_usd": 0.0, "baseline_cost_usd": 0.0, "latency_ms": 0.0, "fallback_redirects": 0,
            "routed_cheap": 0, "routed_flagship": 0,
        }

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _with_context(messages: List[Dict[str, Any]], selection: Optional[ContextSelection]) -> List[Dict[str, Any]]:
        if not selection or not selection.selected:
            return list(messages)
        block = selection.render()
        msgs = [dict(m) for m in messages]
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = f"{msgs[0].get('content') or ''}\n\n{block}".strip()
        else:
            msgs.insert(0, {"role": "system", "content": block})
        return msgs

    def _pick_model(self, msgs: List[Dict[str, Any]], forced: Optional[str]) -> str:
        model = self.selector.select(msgs, forced)
        if not forced:
            key = "routed_flagship" if model == self.selector.flagship else "routed_cheap"
            self._m[key] += 1
        if self.breaker.is_available(provider_of(model)):
            return model
        for alt in [self.settings.flagship_model, self.settings.cheap_model] + self.settings.fallback_models:
            if alt != model and self.breaker.is_available(provider_of(alt)):
                self._m["fallback_redirects"] += 1
                return alt
        return model  # tous ouverts : on tente quand même, l'échec sera compté

    def _baseline_cost(self, msgs_tokens_full: int, tokens_out: int) -> Optional[float]:
        return self.prices.cost(self.settings.flagship_model, msgs_tokens_full, tokens_out)

    async def _call(self, model: str, msgs: List[Dict[str, Any]], max_tokens: int,
                    temperature: Optional[float]) -> LLMResult:
        provider = provider_of(model)
        self.breaker.acquire(provider)
        self._m["llm_calls"] += 1
        try:
            res = await self._llm(model, msgs, max_tokens, temperature, self.settings.enable_tools)
        except Exception as exc:  # erreurs réseau / fournisseur
            self.breaker.record_failure(provider)
            return LLMResult("", 0, 0, False, error=f"{type(exc).__name__}: {exc}")
        if res.error:
            self.breaker.record_failure(provider)
        else:
            self.breaker.record_success(provider)
        return res

    # ── pipeline principal ────────────────────────────────────────────────────
    async def acompletion(
        self,
        messages: List[Dict[str, Any]],
        documents: Optional[Sequence[DocumentLike]] = None,
        constraints: Optional[Dict[str, Any] | ArgConstraints] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        use_cache: bool = True,
        validate_json: bool = False,
        required_keys: Optional[List[str]] = None,
        schema_hint: str = "",
        verify_chain: bool = False,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        self._m["requests"] += 1
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("Le dernier message doit être un message utilisateur.")

        # 1. garde-fous
        user_raw = str(messages[-1].get("content") or "")
        shield = self.shield.analyze(user_raw)
        if not shield["safe"]:
            self._m["blocked"] += 1
            return self._result(t0, content="", model=None, blocked=True, reason=shield["reason"],
                                success=False)
        msgs = [dict(m) for m in messages]
        msgs[-1]["content"] = shield["cleaned"]
        user = shield["cleaned"]

        # 2. ARG : sélection du contexte
        selection: Optional[ContextSelection] = None
        if documents:
            cons = constraints if isinstance(constraints, ArgConstraints) else ArgConstraints.from_dict(constraints)
            selection = self.budgeter.select(user, documents, cons)
        final_msgs = self._with_context(msgs, selection)
        # Référence de coût : modèle phare, TOUT le contexte candidat, mêmes tokens de sortie.
        ctx_saved = selection.tokens_saved if selection else 0

        # 3. cache (clé = modèle routé + empreinte du contexte complet hors question)
        chosen = self._pick_model(final_msgs, model)
        fp = context_fingerprint([m for m in final_msgs[:-1]])
        if use_cache:
            hit = self.cache.lookup(chosen, user, fp)
            if hit:
                self._m["cache_hits"] += 1
                baseline = self._baseline_cost(_messages_tokens(final_msgs) + ctx_saved, hit["tokens_out"])
                return self._result(t0, content=hit["content"], model=chosen, cache_hit=True,
                                    similarity=hit["similarity"], tokens_in=0, tokens_out=0, cost=0.0,
                                    baseline=baseline, selection=selection, success=True,
                                    routing=self.selector.explain(final_msgs))

        # 4-6. appel LLM (routage + disjoncteur déjà appliqués)
        res = await self._call(chosen, final_msgs, max_tokens, temperature)
        tok_in, tok_out, content = res.tokens_in, res.tokens_out, res.content

        # 7. validation JSON avec relances
        json_report = None
        if validate_json and not res.error:
            for attempt in range(self.validator.max_retries + 1):
                json_report = self.validator.validate_json(content, required_keys)
                if json_report["valid"] or attempt == self.validator.max_retries:
                    break
                retry = final_msgs + [{"role": "assistant", "content": content},
                                      {"role": "user", "content": self.validator.build_retry_prompt(
                                          json_report["error"], schema_hint)}]
                r2 = await self._call(chosen, retry, max_tokens, temperature)
                if r2.error:
                    break
                content, tok_in, tok_out = r2.content, tok_in + r2.tokens_in, tok_out + r2.tokens_out

        # 8. vérification de la chaîne
        chain = None
        if verify_chain and content and not res.error:
            ctx = [s.text for s in selection.selected] if selection else []
            chain = self.budgeter.gate.verify_chain(content, ctx, query=user).as_dict()

        success = bool(content) and not res.error
        cost = self.prices.cost(chosen, tok_in, tok_out)
        baseline = self._baseline_cost(tok_in + ctx_saved, tok_out) if success else None
        if success and use_cache and not (validate_json and json_report and not json_report["valid"]):
            self.cache.store(chosen, user, content, tok_in, tok_out, cost, fp)
        if res.error:
            self._m["errors"] += 1
        return self._result(
            t0, content=content, model=chosen, tokens_in=tok_in, tokens_out=tok_out,
            exact_usage=res.exact_usage, cost=cost, baseline=baseline, selection=selection,
            success=success, error=res.error, json=json_report, chain=chain,
            pii=shield["pii_found"], routing=self.selector.explain(final_msgs),
        )

    def _result(self, t0: float, content: str, model: Optional[str], success: bool,
                tokens_in: int = 0, tokens_out: int = 0, cost: Optional[float] = None,
                baseline: Optional[float] = None, selection: Optional[ContextSelection] = None,
                **extra: Any) -> Dict[str, Any]:
        ms = (time.perf_counter() - t0) * 1e3
        self._m["latency_ms"] += ms
        self._m["tokens_in"] += tokens_in
        self._m["tokens_out"] += tokens_out
        if selection:
            self._m["context_tokens_saved"] += selection.tokens_saved
        if cost is not None:
            self._m["cost_usd"] += cost
        if baseline is not None:
            self._m["baseline_cost_usd"] += baseline
        saved = (baseline - cost) if (baseline is not None and cost is not None) else None
        out = {
            "content": content, "model": model, "success": success,
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "cost_usd": None if cost is None else round(cost, 8),
            "baseline_cost_usd": None if baseline is None else round(baseline, 8),
            "saved_usd": None if saved is None else round(saved, 8),
            "context": selection.as_dict() if selection else None,
            "latency_ms": round(ms, 2), "offline": self.offline,
            "cache_hit": extra.pop("cache_hit", False), "blocked": extra.pop("blocked", False),
        }
        out.update(extra)
        return out

    def completion_sync(self, messages: List[Dict[str, Any]], **kw: Any) -> Dict[str, Any]:
        return asyncio.run(self.acompletion(messages, **kw))

    # ── streaming ─────────────────────────────────────────────────────────────
    async def astream(self, messages: List[Dict[str, Any]], documents: Optional[Sequence[DocumentLike]] = None,
                      constraints: Optional[Dict[str, Any]] = None, model: Optional[str] = None,
                      max_tokens: int = 1024, temperature: Optional[float] = None
                      ) -> AsyncGenerator[Dict[str, Any], None]:
        """Événements : meta -> token* -> done. Passe par les mêmes garde-fous que acompletion."""
        user_raw = str(messages[-1].get("content") or "") if messages else ""
        shield = self.shield.analyze(user_raw)
        if not shield["safe"]:
            self._m["blocked"] += 1
            yield {"type": "blocked", "reason": shield["reason"]}
            return
        msgs = [dict(m) for m in messages]
        msgs[-1]["content"] = shield["cleaned"]
        selection = (self.budgeter.select(shield["cleaned"], documents, ArgConstraints.from_dict(constraints))
                     if documents else None)
        final_msgs = self._with_context(msgs, selection)
        chosen = self._pick_model(final_msgs, model)
        yield {"type": "meta", "model": chosen, "context": selection.as_dict() if selection else None}

        text = ""
        if not self._native_stream:  # stub ou LLM injecté : réponse complète découpée en mots
            res = await self._call(chosen, final_msgs, max_tokens, temperature)
            if res.error:
                yield {"type": "error", "error": res.error}
            for word in res.content.split(" ") if res.content else []:
                text += word + " "
                yield {"type": "token", "text": word + " "}
                await asyncio.sleep(0)
        else:  # pragma: no cover - réseau
            provider = provider_of(chosen)
            self.breaker.acquire(provider)
            try:
                resp = await litellm.acompletion(model=chosen, messages=final_msgs, max_tokens=max_tokens,
                                                 stream=True, **({"temperature": temperature} if temperature is not None else {}))
                async for chunk in resp:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        text += delta
                        yield {"type": "token", "text": delta}
                self.breaker.record_success(provider)
            except Exception as exc:
                self.breaker.record_failure(provider)
                yield {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
        tok_in, tok_out = _messages_tokens(final_msgs), estimate_tokens(text)
        cost = self.prices.cost(chosen, tok_in, tok_out)
        self._m["requests"] += 1
        self._m["tokens_in"] += tok_in
        self._m["tokens_out"] += tok_out
        yield {"type": "done", "tokens_in_est": tok_in, "tokens_out_est": tok_out,
               "cost_usd_est": None if cost is None else round(cost, 8)}

    # ── métriques ─────────────────────────────────────────────────────────────
    @property
    def metrics(self) -> Dict[str, Any]:
        n = max(int(self._m["requests"]), 1)
        base, cost = self._m["baseline_cost_usd"], self._m["cost_usd"]
        return {
            **{k: (round(v, 6) if isinstance(v, float) else int(v)) for k, v in self._m.items()},
            "avg_latency_ms": round(self._m["latency_ms"] / n, 2),
            "saved_usd": round(base - cost, 6),
            "saved_pct": round((base - cost) / base, 4) if base > 0 else 0.0,
            "cache": self.cache.stats,
            "circuit_breaker": self.breaker.all_states(),
            "shield": self.shield.stats,
            "offline": self.offline,
        }

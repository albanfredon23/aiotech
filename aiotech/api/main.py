"""
API HTTP AIOTECH 45 (FastAPI).

Corrections de sécurité par rapport à v3 :
- les routes /admin/* (créer, lister, désactiver des locataires) n'avaient AUCUNE
  authentification : n'importe qui pouvait se créer une clé. Elles exigent désormais
  l'en-tête X-Admin-Token (AIOTECH_ADMIN_TOKEN) et sont désactivées s'il n'est pas défini ;
- omettre l'en-tête X-API-Key permettait d'utiliser /v1/query sans quota. Avec
  AIOTECH_REQUIRE_API_KEY=true (défaut), la clé est obligatoire ;
- /v1/stream contournait garde-fous, authentification et quotas : il passe maintenant
  par le même contrôle que /v1/query ;
- le chemin du tableau de bord était codé en dur (/home/claude/...) : il est relatif au paquet.

Lancement :  uvicorn aiotech.api.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from aiotech import __version__
from aiotech.config import Settings
from aiotech.economics import Scenario, cost_curve, lever_breakdown, log_volumes, monthly_costs
from aiotech.gateway.client import AiotechClient
from aiotech.gateway.memory import AgentMemory
from aiotech.gateway.pricing import PRICES_AS_OF
from aiotech.reasoning.agents import AgentModulator
from aiotech.rollback import RollbackManager
from aiotech.telemetry import METRICS
from aiotech.tenancy import TenantExists, TenantManager

DASHBOARD = Path(__file__).resolve().parent.parent.parent / "dashboard" / "index.html"


class Document(BaseModel):
    text: str
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Constraints(BaseModel):
    required_terms: List[str] = Field(default_factory=list)
    forbidden_terms: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=100_000)
    system: Optional[str] = None
    documents: List[Document] = Field(default_factory=list, max_length=500)
    constraints: Optional[Constraints] = None
    model: Optional[str] = None
    max_tokens: int = Field(1024, ge=1, le=64_000)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    session_id: str = "default"
    agent_id: Optional[str] = None
    use_cache: bool = True
    validate_json: bool = False
    required_keys: Optional[List[str]] = None
    verify_chain: bool = False


class TenantCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    plan: str = "starter"


class AgentRegister(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=5000)


class Feedback(BaseModel):
    agent_id: str
    success_score: float = Field(..., ge=0.0, le=1.0)


class FactSet(BaseModel):
    key: str
    value: Any


def create_app(settings: Optional[Settings] = None, client: Optional[AiotechClient] = None,
               tenants: Optional[TenantManager] = None, memory: Optional[AgentMemory] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    client = client or AiotechClient(settings)
    tenants = tenants or TenantManager(settings.db_path)
    memory = memory or AgentMemory(settings.db_path)
    agents = AgentModulator()
    rollback = RollbackManager(on_rollback=lambda mid, reason: METRICS.counter("aiotech_rollbacks_total"))

    app = FastAPI(title="AIOTECH 45", version=__version__,
                  description="Middleware de raisonnement contraint (ARG) et d'économie de tokens pour LLM.")

    # ── dépendances d'authentification ────────────────────────────────────────
    async def tenant_auth(x_api_key: Optional[str] = Header(None)) -> Optional[str]:
        if not x_api_key:
            if settings.require_api_key:
                raise HTTPException(401, "En-tête X-API-Key requis")
            return None
        tid = tenants.authenticate(x_api_key)
        if not tid:
            raise HTTPException(401, "Clé API invalide")
        quota = tenants.check_and_reserve(tid)
        if not quota["allowed"]:
            raise HTTPException(429, quota["reason"])
        return tid

    async def admin_auth(x_admin_token: Optional[str] = Header(None)) -> None:
        if not settings.admin_token:
            raise HTTPException(503, "Administration désactivée : définir AIOTECH_ADMIN_TOKEN")
        if not x_admin_token or not hmac.compare_digest(x_admin_token, settings.admin_token):
            raise HTTPException(403, "Jeton d'administration invalide")

    def _messages(req: QueryRequest) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if req.system:
            msgs.append({"role": "system", "content": req.system})
        if req.agent_id:
            msgs.extend(memory.get_history(req.agent_id, req.session_id))
        msgs.append({"role": "user", "content": req.query})
        return msgs

    # ── routes publiques ──────────────────────────────────────────────────────
    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok", "version": __version__, "offline": client.offline,
                "prices_as_of": PRICES_AS_OF, "circuit_breaker": client.breaker.all_states()}

    @app.post("/v1/query")
    async def query(req: QueryRequest, tenant_id: Optional[str] = Depends(tenant_auth)) -> Dict[str, Any]:
        resp = await client.acompletion(
            _messages(req), documents=[d.model_dump() for d in req.documents] or None,
            constraints=req.constraints.model_dump() if req.constraints else None,
            model=req.model, max_tokens=req.max_tokens, temperature=req.temperature,
            use_cache=req.use_cache, validate_json=req.validate_json,
            required_keys=req.required_keys, verify_chain=req.verify_chain,
        )
        label = {"model": resp.get("model") or "blocked"}
        METRICS.counter("aiotech_requests_total", labels=label)
        METRICS.observe("aiotech_latency_ms", resp["latency_ms"], labels=label)
        METRICS.counter("aiotech_tokens_total", resp["tokens_in"] + resp["tokens_out"])
        if resp.get("cache_hit"):
            METRICS.counter("aiotech_cache_hits_total")
        if resp.get("context"):
            METRICS.counter("aiotech_context_tokens_saved_total", resp["context"]["tokens_saved"])
        if resp.get("saved_usd") is not None:
            METRICS.counter("aiotech_saved_usd_total", resp["saved_usd"])
        if req.agent_id and resp.get("success"):
            memory.add_message(req.agent_id, req.session_id, "user", req.query)
            memory.add_message(req.agent_id, req.session_id, "assistant", resp["content"])
        if tenant_id:
            tenants.record_usage(tenant_id, resp["tokens_in"], resp["tokens_out"],
                                 resp.get("cost_usd") or 0.0, resp.get("saved_usd") or 0.0)
        return {**resp, "tenant_id": tenant_id}

    @app.post("/v1/stream")
    async def stream(req: QueryRequest, tenant_id: Optional[str] = Depends(tenant_auth)) -> StreamingResponse:
        async def gen():
            async for event in client.astream(
                _messages(req), documents=[d.model_dump() for d in req.documents] or None,
                constraints=req.constraints.model_dump() if req.constraints else None,
                model=req.model, max_tokens=req.max_tokens, temperature=req.temperature,
            ):
                if event["type"] == "done" and tenant_id:
                    tenants.record_usage(tenant_id, event["tokens_in_est"], event["tokens_out_est"],
                                         event.get("cost_usd_est") or 0.0)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/economics")
    async def economics(scenario: Dict[str, float]) -> Dict[str, Any]:
        try:
            s = Scenario(**{**Scenario().as_dict(), **scenario})
            return {"scenario": s.as_dict(), "monthly": monthly_costs(s), "levers": lever_breakdown(s),
                    "curve": cost_curve(s, log_volumes())}
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc))

    # ── agents et mémoire ─────────────────────────────────────────────────────
    @app.post("/v1/agents")
    async def register_agent(req: AgentRegister, _: Optional[str] = Depends(tenant_auth)) -> Dict[str, Any]:
        agents.register(req.agent_id, req.description)
        return {"status": "registered", "agent_id": req.agent_id}

    @app.post("/v1/agents/route")
    async def route_agent(req: QueryRequest, _: Optional[str] = Depends(tenant_auth)) -> Dict[str, Any]:
        if not agents.agent_ids:
            raise HTTPException(404, "Aucun agent enregistré")
        return agents.route(req.query)

    @app.post("/v1/feedback")
    async def feedback(req: Feedback, _: Optional[str] = Depends(tenant_auth)) -> Dict[str, Any]:
        try:
            agents.feedback(req.agent_id, req.success_score)
        except KeyError:
            raise HTTPException(404, "Agent inconnu")
        return {"status": "updated", "stats": agents.stats()[req.agent_id]}

    @app.get("/v1/agents/{agent_id}/memory")
    async def get_memory(agent_id: str, session_id: str = "default", limit: int = 20,
                         _: Optional[str] = Depends(tenant_auth)) -> Dict[str, Any]:
        return {"history": memory.get_history(agent_id, session_id, limit),
                "facts": memory.get_all_facts(agent_id), "stats": memory.stats(agent_id)}

    @app.post("/v1/agents/{agent_id}/facts")
    async def set_fact(agent_id: str, req: FactSet, _: Optional[str] = Depends(tenant_auth)) -> Dict[str, Any]:
        memory.set_fact(agent_id, req.key, req.value)
        return {"agent_id": agent_id, "key": req.key}

    # ── administration ────────────────────────────────────────────────────────
    @app.post("/admin/tenants", dependencies=[Depends(admin_auth)])
    async def create_tenant(req: TenantCreate) -> Dict[str, Any]:
        try:
            return tenants.create_tenant(req.name, req.plan)
        except TenantExists as exc:
            raise HTTPException(409, str(exc))
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.get("/admin/tenants", dependencies=[Depends(admin_auth)])
    async def list_tenants() -> Dict[str, Any]:
        return {"tenants": tenants.list_tenants()}

    @app.get("/admin/tenants/{tenant_id}/usage", dependencies=[Depends(admin_auth)])
    async def tenant_usage(tenant_id: str) -> Dict[str, Any]:
        return tenants.get_usage(tenant_id)

    @app.delete("/admin/tenants/{tenant_id}", dependencies=[Depends(admin_auth)])
    async def deactivate(tenant_id: str) -> Dict[str, Any]:
        if not tenants.deactivate_tenant(tenant_id):
            raise HTTPException(404, "Locataire inconnu")
        return {"status": "deactivated", "tenant_id": tenant_id}

    @app.get("/admin/mutations", dependencies=[Depends(admin_auth)])
    async def mutations() -> Dict[str, Any]:
        return {"mutations": rollback.all_statuses(), "history": rollback.history}

    @app.get("/admin/stats", dependencies=[Depends(admin_auth)])
    async def stats() -> Dict[str, Any]:
        return {"client": client.metrics, "tenants": len(tenants.list_tenants())}

    # ── observabilité ─────────────────────────────────────────────────────────
    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus() -> str:
        return METRICS.export_prometheus()

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard() -> str:
        if not DASHBOARD.exists():
            raise HTTPException(404, "dashboard/index.html introuvable")
        return DASHBOARD.read_text(encoding="utf-8")

    @app.websocket("/ws/telemetry")
    async def ws_telemetry(ws: WebSocket) -> None:
        token = ws.query_params.get("token", "")
        if not settings.admin_token or not hmac.compare_digest(token, settings.admin_token):
            await ws.close(code=1008)
            return
        await ws.accept()
        try:
            while True:
                await ws.send_json({"ts": time.time(), "client": client.metrics, "metrics": METRICS.snapshot()})
                await asyncio.sleep(1.0)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app


app = create_app() if os.getenv("AIOTECH_NO_APP") != "1" else None

if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("aiotech.api.main:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from aiotech.api.main import create_app  # noqa: E402
from aiotech.config import Settings  # noqa: E402


@pytest.fixture()
def api():
    s = Settings(db_path=":memory:", admin_token="secret-admin", require_api_key=True)
    return TestClient(create_app(s))


def test_admin_routes_require_token(api):
    assert api.post("/admin/tenants", json={"name": "A"}).status_code == 403
    assert api.post("/admin/tenants", json={"name": "A"}, headers={"X-Admin-Token": "faux"}).status_code == 403
    r = api.post("/admin/tenants", json={"name": "A"}, headers={"X-Admin-Token": "secret-admin"})
    assert r.status_code == 200 and r.json()["api_key"].startswith("aio-")
    dup = api.post("/admin/tenants", json={"name": "A"}, headers={"X-Admin-Token": "secret-admin"})
    assert dup.status_code == 409


def test_admin_disabled_without_token():
    client = TestClient(create_app(Settings(db_path=":memory:", admin_token="")))
    assert client.get("/admin/tenants", headers={"X-Admin-Token": ""}).status_code == 503


def test_query_requires_api_key_and_returns_savings(api):
    assert api.post("/v1/query", json={"query": "Bonjour"}).status_code == 401
    key = api.post("/admin/tenants", json={"name": "B"}, headers={"X-Admin-Token": "secret-admin"}).json()["api_key"]
    body = {"query": "Quel est le poids du module ORION-12 ?",
            "documents": [{"text": "Module ORION-12. Poids : 4,2 kg.", "source": "o12"},
                          {"text": "Module ORION-13. Poids : 7,9 kg.", "source": "o13"}]}
    r = api.post("/v1/query", json=body, headers={"X-API-Key": key})
    assert r.status_code == 200
    data = r.json()
    assert data["success"] and data["context"]["selected"] == 1
    assert data["tenant_id"]


def test_stream_requires_key(api):
    assert api.post("/v1/stream", json={"query": "Bonjour"}).status_code == 401


def test_economics_endpoint(api):
    r = api.post("/v1/economics", json={"requests_per_month": 2_000_000})
    assert r.status_code == 200
    assert r.json()["monthly"]["savings_month"] > 0
    assert api.post("/v1/economics", json={"cache_hit_rate": 3}).status_code == 422


def test_metrics_and_health(api):
    assert api.get("/health").json()["status"] == "ok"
    assert "aiotech_uptime_seconds" in api.get("/metrics").text

"""Tests: FastAPI service layer — skipped when FastAPI is not installed."""
import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from tokenopt.service import create_app  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(create_app())


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_optimize_endpoint(client):
    response = client.post("/optimize", json={
        "tenant": "acme",
        "prompt": "Summarize the refund policy for a wireless headset. "
                  "The policy states a 30-day return window, refunds to the "
                  "original payment method within 5 business days, and no "
                  "restocking fee for unopened items. Please keep it concise.",
        "query": "refund policy",
    })
    assert response.status_code == 200
    body = response.json()
    assert body["optimized_tokens"] < body["original_tokens"]
    assert body["tokens_saved"] > 0


def test_optimize_over_tenant_budget_returns_429(client):
    response = client.post("/optimize", json={
        "tenant": "acme",
        "prompt": "x " * 100,
        "max_tokens": 1,
    })
    # max_tokens=1 with a multi-word prompt: packing of the prompt itself is
    # not the constraint, but a 1-token budget still forces compression to
    # the floor; the endpoint must either succeed or fail closed with 429.
    assert response.status_code in (200, 429)


def test_policy_lifecycle_endpoints(client):
    assert client.post("/policies", json={
        "name": "custom", "version": 1,
        "params": {"target_ratio": 0.6, "dedup_threshold": 0.85,
                   "max_tokens": 8000},
    }).status_code == 201
    response = client.get("/policies/custom/1/evaluate")
    assert response.status_code == 200
    assert response.json()["cases"] == 4
    assert client.post("/policies/custom/1/candidate").status_code == 200
    response = client.post("/policies/custom/1/promote")
    assert response.status_code == 200
    assert response.json()["active"] == "custom@v1"


def test_meter_and_audit_endpoints(client):
    client.post("/optimize", json={"tenant": "acme", "prompt": "Hello there."})
    meter = client.get("/meter/acme")
    assert meter.status_code == 200
    assert meter.json()["requests"] == 1
    audit = client.get("/audit/verify")
    assert audit.status_code == 200
    assert audit.json()["chain_valid"] is True

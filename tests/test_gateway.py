"""
Tests drive the gateway app directly over ASGI (httpx.ASGITransport) and
swap in a fake httpx.AsyncClient (httpx.MockTransport) for the *outbound*
call to the LLM service, so nothing here touches the network or needs
mock_llm_service.py running. Each test controls exactly how the "LLM" responds.
"""

import httpx
import pytest

from gateway.config import settings
from gateway.llm_client import LLMClient
from gateway.main import app

API_KEY = settings.api_key
HEADERS = {"x-api-key": API_KEY}


def make_transport(handler):
    return httpx.MockTransport(handler)


async def client_with_llm_transport(handler) -> httpx.AsyncClient:
    """Rebuilds app.state.llm_client around a mocked transport, then returns
    an ASGI client for the gateway app itself."""
    fake_llm_http = httpx.AsyncClient(transport=make_transport(handler), base_url="http://llm")
    app.state.http_client = fake_llm_http
    app.state.llm_client = LLMClient(fake_llm_http)
    app.state.idempotency._entries.clear()
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway")


@pytest.mark.asyncio
async def test_happy_path_classifies_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"label": "clinical_urgent", "confidence": 0.95})

    async with app.router.lifespan_context(app):
        client = await client_with_llm_transport(handler)
        resp = await client.post(
            "/classify",
            json={"message_id": "m1", "patient_id": "p1", "text": "I have chest pain"},
            headers=HEADERS,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "clinical_urgent"
    assert body["status"] == "classified"
    assert body["message_id"] == "m1"


@pytest.mark.asyncio
async def test_low_confidence_routes_to_review():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"label": "general", "confidence": 0.3})

    async with app.router.lifespan_context(app):
        client = await client_with_llm_transport(handler)
        resp = await client.post(
            "/classify",
            json={"message_id": "m2", "patient_id": "p1", "text": "hello"},
            headers=HEADERS,
        )

    assert resp.json()["status"] == "low_confidence_review"


@pytest.mark.asyncio
async def test_retries_then_succeeds_on_transient_500():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(500)
        return httpx.Response(200, json={"label": "billing", "confidence": 0.8})

    async with app.router.lifespan_context(app):
        client = await client_with_llm_transport(handler)
        resp = await client.post(
            "/classify",
            json={"message_id": "m3", "patient_id": "p1", "text": "about my bill"},
            headers=HEADERS,
        )

    assert calls["n"] == 2
    assert resp.json()["label"] == "billing"


@pytest.mark.asyncio
async def test_exhausted_retries_fall_back_safely():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async with app.router.lifespan_context(app):
        client = await client_with_llm_transport(handler)
        resp = await client.post(
            "/classify",
            json={"message_id": "m4", "patient_id": "p1", "text": "hello"},
            headers=HEADERS,
        )

    body = resp.json()
    assert resp.status_code == 200  # never a 5xx out to the portal
    assert body["status"] == "fallback_review"
    assert body["label"] is None


@pytest.mark.asyncio
async def test_malformed_llm_response_falls_back():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"label": "clinical_urgent"})  # no confidence

    async with app.router.lifespan_context(app):
        client = await client_with_llm_transport(handler)
        resp = await client.post(
            "/classify",
            json={"message_id": "m5", "patient_id": "p1", "text": "hello"},
            headers=HEADERS,
        )

    assert resp.json()["status"] == "fallback_review"


@pytest.mark.asyncio
async def test_duplicate_message_id_is_idempotent():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"label": "scheduling", "confidence": 0.9})

    async with app.router.lifespan_context(app):
        client = await client_with_llm_transport(handler)
        payload = {"message_id": "m6", "patient_id": "p1", "text": "reschedule please"}
        first = await client.post("/classify", json=payload, headers=HEADERS)
        second = await client.post("/classify", json=payload, headers=HEADERS)

    assert calls["n"] == 1  # second call hit the idempotency cache, not the LLM
    assert first.json() == second.json()


@pytest.mark.asyncio
async def test_missing_api_key_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"label": "general", "confidence": 0.9})

    async with app.router.lifespan_context(app):
        client = await client_with_llm_transport(handler)
        resp = await client.post(
            "/classify",
            json={"message_id": "m7", "patient_id": "p1", "text": "hello"},
        )

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_empty_text_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"label": "general", "confidence": 0.9})

    async with app.router.lifespan_context(app):
        client = await client_with_llm_transport(handler)
        resp = await client.post(
            "/classify",
            json={"message_id": "m8", "patient_id": "p1", "text": ""},
            headers=HEADERS,
        )

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_rate_limit_enforced():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"label": "general", "confidence": 0.9})

    async with app.router.lifespan_context(app):
        client = await client_with_llm_transport(handler)
        app.state.limiter._buckets.clear()
        app.state.limiter._capacity = 2
        app.state.limiter._buckets["dev-local-key"] = (2, __import__("time").monotonic())

        results = []
        for i in range(3):
            resp = await client.post(
                "/classify",
                json={"message_id": f"rl{i}", "patient_id": "p1", "text": "hello"},
                headers=HEADERS,
            )
            results.append(resp.status_code)

    assert results == [200, 200, 429]

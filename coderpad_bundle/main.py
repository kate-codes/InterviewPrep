"""
Patient Message Gateway -- single-file bundle for CoderPad.

Merged from gateway/{config,models,logging_utils,ratelimit,idempotency,
llm_client,main}.py plus tests/test_gateway.py. Internal `from gateway.X
import Y` lines were removed since everything now lives in one module;
third-party imports are unchanged.

Requires: fastapi, httpx, pydantic, pydantic-settings, pytest, pytest-asyncio
(and uvicorn if you want to actually serve it rather than just run the tests).
If CoderPad's pad type doesn't allow `pip install`, this will fail on import --
see the chat message for how to check that before the interview.

Run the embedded tests with:  python main.py
(or, in a terminal-enabled pad:  pytest main.py -v)
"""

import asyncio
import hashlib
import json
import logging
import random
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import httpx
import pytest
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ============================================================
# ---- config.py ----
# ============================================================


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GATEWAY_")

    llm_service_url: str = "http://localhost:9000"
    llm_timeout_seconds: float = 2.0
    llm_max_attempts: int = 3
    llm_backoff_base_seconds: float = 0.2

    # Open the circuit after this many consecutive LLM failures; stay open
    # (short-circuit straight to fallback) for cooldown_seconds before
    # trying the LLM again.
    circuit_failure_threshold: int = 5
    circuit_cooldown_seconds: float = 30.0

    low_confidence_threshold: float = 0.6

    api_key: str = "dev-local-key"
    rate_limit_per_minute: int = 60
    # So each API key gets a bucket of 60 tokens, refilling at 1 token/sec
    # i.e. a burst of up to 60 requests, sustained at 60/minute thereafter.
    # It's keyed per x-api-key value, so right now (single dev key).
    # It's effectively one global limit.

    idempotency_ttl_seconds: float = 300.0

    max_text_length: int = 4000


settings = Settings()


# ============================================================
# ---- models.py ----
# ============================================================

Label = Literal["clinical_urgent", "clinical_routine", "billing", "scheduling", "general"]

# What we show the caller. Deliberately has no "error" value: a patient
# message should never just vanish into a 500. If we can't get a confident
# answer from the LLM, we say so and route to a human instead of failing.
Status = Literal["classified", "low_confidence_review", "fallback_review"]


class IncomingMessage(BaseModel):
    message_id: str = Field(min_length=1, max_length=128)
    patient_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def text_within_limit(cls, v: str) -> str:
        if len(v) > settings.max_text_length:
            raise ValueError(f"text exceeds {settings.max_text_length} character limit")
        return v


class LLMClassification(BaseModel):
    """Shape we require the LLM service's response to match. Anything that
    doesn't validate against this (missing field, wrong type, unknown label)
    is treated as a failed call, same as a timeout or 500 -- the LLM is a
    black box and we don't trust its output any more than its uptime."""

    label: Label
    confidence: float = Field(ge=0.0, le=1.0)


class GatewayResponse(BaseModel):
    message_id: str
    # None only when status == "fallback_review": we genuinely don't have an
    # opinion from the LLM and would rather say that than guess a label.
    label: Label | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    status: Status
    latency_ms: float
    request_id: str


# ============================================================
# ---- logging_utils.py ----
# ============================================================

logger = logging.getLogger("gateway")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Anything passed via logger.info(..., extra={...}) rides along here.
        for key, value in getattr(record, "fields", {}).items():
            payload[key] = value
        return json.dumps(payload)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False


def text_fingerprint(text: str) -> str:
    """Stand-in for the raw message body in logs. Lets us correlate 'same
    input produced different classifications across two calls' during
    debugging without ever writing patient-authored text to a log stream."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def log_event(message: str, **fields) -> None:
    logger.info(message, extra={"fields": fields})


# ============================================================
# ---- ratelimit.py ----
# ============================================================


class TokenBucketLimiter:
    """Per-key token bucket. In-memory, so it's per-process -- fine for a
    single CoderPad instance / this exercise, but the honest answer in the
    interview is "this needs to move to Redis (INCR + TTL, or a Lua script)
    the moment there's more than one gateway replica, since each replica
    would otherwise get its own independent budget."
    """

    def __init__(self, capacity: int, refill_per_second: float):
        self._capacity = capacity
        self._refill_per_second = refill_per_second
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_refill)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        tokens, last_refill = self._buckets.get(key, (self._capacity, now))

        elapsed = now - last_refill
        tokens = min(self._capacity, tokens + elapsed * self._refill_per_second)

        if tokens < 1:
            self._buckets[key] = (tokens, now)
            return False

        self._buckets[key] = (tokens - 1, now)
        return True


# ============================================================
# ---- idempotency.py ----
# ============================================================


class IdempotencyCache:
    """Patient-messaging portals retry on timeout; without this, a retried
    submit re-runs the LLM call and can -- on a flaky classifier -- produce a
    different label for the same message_id. TTL-bounded in-memory cache is
    fine for the exercise; production would be Redis so it survives restarts
    and is shared across replicas."""

    def __init__(self, ttl_seconds: float):
        self._ttl = ttl_seconds
        self._entries: dict[str, tuple[float, GatewayResponse]] = {}

    def get(self, message_id: str) -> GatewayResponse | None:
        entry = self._entries.get(message_id)
        if entry is None:
            return None
        stored_at, response = entry
        if time.monotonic() - stored_at > self._ttl:
            del self._entries[message_id]
            return None
        return response

    def set(self, message_id: str, response: GatewayResponse) -> None:
        self._entries[message_id] = (time.monotonic(), response)


# ============================================================
# ---- llm_client.py ----
# ============================================================


class LLMUnavailableError(Exception):
    """Raised once retries are exhausted (or the circuit is open). Callers
    should treat this as "we don't know the answer", not as a 500 -- the
    message still needs a response, just not a confident one."""


class LLMClient:
    """Thin wrapper around the classification service with timeouts,
    bounded retries with jittered backoff, and a simple circuit breaker.

    The circuit breaker exists because if the LLM service is genuinely down,
    retrying every single request for timeout_seconds * max_attempts just
    piles up latency and load on a service that's already struggling. After
    enough consecutive failures we stop calling it for a cooldown window and
    fail straight to the fallback path instead.
    """

    def __init__(self, client: httpx.AsyncClient):
        self._client = client
        self._consecutive_failures = 0
        self._circuit_opened_at: float | None = None

    @property
    def circuit_open(self) -> bool:
        if self._circuit_opened_at is None:
            return False
        if time.monotonic() - self._circuit_opened_at >= settings.circuit_cooldown_seconds:
            return False  # cooldown elapsed; let the next call through as a trial
        return True

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._circuit_opened_at = None

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= settings.circuit_failure_threshold:
            self._circuit_opened_at = time.monotonic()

    async def classify(self, text: str) -> LLMClassification:
        if self.circuit_open:
            raise LLMUnavailableError("circuit open: LLM service recently failing repeatedly")

        last_error: Exception | None = None

        for attempt in range(1, settings.llm_max_attempts + 1):
            try:
                response = await self._client.post(
                    "/classify",
                    json={"text": text},
                    timeout=settings.llm_timeout_seconds,
                )

                if response.status_code >= 500:
                    # Transient -- worth retrying.
                    raise httpx.HTTPStatusError(
                        "server error", request=response.request, response=response
                    )
                if response.status_code >= 400:
                    # Our request was malformed in some way a retry won't fix.
                    self._record_failure()
                    raise LLMUnavailableError(f"LLM service rejected request: {response.status_code}")

                result = LLMClassification.model_validate(response.json())
                self._record_success()
                return result

            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                self._record_failure()
            except (ValidationError, ValueError) as exc:
                # Response came back 2xx but didn't match the schema we
                # require. Not a network problem, but we still don't trust
                # it, and retrying occasionally helps if it's a transient
                # generation glitch rather than a contract break.
                last_error = exc
                self._record_failure()

            if attempt < settings.llm_max_attempts:
                backoff = settings.llm_backoff_base_seconds * (2 ** (attempt - 1))
                jitter = random.uniform(0, settings.llm_backoff_base_seconds)
                await asyncio.sleep(backoff + jitter)

        raise LLMUnavailableError(f"LLM call failed after {settings.llm_max_attempts} attempts") from last_error


# ============================================================
# ---- main.py ----
# ============================================================

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(base_url=settings.llm_service_url)
    app.state.llm_client = LLMClient(app.state.http_client)
    app.state.limiter = TokenBucketLimiter(
        capacity=settings.rate_limit_per_minute,
        refill_per_second=settings.rate_limit_per_minute / 60.0,
    )
    app.state.idempotency = IdempotencyCache(ttl_seconds=settings.idempotency_ttl_seconds)
    app.state.metrics = {"requests_total": 0, "llm_failures_total": 0, "fallback_total": 0}
    try:
        yield
    finally:
        await app.state.http_client.aclose()


app = FastAPI(title="Patient Message Gateway", lifespan=lifespan)


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.post("/classify", response_model=GatewayResponse, dependencies=[Depends(verify_api_key)])
async def classify(message: IncomingMessage, request: Request, x_api_key: str | None = Header(default=None)):
    request_id = request.state.request_id
    app.state.metrics["requests_total"] += 1

    if not app.state.limiter.allow(x_api_key):
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    cached = app.state.idempotency.get(message.message_id)
    if cached is not None:
        log_event("idempotent_hit", message_id=message.message_id, request_id=request_id)
        return cached

    start = time.monotonic()
    try:
        result = await app.state.llm_client.classify(message.text)
        status = "classified" if result.confidence >= settings.low_confidence_threshold else "low_confidence_review"
        gateway_response = GatewayResponse(
            message_id=message.message_id,
            label=result.label,
            confidence=result.confidence,
            status=status,
            latency_ms=round((time.monotonic() - start) * 1000, 1),
            request_id=request_id,
        )
    except LLMUnavailableError as exc:
        app.state.metrics["llm_failures_total"] += 1
        app.state.metrics["fallback_total"] += 1
        gateway_response = GatewayResponse(
            message_id=message.message_id,
            label=None,
            confidence=None,
            status="fallback_review",
            latency_ms=round((time.monotonic() - start) * 1000, 1),
            request_id=request_id,
        )
        log_event(
            "llm_fallback",
            message_id=message.message_id,
            request_id=request_id,
            text_fingerprint=text_fingerprint(message.text),
            text_length=len(message.text),
            error=str(exc),
        )

    app.state.idempotency.set(message.message_id, gateway_response)

    log_event(
        "classification_complete",
        message_id=message.message_id,
        request_id=request_id,
        status=gateway_response.status,
        label=gateway_response.label,
        confidence=gateway_response.confidence,
        latency_ms=gateway_response.latency_ms,
        text_fingerprint=text_fingerprint(message.text),
        text_length=len(message.text),
    )

    return gateway_response


@app.get("/healthz")
async def healthz():
    circuit_open = app.state.llm_client.circuit_open
    return JSONResponse(
        status_code=503 if circuit_open else 200,
        content={"status": "degraded" if circuit_open else "ok", "llm_circuit_open": circuit_open},
    )


@app.get("/metrics")
async def metrics():
    return app.state.metrics


# ============================================================
# ---- tests/test_gateway.py ----
# ============================================================
#
# Tests drive the gateway app directly over ASGI (httpx.ASGITransport) and
# swap in a fake httpx.AsyncClient (httpx.MockTransport) for the *outbound*
# call to the LLM service, so nothing here touches the network or needs a
# separate mock LLM service running. Each test controls exactly how the
# "LLM" responds.

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
        app.state.limiter._buckets["dev-local-key"] = (2, time.monotonic())

        results = []
        for i in range(3):
            resp = await client.post(
                "/classify",
                json={"message_id": f"rl{i}", "patient_id": "p1", "text": "hello"},
                headers=HEADERS,
            )
            results.append(resp.status_code)

    assert results == [200, 200, 429]


if __name__ == "__main__":
    # `python main.py` runs the embedded pytest suite directly --
    # handy on a pad with no separate terminal/pytest invocation available.
    raise SystemExit(pytest.main([__file__, "-v"]))

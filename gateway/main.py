import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from gateway.config import settings
from gateway.idempotency import IdempotencyCache
from gateway.llm_client import LLMClient, LLMUnavailableError
from gateway.logging_utils import configure_logging, log_event, text_fingerprint
from gateway.models import GatewayResponse, IncomingMessage
from gateway.ratelimit import TokenBucketLimiter

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

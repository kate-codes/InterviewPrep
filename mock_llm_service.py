"""
Stand-in for the "LLM classification service" the real gateway talks to.

Not part of the interview deliverable — this just gives the gateway something
real to call so the whole thing runs end-to-end on a laptop. It can be told to
misbehave (slow responses, 500s, malformed payloads) via headers, so the
gateway's resilience code has something real to exercise.

Run it with:  .venv\\Scripts\\python -m uvicorn mock_llm_service:app --port 9000
"""

import asyncio
import random

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Mock LLM Classification Service")

LABELS = ["clinical_urgent", "clinical_routine", "billing", "scheduling", "general"]

URGENT_KEYWORDS = ("chest pain", "can't breathe", "bleeding", "suicide", "overdose")


class ClassifyRequest(BaseModel):
    text: str


@app.post("/classify")
async def classify(
    req: ClassifyRequest,
    request: Request,
    x_mock_fail_rate: float = Header(default=0.0),
    x_mock_delay_seconds: float = Header(default=0.0),
    x_mock_malformed_rate: float = Header(default=0.0),
):
    """
    Headers let tests/clients dial in misbehavior without touching this file:
      X-Mock-Fail-Rate: 0.0-1.0      -> probability of a 500
      X-Mock-Delay-Seconds: float    -> artificial latency (to trigger client timeouts)
      X-Mock-Malformed-Rate: 0.0-1.0 -> probability of returning a schema-breaking body
    """
    if x_mock_delay_seconds:
        await asyncio.sleep(x_mock_delay_seconds)

    if x_mock_fail_rate and random.random() < x_mock_fail_rate:
        return JSONResponse(status_code=500, content={"error": "classifier overloaded"})

    if x_mock_malformed_rate and random.random() < x_mock_malformed_rate:
        # Missing "confidence", wrong type, etc. Exercises the gateway's
        # response-schema validation / fallback path.
        return JSONResponse(status_code=200, content={"label": "clinical_urgent"})

    text_lower = req.text.lower()
    if any(kw in text_lower for kw in URGENT_KEYWORDS):
        label, confidence = "clinical_urgent", round(random.uniform(0.9, 0.99), 2)
    elif "bill" in text_lower or "invoice" in text_lower or "charge" in text_lower:
        label, confidence = "billing", round(random.uniform(0.75, 0.95), 2)
    elif "appointment" in text_lower or "reschedul" in text_lower:
        label, confidence = "scheduling", round(random.uniform(0.75, 0.95), 2)
    else:
        label, confidence = random.choice(LABELS[1:]), round(random.uniform(0.4, 0.85), 2)

    return {"label": label, "confidence": confidence}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

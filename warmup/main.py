"""
Warm-up version: the "get the basic flow working" step.

Patient-messaging portal -> this gateway -> LLM classification service -> back
to the portal. No retries, no auth, no validation beyond basic types. This is
roughly what ~15 minutes of a cold start looks like, before the "harden it"
phase of the interview begins.

Run:    .venv\\Scripts\\python -m uvicorn warmup.main:app --port 8000
Then:   curl -X POST localhost:8000/classify -H "Content-Type: application/json" \
          -d "{\"message_id\": \"m1\", \"patient_id\": \"p1\", \"text\": \"I have chest pain\"}"
"""

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

LLM_SERVICE_URL = "http://localhost:9000"

app = FastAPI(title="Patient Message Gateway (warm-up)")


class IncomingMessage(BaseModel):
    message_id: str
    patient_id: str
    text: str


@app.post("/classify")
def classify(message: IncomingMessage):
    response = httpx.post(f"{LLM_SERVICE_URL}/classify", json={"text": message.text})
    response.raise_for_status()
    result = response.json()

    return {
        "message_id": message.message_id,
        "label": result["label"],
        "confidence": result["confidence"],
    }

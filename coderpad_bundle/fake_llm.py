"""
Fake LLM classification service for demoing the gateway live.

Listens on :9000 and answers the same /classify contract the gateway
expects (see LLMClassification in main.py): {"text": str} in,
{"label": ..., "confidence": ...} out. Classification is just keyword
matching -- this exists to make the gateway's "classified" path visible
during a demo, not to be a real model.

Run:  python fake_llm.py
Then, in another terminal:  python -m uvicorn main:app --port 8000
(the gateway's default GATEWAY_LLM_SERVICE_URL is http://localhost:9000)
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Fake LLM")


class ClassifyRequest(BaseModel):
    text: str


KEYWORDS = [
    (("chest pain", "can't breathe", "bleeding", "emergency"), "clinical_urgent", 0.95),
    (("bill", "invoice", "charge", "payment"), "billing", 0.9),
    (("reschedule", "appointment", "book", "cancel"), "scheduling", 0.88),
    (("prescription", "refill", "symptom", "medication"), "clinical_routine", 0.85),
]


@app.post("/classify")
async def classify(request: ClassifyRequest):
    lowered = request.text.lower()
    for terms, label, confidence in KEYWORDS:
        if any(term in lowered for term in terms):
            return {"label": label, "confidence": confidence}
    return {"label": "general", "confidence": 0.5}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=9000)

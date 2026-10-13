from typing import Literal

from pydantic import BaseModel, Field, field_validator

from gateway.config import settings

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
    is treated as a failed call, same as a timeout or 500 — the LLM is a
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

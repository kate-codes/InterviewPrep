import asyncio
import random
import time

import httpx
from pydantic import ValidationError

from gateway.config import settings
from gateway.models import LLMClassification


class LLMUnavailableError(Exception):
    """Raised once retries are exhausted (or the circuit is open). Callers
    should treat this as "we don't know the answer", not as a 500 — the
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
                    # Transient — worth retrying.
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

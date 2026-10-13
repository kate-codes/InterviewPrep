import time

from gateway.models import GatewayResponse


class IdempotencyCache:
    """Patient-messaging portals retry on timeout; without this, a retried
    submit re-runs the LLM call and can — on a flaky classifier — produce a
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

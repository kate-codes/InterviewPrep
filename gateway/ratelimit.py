import time


class TokenBucketLimiter:
    """Per-key token bucket. In-memory, so it's per-process — fine for a
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

# Interview prep — patient-message classification gateway

## Layout

- `mock_llm_service.py` — stands in for "the LLM classification service."
  Returns `{label, confidence}`, with headers to inject delay / 500s /
  malformed responses so resilience code has something real to react to.
- `warmup/main.py` — the ~15-minute cold-start version: one endpoint, one
  synchronous call to the LLM service, no hardening. This is roughly the
  warm-up task.
- `gateway/` — the hardened version, split into:
  - `config.py` — settings (timeouts, thresholds, rate limit) via `pydantic-settings`
  - `models.py` — request/response schemas, including validating the LLM's
    own response shape
  - `llm_client.py` — timeout + retry/backoff + circuit breaker around the
    LLM call
  - `ratelimit.py`, `idempotency.py` — small in-memory primitives
  - `logging_utils.py` — structured logs that never contain raw patient text
  - `main.py` — wires it together: auth, rate limiting, idempotency,
    request IDs, `/healthz`, `/metrics`
- `tests/test_gateway.py` — exercises every hardening behavior against a
  mocked LLM transport (no network, no real server needed)

## What / Why Idempotency?

Idempotency is a design property where executing an operation multiple times produces the exact same result and side effects as running it once. In Python development, this concept is crucial for making APIs, data pipelines, and system automation resilient to network timeouts, retries, and duplicate event deliveries.

## Run it

```
.venv\Scripts\python -m uvicorn mock_llm_service:app --port 9000
.venv\Scripts\python -m uvicorn gateway.main:app --port 8000      # hardened
.venv\Scripts\python -m uvicorn warmup.main:app --port 8001       # warm-up

curl -X POST localhost:8000/classify -H "Content-Type: application/json" -H "x-api-key: dev-local-key" -d "{\"message_id\":\"m1\",\"patient_id\":\"p1\",\"text\":\"I have chest pain\"}"
```

Tests: `.venv\Scripts\python -m pytest -v`

### Priority 1 — things that would actually break in production

- **Timeouts on the outbound call.** An LLM call with no timeout means one
  slow request can hang a worker indefinitely. Say this out loud even if I
  don't get further.
- **Retries with backoff for transient failures** (timeouts, 5xx)
  *not* for 4xx (retrying a malformed request just repeats the same failure).
  Jitter matters so retries don't synchronize into a thundering herd.
- **Never let a classification failure become a 500 to the portal.** A
  patient's message should always get *some* response. IRL we should require either a real
  classification or a clear "couldn't classify, needs human review" signal.
  Silently dropping or 500-ing a patient message is the actual worst-case
  failure mode in this domain, worse than a wrong classification.
- **Validate the LLM's response, not just the caller's request.** Treat the
  LLM as a black box I don't fully trust because it can return malformed JSON, a
  label outside your known set, or a confidence out of range. Schema-validate
  it and fall back exactly like a timeout.

### Priority 2 — protecting the system under load / partial failure

- **Circuit breaker.** If the LLM service is actually down, retrying every
  request just burns latency. Stop calling it for a cooldown window once
  failures cluster.
- **Rate limiting.** Protects the (presumably more expensive/slower) LLM
  service from the portal. Mention that in-memory only works per-process. For a
  real deployment needs Redis or similar if there's more than one replica.
- **Idempotency on `message_id`.** Portals retry on timeout; without this, a
  retried submit can re-run the LLM and get a *different* label for the same
  message on a non-deterministic classifier.

### Priority 3 — this is healthcare data, so say it explicitly

- **Never log raw message text.** It's PHI. Log a hash/fingerprint + length
  for correlation instead. This is a very cheap thing to mention and
  demonstrates the right instinct even before writing the code.
- **Auth between portal and gateway** (API key / mTLS / signed request —
  whatever's appropriate) — the gateway shouldn't accept unauthenticated
  patient data forwarding.
- **Low-confidence routing.** A confident wrong answer and a hedged "not
  sure" are very different outcomes when the thing being triaged might be
  "chest pain" vs. "reschedule my appointment." Threshold on confidence and
  route low-confidence results to human review rather than acting on them
  silently.
- **Prompt-injection awareness**, if the "LLM" is doing anything beyond
  strict classification (e.g., if it's asked to also summarize or extract
  structured fields): a patient could write a message designed to manipulate
  the model's behavior. Keeping the LLM's role narrow (classify into a fixed
  label set) and validating its output against that fixed set is itself a
  mitigation — worth saying explicitly.

### Lower priority but good to mention if time allows

- Structured/JSON logging with request IDs for tracing across
  portal → gateway → LLM.
- `/healthz` reflecting real downstream state (e.g., circuit-breaker status),
  not just "process is up."
- Basic `/metrics` (even a plain counter dict) I'd wire
  to Prometheus/Datadog in a real deployment.
- Async (`httpx.AsyncClient`) instead of sync calls, so one slow
  classification doesn't block the whole worker.

## Likely interviewer follow-ups — have an answer ready

- *"What happens if the LLM service is down for 10 minutes?"* → circuit
  breaker opens, gateway fails fast to `fallback_review` instead of piling up
  slow requests; `/healthz` reflects degraded state.
- *"How would this change with multiple gateway replicas?"* → rate limiter
  and idempotency cache both need to move to shared storage (Redis); circuit
  breaker state technically should too, though per-replica circuit state is
  a defensible simplification to mention and move on from.
- *"How do I know the LLM's classification is actually correct?"* → I
  don't, fully — mention confidence thresholding, human-review routing for
  low confidence, and an evaluation/feedback loop (sampling
  classified messages for human audit a.k.a. HITL) as the real answer to drift/accuracy
  over time. That's a monitoring problem, not something the gateway alone
  solves.
- *"Why fail to `needs_review` instead of retrying forever / blocking?"* →
  clinical safety: a slow but "safe" response (hedge to a human) beats either
  blocking a patient message indefinitely or guessing.
- *"Why not just increase the timeout instead of retrying?"* → distinguishes
  transient blips from genuine outages; a single long timeout can't do that,
  and it makes every request pay the worst-case latency.

## If the CoderPad template differs from this

I don't know their exact starter (framework, sync vs async, provided
client for the LLM call). 

The *patterns* here: timeout, bounded retry with
backoff, circuit breaker, schema-validate the LLM response, fail safe not
silent, PHI-safe logging, idempotency, rate limiting, confidence thresholds, etc. port directly to Flask, raw `requests`, or anything else. Don't force this
exact file layout onto a different template; re-derive it inline if needed.

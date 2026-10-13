from pydantic_settings import BaseSettings, SettingsConfigDict


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

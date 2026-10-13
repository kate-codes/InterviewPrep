import hashlib
import json
import logging
import sys

logger = logging.getLogger("gateway")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Anything passed via logger.info(..., extra={...}) rides along here.
        for key, value in getattr(record, "fields", {}).items():
            payload[key] = value
        return json.dumps(payload)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False


def text_fingerprint(text: str) -> str:
    """Stand-in for the raw message body in logs. Lets us correlate 'same
    input produced different classifications across two calls' during
    debugging without ever writing patient-authored text to a log stream."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def log_event(message: str, **fields) -> None:
    logger.info(message, extra={"fields": fields})

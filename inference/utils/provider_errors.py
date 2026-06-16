"""Helpers for classifying model-provider failures."""

from __future__ import annotations

import re
from typing import Optional


_NON_RETRYABLE_PATTERNS = (
    "invalid schema",
    "schema for function",
    "invalid parameter",
    "bad request",
    "error code: 400",
)

_RETRYABLE_PATTERNS = (
    "service temporarily unavailable",
    "temporarily unavailable",
    "rate limit",
    "rate_limit",
    "timeout",
    "timed out",
    "connection error",
    "api connection",
    "empty response",
    "no messages returned",
    "server error",
    "internal server error",
    "bad gateway",
    "gateway timeout",
)

_RETRYABLE_STATUS_RE = re.compile(r"(?:error code|status code|status)[^\d]*(429|500|502|503|504)\b", re.I)


def classify_provider_error(message: object) -> Optional[str]:
    """Return a stable provider failure type for known provider errors."""
    text = str(message or "").strip()
    if not text:
        return None
    lowered = text.lower()

    if any(pattern in lowered for pattern in _NON_RETRYABLE_PATTERNS):
        return "provider_schema_error"

    if _RETRYABLE_STATUS_RE.search(text) or any(
        pattern in lowered for pattern in _RETRYABLE_PATTERNS
    ):
        return "provider_transient_error"

    return None


def is_retryable_provider_error(message: object) -> bool:
    return classify_provider_error(message) == "provider_transient_error"


def is_provider_schema_error(message: object) -> bool:
    return classify_provider_error(message) == "provider_schema_error"

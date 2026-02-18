from __future__ import annotations

from typing import Any, Dict


def extract_token_usage(response: Any) -> Dict[str, int]:
    """Best-effort extraction of token usage from a CAMEL/OpenAI-style response.

    Returns a normalized dict with:
      - input_tokens
      - output_tokens
      - total_tokens
    """
    usage_info: Dict[str, Any] = {}

    if hasattr(response, "info") and isinstance(getattr(response, "info"), dict):
        usage_info = (response.info or {}).get("usage", {}) or {}

    if not usage_info and hasattr(response, "metadata") and isinstance(getattr(response, "metadata"), dict):
        usage_info = (response.metadata or {}).get("usage", {}) or {}

    input_tokens = usage_info.get("prompt_tokens", 0) or usage_info.get("input_tokens", 0) or 0
    output_tokens = usage_info.get("completion_tokens", 0) or usage_info.get("output_tokens", 0) or 0
    total_tokens = usage_info.get("total_tokens")
    if total_tokens is None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


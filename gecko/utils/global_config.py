"""Global configuration manager for Gecko model settings."""

from typing import Optional

_state_model: Optional[str] = None
_response_model: Optional[str] = None
_validation_model: Optional[str] = None


def _normalize_model_name(value: Optional[str]) -> Optional[str]:
    """Normalize CLI/env model names."""
    if value is None:
        return None
    if not isinstance(value, str):
        return str(value)
    normalized = value.strip()
    if not normalized or normalized.lower() in {"none", "null"}:
        return None
    return normalized


def set_model_config(
    state_model: str = "gpt-5-mini",
    response_model: str = "gpt-5-mini",
    validation_model: str = "gpt-5-mini",
) -> None:
    """Set global model configuration for Gecko."""
    global _state_model, _response_model, _validation_model
    _state_model = _normalize_model_name(state_model)
    _response_model = _normalize_model_name(response_model) or "gpt-5-mini"
    _validation_model = _normalize_model_name(validation_model) or "gpt-5-mini"


def get_state_model() -> str:
    """Get the state update model name."""
    import os

    env = _normalize_model_name(os.getenv("STATE_MODEL"))
    return _state_model or env or "gpt-5-mini"


def get_response_model() -> str:
    """Get the response model name."""
    return _response_model or "gpt-5-mini"


def get_validation_model() -> str:
    """Get the validation model name."""
    return _validation_model or "gpt-5-mini"


def reset_model_config() -> None:
    """Reset global model configuration."""
    global _state_model, _response_model, _validation_model
    _state_model = None
    _response_model = None
    _validation_model = None

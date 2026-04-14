from typing import Optional

_state_model: Optional[str] = None
_response_model: Optional[str] = None
_validation_model: Optional[str] = None
_response_timeout: float = 300.0
_agent_timeout: float = 360.0
_STATE_MODEL_DISABLED = "__STATE_MODEL_DISABLED__"


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
    state_model: str = "gpt-4.1-mini",
    response_model: str = "gpt-4.1-mini",
    validation_model: str = "gpt-4.1-mini",
) -> None:
    """Set global model configuration for Gecko."""
    global _state_model, _response_model, _validation_model
    normalized_state_model = _normalize_model_name(state_model)
    if normalized_state_model is None and isinstance(state_model, str) and state_model.strip().lower() in {"none", "null", ""}:
        _state_model = _STATE_MODEL_DISABLED
    else:
        _state_model = normalized_state_model
    _response_model = _normalize_model_name(response_model) or "gpt-4.1-mini"
    _validation_model = _normalize_model_name(validation_model) or "gpt-4.1-mini"


def get_state_model() -> Optional[str]:
    """Get the state update model name.

    Returns None when state updater is intentionally disabled
    (e.g., STATE_MODEL=none/null).
    """
    import os

    env = _normalize_model_name(os.getenv("STATE_MODEL"))
    if _state_model == _STATE_MODEL_DISABLED:
        return None
    if _state_model is not None:
        return _state_model
    return env or "gpt-4.1-mini"


def get_response_model() -> str:
    """Get the response model name."""
    return _response_model or "gpt-4.1-mini"


def get_validation_model() -> str:
    """Get the validation model name."""
    return _validation_model or "gpt-4.1-mini"


def set_response_timeout(timeout: float) -> None:
    """Set global response timeout in seconds."""
    global _response_timeout
    _response_timeout = timeout


def get_response_timeout() -> float:
    """Get global response timeout in seconds."""
    return _response_timeout


def set_agent_timeout(timeout: float) -> None:
    """Set global agent step timeout in seconds (used by internal LLM agents)."""
    global _agent_timeout
    _agent_timeout = timeout


def get_agent_timeout() -> float:
    """Get global agent step timeout in seconds."""
    return _agent_timeout


def reset_model_config() -> None:
    """Reset global model configuration."""
    global _state_model, _response_model, _validation_model, _response_timeout, _agent_timeout
    _state_model = None
    _response_model = None
    _validation_model = None
    _response_timeout = 300.0
    _agent_timeout = 360.0

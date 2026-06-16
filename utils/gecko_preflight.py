"""Preflight checks for benchmark runners that require Gecko."""

from __future__ import annotations

from typing import Any

import requests


class GeckoPreflightError(RuntimeError):
    """Raised when Gecko is unavailable or undersized for a run."""


def require_gecko_preflight(
    base_url: str = "http://localhost:8000",
    *,
    min_workers: int,
    require_state_model_disabled: bool = False,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Verify that Gecko is alive and has the required runtime settings.

    Benchmark workers can issue concurrent mock-tool requests. Starting a run
    when Gecko is down, or when Gecko has fewer worker processes than the
    runner, usually creates invalid benchmark artifacts rather than useful
    partial results.

    Some benchmark paths intentionally require ``--state-model none`` so Gecko
    does not run LLM state updates or context extraction. Those callers can set
    ``require_state_model_disabled`` to fail fast when the server was started
    with a state model.
    """

    normalized_url = base_url.rstrip("/")
    health_url = f"{normalized_url}/health"
    try:
        response = requests.get(health_url, timeout=timeout)
        response.raise_for_status()
        health = response.json()
    except Exception as exc:  # pragma: no cover - exercised by runner preflight
        raise GeckoPreflightError(
            f"Gecko is not alive at {health_url}. Start Gecko before running "
            "benchmarks that use GATS."
        ) from exc

    if health.get("status") != "ok" or health.get("service") != "gecko":
        raise GeckoPreflightError(
            f"Gecko health check at {health_url} returned an unexpected payload: {health!r}"
        )

    config = health.get("config") or {}
    raw_workers = config.get("workers")
    if raw_workers is None:
        raise GeckoPreflightError(
            "Gecko /health does not report worker count. Restart Gecko with the "
            "current server code before running this benchmark."
        )
    try:
        gecko_workers = int(raw_workers)
    except (TypeError, ValueError) as exc:
        raise GeckoPreflightError(
            f"Gecko /health reported invalid worker count: {raw_workers!r}"
        ) from exc

    if gecko_workers < min_workers:
        raise GeckoPreflightError(
            f"Gecko workers ({gecko_workers}) are fewer than benchmark workers "
            f"({min_workers}). Restart Gecko with --workers {min_workers} or higher, "
            "or lower the benchmark --workers value."
        )

    if require_state_model_disabled:
        state_model_enabled = config.get("state_model_enabled")
        if state_model_enabled is not False:
            raise GeckoPreflightError(
                "This benchmark requires Gecko to run with --state-model none so "
                "LLM state updates and context extraction are disabled. "
                f"/health reports state_model={config.get('state_model')!r}, "
                f"state_model_enabled={state_model_enabled!r}. Restart Gecko with "
                "--state-model none."
            )

    return health

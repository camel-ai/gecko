"""Legacy compatibility layer for inference.engine.

Historically this module depended on the in-repo Mock Server package. After
Mock Server was split into the standalone Gecko repository, callers should use
`inference.client_engine` / `inference.universal` instead.
"""

from __future__ import annotations

import warnings

from .client_engine import (  # re-export for backward compatibility
    InferenceConfig,
    TurnResult,
    InferenceResult,
    ClientInferenceEngine,
    get_client_inference_engine,
)

warnings.warn(
    "inference.engine is deprecated. Use inference.client_engine or "
    "inference.universal instead.",
    DeprecationWarning,
    stacklevel=2,
)


class InferenceEngine(ClientInferenceEngine):
    """Backward-compatible alias of ClientInferenceEngine."""


def get_inference_engine(config: InferenceConfig) -> InferenceEngine:
    """Return a backward-compatible engine instance."""
    engine = get_client_inference_engine(config)
    # Runtime type is ClientInferenceEngine; this cast-level alias preserves API.
    return engine  # type: ignore[return-value]


__all__ = [
    "InferenceConfig",
    "TurnResult",
    "InferenceResult",
    "InferenceEngine",
    "get_inference_engine",
]

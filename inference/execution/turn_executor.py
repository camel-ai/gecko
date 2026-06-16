"""Deprecated turn executor module.

This module previously relied on in-repo Mock Server internals. The server
has been migrated to standalone Gecko, and current execution paths use
`inference.client_engine` / `inference.multi_turn_executor*`.
"""

from __future__ import annotations


class TurnExecutor:  # pragma: no cover - compatibility guard only
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "inference.execution.turn_executor.TurnExecutor is deprecated after "
            "Mock Server extraction. Use inference.client_engine or "
            "inference.multi_turn_executor instead."
        )

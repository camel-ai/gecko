"""
BFCL multi-turn runtime helpers.

Checker functions are exposed lazily to avoid circular imports with
`benchmarks.bfcl.official_eval`.
"""

from .executor import MultiTurnExecutor, execute_multi_turn_func_call
from .utils import compare_instances, is_empty_execute_response, is_subsequence_unordered

__all__ = [
    "multi_turn_checker",
    "multi_turn_irrelevance_checker",
    "state_checker",
    "response_checker",
    "execute_multi_turn_func_call",
    "MultiTurnExecutor",
    "is_empty_execute_response",
    "compare_instances",
    "is_subsequence_unordered",
]


def __getattr__(name: str):
    if name in {
        "multi_turn_checker",
        "multi_turn_irrelevance_checker",
        "state_checker",
        "response_checker",
    }:
        from benchmarks.bfcl.official_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
            multi_turn_checker,
            multi_turn_irrelevance_checker,
            response_checker,
            state_checker,
        )

        mapping = {
            "multi_turn_checker": multi_turn_checker,
            "multi_turn_irrelevance_checker": multi_turn_irrelevance_checker,
            "state_checker": state_checker,
            "response_checker": response_checker,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

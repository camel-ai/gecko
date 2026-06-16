"""
Logging helpers for tagging records with execution context via contextvars.

Use `set_task_context(task_id)` and/or `bind_log_context(...)` before execution.
Install the provided `TaskContextFilter` on interested loggers to automatically
prefix messages with task_id / agent_role / turn / attempt.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
import logging
from typing import Iterator, Optional

_task_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "task_id", default=None
)
_agent_role_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "agent_role", default=None
)
_turn_idx_var: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "turn_idx", default=None
)
_attempt_var: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "attempt", default=None
)
_registered_tools_summary_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "registered_tools_summary", default=None
)


def set_task_context(task_id: str) -> None:
    """Set current task id for logging context."""
    _task_id_var.set(task_id)


def set_execution_context(
    *,
    agent_role: Optional[str] = None,
    turn_idx: Optional[int] = None,
    attempt: Optional[int] = None,
    registered_tools_summary: Optional[str] = None,
) -> None:
    """Set current execution context fields for logging."""
    _agent_role_var.set(agent_role)
    _turn_idx_var.set(turn_idx)
    _attempt_var.set(attempt)
    _registered_tools_summary_var.set(registered_tools_summary)


def clear_execution_context() -> None:
    """Clear execution context fields while preserving task id."""
    _agent_role_var.set(None)
    _turn_idx_var.set(None)
    _attempt_var.set(None)
    _registered_tools_summary_var.set(None)


def clear_task_context() -> None:
    """Clear all logging context."""
    _task_id_var.set(None)
    clear_execution_context()


@contextmanager
def bind_log_context(
    *,
    task_id: Optional[str] = None,
    agent_role: Optional[str] = None,
    turn_idx: Optional[int] = None,
    attempt: Optional[int] = None,
    registered_tools_summary: Optional[str] = None,
) -> Iterator[None]:
    """Temporarily bind logging context for the current execution scope."""
    tokens = []
    try:
        if task_id is not None:
            tokens.append((_task_id_var, _task_id_var.set(task_id)))
        if agent_role is not None:
            tokens.append((_agent_role_var, _agent_role_var.set(agent_role)))
        if turn_idx is not None:
            tokens.append((_turn_idx_var, _turn_idx_var.set(turn_idx)))
        if attempt is not None:
            tokens.append((_attempt_var, _attempt_var.set(attempt)))
        if registered_tools_summary is not None:
            tokens.append(
                (_registered_tools_summary_var, _registered_tools_summary_var.set(registered_tools_summary))
            )
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


class TaskContextFilter(logging.Filter):
    """Attach execution context from contextvars to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        parts = []
        task_id = _task_id_var.get()
        agent_role = _agent_role_var.get()
        turn_idx = _turn_idx_var.get()
        attempt = _attempt_var.get()
        registered_tools_summary = _registered_tools_summary_var.get()
        if task_id:
            parts.append(f"task {task_id}")
        if agent_role:
            parts.append(f"agent {agent_role}")
        if turn_idx is not None:
            parts.append(f"turn {turn_idx}")
        if attempt is not None:
            parts.append(f"attempt {attempt}")
        if parts:
            record.msg = f"[{' | '.join(parts)}] {record.msg}"
        if (
            registered_tools_summary
            and isinstance(record.msg, str)
            and "not found in registered tools" in record.msg
        ):
            record.msg = f"{record.msg} [registered_tools: {registered_tools_summary}]"
        return True

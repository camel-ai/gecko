"""
Logging helpers for tagging records with task_id via contextvars.

Use `set_task_context(task_id)` before task execution and `clear_task_context()`
afterwards. Install the provided `TaskContextFilter` on interested loggers
to automatically prefix messages with the task_id.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Optional

# Context variable to hold current task identifier
_task_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "task_id", default=None
)


def set_task_context(task_id: str) -> None:
    """Set current task id for logging context."""
    _task_id_var.set(task_id)


def clear_task_context() -> None:
    """Clear current task id from logging context."""
    _task_id_var.set(None)


class TaskContextFilter(logging.Filter):
    """Attach task_id from contextvars to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        task_id = _task_id_var.get()
        if task_id:
            # Prefix message with task_id for readability
            record.msg = f"[task {task_id}] {record.msg}"
        return True


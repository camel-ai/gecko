"""
Session Context Manager for Real Tool Execution

Uses contextvars for thread-safe and asyncio-safe session tracking.
"""

import contextvars
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


# Use contextvars instead of global variables for thread/async safety
_session_context: contextvars.ContextVar[Optional['SessionContext']] = contextvars.ContextVar(
    'session_context',
    default=None
)

@dataclass
class SessionContext:
    """Session context for real tool execution.

    Attributes:
        session_id: Mock server session ID
        mock_server_url: Mock server base URL (e.g., "http://localhost:8000")
        task_id: Optional task identifier for logging/debugging
        buffer_updates: Whether to buffer tool updates instead of syncing immediately
        _buffer: Internal buffer for tool calls
    """
    session_id: str
    mock_server_url: str
    task_id: str | None = None
    buffer_updates: bool = False
    _buffer: List[Dict[str, Any]] = field(default_factory=list)
    _last_flushed_state: Optional[Dict[str, Any]] = None

    _READ_ONLY_OPS = {
        "ls",
        "pwd",
        "grep",
        "find",
        "cat",
        "head",
        "tail",
        "wc",
        "du",
        "sort",
        "diff",
        "read",
        "search",
        "query",
        "list",
        "show",
        "get",
        "status",
    }

    _STATEFUL_KEYWORDS = {
        "cd",
        "mkdir",
        "mv",
        "cp",
        "rm",
        "rmdir",
        "touch",
        "create",
        "update",
        "delete",
        "add",
        "remove",
        "insert",
        "append",
        "set",
        "post",
        "write",
        "rename",
        "move",
        "auth",
        "login",
        "logout",
        "follow",
        "unfollow",
        "retweet",
        "comment",
        "mention",
    }

    def add_to_buffer(self, tool_call: Dict[str, Any]) -> None:
        """Add a tool call to the buffer."""
        self._buffer.append(tool_call)

    def get_buffer(self) -> List[Dict[str, Any]]:
        """Get the current buffer content."""
        return list(self._buffer)

    def clear_buffer(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()

    @classmethod
    def _is_state_relevant_tool_call(cls, tool_call: Dict[str, Any]) -> bool:
        """Best-effort filtering: keep calls that may mutate or switch runtime context."""
        name = str(tool_call.get("name", "")).strip().lower()
        if not name:
            return False

        op = name.split("_")[-1]

        if op in cls._READ_ONLY_OPS:
            return False

        if any(keyword in op for keyword in cls._STATEFUL_KEYWORDS):
            return True

        # Conservative fallback for unknown operations: keep to avoid losing state changes.
        return True

    @staticmethod
    def _normalize_tool_call_for_sync(tool_call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize buffered call payload before sending to /update-state-from-real."""
        if not isinstance(tool_call, dict):
            return None

        name = tool_call.get("name") or tool_call.get("function") or ""
        raw_arguments = tool_call.get("arguments") or tool_call.get("args") or {}
        arguments = raw_arguments

        # Handle envelope-style arguments from some tool executors.
        if isinstance(arguments, dict):
            nested_kwargs = arguments.get("kwargs")
            if isinstance(nested_kwargs, dict):
                arguments = nested_kwargs
            if not name:
                nested_name = raw_arguments.get("_tool_name")
                if isinstance(nested_name, str) and nested_name:
                    name = nested_name

        if not isinstance(name, str) or not name.strip():
            return None
        if not isinstance(arguments, dict):
            arguments = {}

        return {
            "name": name.strip(),
            "arguments": arguments,
            "result": tool_call.get("result"),
        }

    def __enter__(self):
        """Context manager entry - set this context as current."""
        _session_context.set(self)
        return self

    def __exit__(self, *args):
        """Context manager exit - clear current context."""
        _session_context.set(None)

    def flush_buffer(self) -> Optional[Dict[str, Any]]:
        """Flush buffered real tool updates to mock server.

        Returns:
            Updated state returned by Gecko when available.
        """
        if not self._buffer:
            return None

        import requests
        import logging
        logger = logging.getLogger(__name__)

        filtered_calls = [tc for tc in self._buffer if self._is_state_relevant_tool_call(tc)]
        normalized_calls = [
            normalized
            for normalized in (self._normalize_tool_call_for_sync(tc) for tc in filtered_calls)
            if normalized is not None
        ]
        skipped = len(self._buffer) - len(filtered_calls)
        logger.info(
            "[CONTEXT] Flushing %d buffered real tool updates (%d skipped as read-only)",
            len(normalized_calls),
            skipped,
        )

        if not normalized_calls:
            self.clear_buffer()
            logger.info("[CONTEXT] No state-relevant tool updates to flush")
            return self._last_flushed_state

        # [FIX] Removed try-except block to allow exceptions to propagate (Fail Fast)
        # Send batch update
        url = f"{self.mock_server_url}/update-state-from-real"
        # Send as 'tool_calls' (plural)
        payload = {"tool_calls": normalized_calls}

        response = requests.post(
            url,
            json=payload,
            headers={"X-Session-ID": self.session_id},
            timeout=120  # Longer timeout for batch
        )
        response.raise_for_status()

        updated_state: Optional[Dict[str, Any]] = None
        try:
            body = response.json() if response.content else {}
            if isinstance(body, dict) and isinstance(body.get("updated_state"), dict):
                updated_state = body.get("updated_state")
        except Exception:
            updated_state = None

        self._last_flushed_state = updated_state
        self.clear_buffer()
        logger.info(f"[CONTEXT] Successfully flushed buffer")
        return updated_state

    def maybe_flush_buffer(self) -> Optional[Dict[str, Any]]:
        """Flush buffer only when non-empty.

        Returns:
            Updated state returned by Gecko when available.
        """
        if not self._buffer:
            return None
        return self.flush_buffer()


def get_current_session() -> Optional[SessionContext]:
    """Get the current session context.

    Returns:
        Current SessionContext if set, None otherwise.

    Example:
        >>> context = get_current_session()
        >>> if context:
        ...     print(f"Session ID: {context.session_id}")
    """
    return _session_context.get()


def set_session_context(
    session_id: str, 
    mock_server_url: str, 
    task_id: str | None = None,
    buffer_updates: bool = False
) -> SessionContext:
    """Set the current session context.

    Args:
        session_id: Mock server session ID
        mock_server_url: Mock server base URL
        task_id: Optional task identifier for logging
        buffer_updates: Whether to buffer tool updates

    Returns:
        The created SessionContext

    Example:
        >>> set_session_context("abc123", "http://localhost:8000", task_id="airline:0")
        >>> # Real tool wrappers can now access session info
        >>> agent.step(message)
    """
    context = SessionContext(
        session_id=session_id, 
        mock_server_url=mock_server_url, 
        task_id=task_id,
        buffer_updates=buffer_updates
    )
    _session_context.set(context)
    return context


def clear_session_context() -> None:
    """Clear the current session context.

    Should be called at the end of each turn to avoid context leakage.

    Example:
        >>> set_session_context("abc123", "http://localhost:8000")
        >>> try:
        ...     agent.step(message)
        ... finally:
        ...     clear_session_context()
    """
    _session_context.set(None)

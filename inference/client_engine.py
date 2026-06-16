import json
import logging
import os
import threading
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import requests

from utils.bfcl_multi_turn_tool_names import normalize_bfcl_multi_turn_tool_call
from utils.test_case_adapter import TestCaseAdapter

logger = logging.getLogger(__name__)

REAL_SYNC_HTTP_TIMEOUT_SECONDS = 300.0


class TaskInfraError(RuntimeError):
    """Infrastructure-level error: task should be treated as not executed."""


class MockServerClient:
    """Small HTTP client for Gecko session and state endpoints."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.close_connections = os.getenv("GECKO_CLIENT_KEEP_ALIVE", "").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.timeout = 600
        self._lock_guard = threading.Lock()
        self._single_flight_locks: Dict[str, threading.Lock] = {}

    @staticmethod
    def _new_request_id() -> str:
        return f"gecko-{uuid.uuid4().hex[:12]}"

    def _headers(
        self,
        *,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        extra: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if session_id:
            headers["X-Session-ID"] = session_id
        headers["X-Request-ID"] = request_id or self._new_request_id()
        if self.close_connections:
            headers["Connection"] = "close"
        if extra:
            headers.update(extra)
        return headers

    @staticmethod
    def _state_chars(state: Dict[str, Any]) -> int:
        return len(json.dumps(state or {}, ensure_ascii=False, default=str))

    def _task_lock_key(
        self,
        test_case: Optional[Any] = None,
        session_id: Optional[str] = None,
    ) -> str:
        if session_id:
            return f"session:{session_id}"
        if test_case is not None:
            try:
                test_id = TestCaseAdapter.get_id(test_case)
                if test_id:
                    return f"task:{test_id}"
            except Exception:
                pass
        return "global"

    @contextmanager
    def _single_flight(self, key: str):
        with self._lock_guard:
            lock = self._single_flight_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._single_flight_locks[key] = lock
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    def create_session(
        self,
        test_case: Optional[Any] = None,
        *,
        timeout: Optional[float] = None,
        retries: int = 0,
        backoff_sec: float = 0.5,
    ) -> str:
        request_timeout = float(timeout if timeout is not None else self.timeout)
        key = self._task_lock_key(test_case=test_case)
        with self._single_flight(key):
            request_id = self._new_request_id()
            try:
                response = self.session.get(
                    f"{self.base_url}/session-id",
                    headers=self._headers(request_id=request_id),
                    timeout=request_timeout,
                )
                response.raise_for_status()
                session_id = response.json()["session_id"]
                self._last_session_id = session_id
                logger.info(
                    "Created session: %s (request_id=%s timeout=%.1fs)",
                    session_id,
                    request_id,
                    request_timeout,
                )
                return session_id
            except Exception as exc:
                logger.error(
                    "Failed to create session (request_id=%s timeout=%.1fs): %s",
                    request_id,
                    request_timeout,
                    exc,
                )
                raise TaskInfraError(
                    f"UNTESTED_TASK: create-session failed "
                    f"(timeout={request_timeout:.1f}s): {exc}"
                ) from exc

    @staticmethod
    def _normalize_real_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for tool_call in tool_calls or []:
            if not isinstance(tool_call, dict):
                continue
            normalized_call = normalize_bfcl_multi_turn_tool_call(
                tool_call,
                rename=True,
                preserve_original=True,
            )
            name = (
                normalized_call.get("name")
                or normalized_call.get("function")
                or normalized_call.get("function_name")
                or ""
            )
            if not name:
                continue
            arguments = (
                normalized_call.get("arguments")
                or normalized_call.get("args")
                or {}
            )
            payload = {
                "name": name,
                "arguments": arguments if isinstance(arguments, dict) else arguments,
                "result": normalized_call.get("result"),
            }
            if normalized_call.get("toolkit"):
                payload["toolkit"] = normalized_call["toolkit"]
            normalized.append(payload)
        return normalized

    def sync_state_from_real_results(
        self,
        *,
        base_state: Dict[str, Any],
        tool_calls: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        test_case: Optional[Any] = None,
        io_timeout: Optional[float] = None,
        retries: int = 0,
    ) -> Dict[str, Any]:
        """Sync real tool results to Gecko and return updated state."""
        sid = session_id or self.create_session(test_case)
        request_timeout = float(
            io_timeout if io_timeout is not None else REAL_SYNC_HTTP_TIMEOUT_SECONDS
        )

        self.set_session_state(
            sid,
            base_state or {},
            timeout=request_timeout,
            retries=retries,
            backoff_sec=0.5,
        )

        normalized = self._normalize_real_tool_calls(tool_calls)
        if not normalized:
            return base_state or {}

        key = self._task_lock_key(session_id=sid)
        with self._single_flight(key):
            request_id = self._new_request_id()
            try:
                response = self.session.post(
                    f"{self.base_url}/update-state-from-real",
                    headers=self._headers(session_id=sid, request_id=request_id),
                    json={"tool_calls": normalized},
                    timeout=request_timeout,
                )
                response.raise_for_status()
                payload = response.json() if response.content else {}
                updated = payload.get("updated_state") if isinstance(payload, dict) else None
                if isinstance(updated, dict):
                    logger.info(
                        "Synced real state (request_id=%s session=%s tool_calls=%s timeout=%.1fs)",
                        request_id,
                        sid,
                        len(normalized),
                        request_timeout,
                    )
                    return updated
            except Exception as exc:
                logger.error(
                    "Failed to sync state from real results "
                    "(request_id=%s session=%s tool_calls=%s timeout=%.1fs): %s",
                    request_id,
                    sid,
                    len(normalized),
                    request_timeout,
                    exc,
                )
                raise TaskInfraError(
                    f"UNTESTED_TASK: update-state-from-real failed (session={sid}): {exc}"
                ) from exc

        return self.get_session_state(
            sid,
            timeout=request_timeout,
            retries=retries,
            backoff_sec=0.5,
        )

    def init_session_state(
        self,
        session_id: str,
        state: Dict[str, Any],
        *,
        involved_classes: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        retries: int = 0,
        backoff_sec: float = 0.5,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"state": state}
        if involved_classes:
            payload["involved_classes"] = list(involved_classes)
        request_timeout = float(timeout if timeout is not None else self.timeout)

        key = self._task_lock_key(session_id=session_id)
        with self._single_flight(key):
            request_id = self._new_request_id()
            try:
                response = self.session.post(
                    f"{self.base_url}/init-session-state",
                    headers=self._headers(session_id=session_id, request_id=request_id),
                    json=payload,
                    timeout=request_timeout,
                )
                response.raise_for_status()
                body = response.json() if response.content else {}
                initialized_state = body.get("state", {}) if isinstance(body, dict) else {}
                logger.info(
                    "Initialized state for session %s "
                    "(request_id=%s state_chars=%s involved_classes=%s timeout=%.1fs)",
                    session_id,
                    request_id,
                    self._state_chars(state),
                    len(involved_classes or []),
                    request_timeout,
                )
                return initialized_state if isinstance(initialized_state, dict) else {}
            except Exception as exc:
                logger.error(
                    "Failed to initialize session state "
                    "(request_id=%s timeout=%.1fs, session=%s state_chars=%s involved_classes=%s): %s",
                    request_id,
                    request_timeout,
                    session_id,
                    self._state_chars(state),
                    len(involved_classes or []),
                    exc,
                )
                raise TaskInfraError(
                    f"UNTESTED_TASK: init-session-state failed "
                    f"(session={session_id}, timeout={request_timeout:.1f}s): {exc}"
                ) from exc

    def set_session_state(
        self,
        session_id: str,
        state: Dict[str, Any],
        *,
        timeout: Optional[float] = None,
        retries: int = 0,
        backoff_sec: float = 0.5,
    ) -> bool:
        request_timeout = float(timeout if timeout is not None else self.timeout)

        key = self._task_lock_key(session_id=session_id)
        with self._single_flight(key):
            request_id = self._new_request_id()
            try:
                response = self.session.post(
                    f"{self.base_url}/set-session-state",
                    headers=self._headers(session_id=session_id, request_id=request_id),
                    json={"state": state},
                    timeout=request_timeout,
                )
                response.raise_for_status()
                logger.info(
                    "Set state for session %s (request_id=%s state_chars=%s timeout=%.1fs)",
                    session_id,
                    request_id,
                    self._state_chars(state),
                    request_timeout,
                )
                return True
            except Exception as exc:
                logger.error(
                    "Failed to set session state "
                    "(request_id=%s timeout=%.1fs, session=%s state_chars=%s): %s",
                    request_id,
                    request_timeout,
                    session_id,
                    self._state_chars(state),
                    exc,
                )
                raise TaskInfraError(
                    f"UNTESTED_TASK: set-session-state failed "
                    f"(session={session_id}, timeout={request_timeout:.1f}s): {exc}"
                ) from exc

    def get_session_state(
        self,
        session_id: str,
        *,
        timeout: Optional[float] = None,
        retries: int = 0,
        backoff_sec: float = 0.5,
    ) -> Dict[str, Any]:
        request_timeout = float(timeout if timeout is not None else self.timeout)

        key = self._task_lock_key(session_id=session_id)
        with self._single_flight(key):
            request_id = self._new_request_id()
            try:
                response = self.session.get(
                    f"{self.base_url}/get-session-state",
                    headers=self._headers(session_id=session_id, request_id=request_id),
                    timeout=request_timeout,
                )
                response.raise_for_status()
                return response.json().get("state", {})
            except Exception as exc:
                logger.error(
                    "Failed to get session state "
                    "(request_id=%s timeout=%.1fs, session=%s): %s",
                    request_id,
                    request_timeout,
                    session_id,
                    exc,
                )
                raise TaskInfraError(
                    f"UNTESTED_TASK: get-session-state failed "
                    f"(session={session_id}, timeout={request_timeout:.1f}s): {exc}"
                ) from exc

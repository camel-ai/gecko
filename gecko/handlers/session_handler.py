import asyncio
import json
import os
import uuid
import sqlite3
from typing import Any, Dict, List, Optional, Union
import threading

from fastapi import APIRouter, Request, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time

from ..utils.config_updater import update_state as util_update_state
from ..utils.config_updater import bootstrap_state, extract_toolkit_summaries
from ..schemas.global_loader import get_global_schema_loader
from ..utils.global_config import (
    get_response_model,
    get_state_model,
    get_validation_model,
)

import logging

logger = logging.getLogger(__name__)


class UpdateStateRequest(BaseModel):
    task: Optional[str] = None
    previous_state: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]
    execution_results: Optional[List[Union[Dict[str, Any], None]]] = None
    tool_descriptions: Optional[Dict[str, Any]] = None


class SessionHandler:
    """Handler for managing session data with persistence using SQLite."""
    
    def __init__(self, db_path="sessions.db"):
        """Initialize the session handler."""
        self.db_path = db_path
        self.started_at = time.time()
        self.local = threading.local()
        self._init_db()
        self.router = APIRouter()
        self._register_routes()
    
    def _get_connection(self):
        """Get a thread-local database connection."""
        if not hasattr(self.local, "connection"):
            self.local.connection = sqlite3.connect(self.db_path, timeout=5.0)
            self.local.connection.row_factory = sqlite3.Row
            self._configure_connection(self.local.connection)
        return self.local.connection

    @staticmethod
    def _configure_connection(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")

    @staticmethod
    def _rollback_quietly(conn: sqlite3.Connection, context: str) -> None:
        try:
            conn.rollback()
        except Exception:
            logger.exception("Failed to rollback SQLite transaction after %s", context)
    
    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        self._configure_connection(conn)

        table_info = conn.execute("PRAGMA table_info(sessions)").fetchall()
        if table_info:
            columns = {row[1] for row in table_info}
            if "latest_state" not in columns:
                conn.execute("DROP TABLE IF EXISTS sessions")

        conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                latest_state TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    @staticmethod
    async def _run_in_thread(
        fn,
        *args,
        **kwargs,
    ):
        """Offload a blocking call to the threadpool."""
        return await asyncio.to_thread(fn, *args, **kwargs)

    def _register_routes(self):
        """Register the session routes."""
        @self.router.get("/health")
        async def health_check():
            """Return Gecko health and current runtime configuration."""
            return JSONResponse(content=self.get_health_status())

        @self.router.get("/session-id")
        async def get_session_id():
            """Get a new session ID."""
            session_id = str(uuid.uuid4())
            conn = self._get_connection()
            try:
                conn.execute(
                    "INSERT INTO sessions (session_id, latest_state) VALUES (?, ?)",
                    (session_id, None)
                )
                conn.commit()
            except Exception:
                self._rollback_quietly(conn, "create session")
                logger.exception("Failed to create session row (session=%s)", session_id)
                raise
            return JSONResponse(
                content={
                    "session_id": session_id,
                    "message": "New session ID generated. Include this in the X-Session-ID header for subsequent requests."
                }
            )
        
        @self.router.post("/init-session-state")
        async def init_session_state(request: Request):
            """Initialize the session state once with defaults merge + bootstrap."""
            session_id = request.headers.get("X-Session-ID")
            if not session_id or not self.validate_session(session_id):
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid or missing session ID"}
                )

            body = await request.json()
            state = body.get("state")
            involved_classes = body.get("involved_classes") or []
            if state is None:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Missing state in request body"}
                )

            state = self._merge_schema_defaults(state, involved_classes)
            enriched_state = await self._run_in_thread(
                self._maybe_bootstrap,
                session_id,
                state,
                "auto",
            )
            self.add_to_state(session_id, enriched_state)
            return JSONResponse(
                content={
                    "message": "Session state initialized successfully",
                    "state": enriched_state,
                }
            )

        @self.router.post("/set-session-state")
        async def set_session_state(request: Request):
            """Persist the provided session state without initialization work."""
            session_id = request.headers.get("X-Session-ID")
            if not session_id or not self.validate_session(session_id):
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid or missing session ID"}
                )

            body = await request.json()
            state = body.get("state")
            if state is None:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Missing state in request body"}
                )
            self.add_to_state(session_id, state)
            return JSONResponse(
                content={
                    "message": "Session state set successfully"
                }
            )

        @self.router.get("/get-session-state")
        async def get_session_state_endpoint(request: Request):
            """Get the session state."""
            session_id = request.headers.get("X-Session-ID")
            if not session_id or not self.validate_session(session_id):
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid or missing session ID"}
                )

            return JSONResponse(
                content={"state": self.get_session_state(session_id)}
            )

        @self.router.post("/update_state", response_model=Dict[str, Any])
        async def update_state_endpoint(request_body: UpdateStateRequest, session_id: str = Header(alias="X-Session-ID")):
            """Update state based on task, previous_state, and tool_calls."""
            try:
                updated_state = await self._run_in_thread(
                    util_update_state,
                    previous_state=request_body.previous_state,
                    tool_calls=request_body.tool_calls,
                    task=request_body.task,
                    execution_results=request_body.execution_results,
                    tool_descriptions=request_body.tool_descriptions,
                    session_id=session_id,
                )
                return updated_state
            except Exception as e:
                logger.exception("Error in /update_state endpoint: %s", e)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.post("/update-state-from-real")
        async def update_state_from_real_execution(request: Request):
            """Sync real tool execution results into session state."""
            session_id = request.headers.get("X-Session-ID")
            try:
                if not session_id or not self.validate_session(session_id):
                    raise HTTPException(status_code=400, detail="Invalid or missing session ID")

                try:
                    body = await request.json()
                    tool_call = body.get("tool_call")
                    tool_calls = body.get("tool_calls")
                    read_only_sync_policy = body.get("read_only_sync_policy")

                    if not tool_call and not tool_calls:
                        raise HTTPException(status_code=400, detail="Missing tool_call or tool_calls in request body")

                    calls_to_process = []
                    if tool_calls:
                        if not isinstance(tool_calls, list):
                            raise HTTPException(status_code=400, detail="tool_calls must be a list")
                        calls_to_process.extend(tool_calls)
                    if tool_call:
                        calls_to_process.append(tool_call)

                    for tc in calls_to_process:
                        if not all(k in tc for k in ["name", "arguments", "result"]):
                            raise HTTPException(
                                status_code=400,
                                detail="tool_call must contain 'name', 'arguments', and 'result'"
                            )
                except json.JSONDecodeError:
                    raise HTTPException(status_code=400, detail="Invalid JSON in request body")

                current_state = self.get_session_state(session_id)

                try:
                    from ..utils.config_updater import update_state_from_real_tool
                    from ..utils.global_config import get_state_model

                    updated_state = await self._run_in_thread(
                        update_state_from_real_tool,
                        previous_state=current_state,
                        tool_call=calls_to_process,
                        session_id=session_id,
                        state_model=get_state_model(),
                        read_only_sync_policy=read_only_sync_policy if isinstance(read_only_sync_policy, str) else None,
                    )
                except Exception as e:
                    tool_names = []
                    for tc in calls_to_process:
                        if isinstance(tc, dict):
                            tool_names.append(str(tc.get("name")))
                        else:
                            tool_names.append(type(tc).__name__)
                    logger.exception(
                        "Failed to update state from real tools (session=%s, calls=%d, tool_names=%s): %s",
                        session_id,
                        len(calls_to_process),
                        tool_names,
                        e,
                    )
                    raise HTTPException(status_code=500, detail=f"State update failed: {str(e)}")

                return JSONResponse(
                    content={
                        "success": True,
                        "updated_state": updated_state,
                    }
                )
            except HTTPException:
                raise
            except Exception:
                raise
        
    
    def _merge_schema_defaults(
        self,
        state: Dict[str, Any],
        involved_classes: List[str],
    ) -> Dict[str, Any]:
        """Merge `info.x-default-state.global.runtime_defaults` under each toolkit."""
        if not isinstance(state, dict):
            return state

        schema_loader = get_global_schema_loader()
        if schema_loader is None:
            return state

        state_toolkits = [
            k for k in state.keys()
            if k not in ("runtime_state",) and isinstance(state.get(k), dict)
        ]
        candidates = list(dict.fromkeys(list(involved_classes) + state_toolkits))

        merged = dict(state)
        for name in candidates:
            schema_path = schema_loader.find_schema_file(name)
            if not schema_path:
                continue
            try:
                schema = schema_loader.load_schema(schema_path)
            except Exception as e:
                logger.warning("[DEFAULTS] Failed to load schema for %s: %s", name, e)
                continue

            defaults = (
                schema.get("info", {})
                .get("x-default-state", {})
                .get("global", {})
                .get("runtime_defaults", {})
            )
            if not isinstance(defaults, dict) or not defaults:
                continue

            current = merged.get(name)
            if not isinstance(current, dict):
                current = {}
            merged[name] = {**defaults, **current}
            logger.info(
                "[DEFAULTS] Merged %d default keys into toolkit %s",
                len(set(defaults.keys()) - set(current.keys())),
                name,
            )

        return merged
    
    def _maybe_bootstrap(
        self,
        session_id: str,
        state: Dict[str, Any],
        bootstrap_mode: str = "auto",
    ) -> Dict[str, Any]:
        """Bootstrap runtime_state on first state set for a session.

        If the session has no prior state entries, initializes runtime
        state from structured x-default-state runtime rules when available.
        Returns the enriched state or the original state if bootstrap is
        skipped or no structured bootstrap applies.
        """
        mode = str(bootstrap_mode or "auto").lower()
        if mode not in {"auto", "skip", "force"}:
            logger.warning(
                "[BOOTSTRAP] Invalid bootstrap_mode=%s for session %s, fallback to auto",
                bootstrap_mode,
                session_id,
            )
            mode = "auto"

        if mode == "skip":
            logger.info("[BOOTSTRAP] bootstrap_mode=skip for session %s", session_id)
            return state

        if self.has_session_state(session_id) and mode != "force":
            return state

        if mode == "auto":
            runtime_state = state.get("runtime_state") if isinstance(state, dict) else None
            if isinstance(runtime_state, dict):
                nested_toolkits = runtime_state.get("toolkits")
                has_nested_runtime = isinstance(nested_toolkits, dict) and bool(nested_toolkits)
                has_flat_runtime = any(
                    k != "toolkits" for k in runtime_state.keys()
                )
                if has_nested_runtime or has_flat_runtime:
                    logger.info(
                        "[BOOTSTRAP] Runtime state already present for session %s, skipping bootstrap",
                        session_id,
                    )
                    return state

        schema_loader = get_global_schema_loader()
        if schema_loader is None:
            logger.debug("[BOOTSTRAP] No global schema loader available, skipping bootstrap")
            return state

        toolkit_names = [
            k for k in (state or {}).keys()
            if k not in ("runtime_state",) and isinstance(state.get(k), dict)
        ]
        if not toolkit_names:
            logger.debug("[BOOTSTRAP] No toolkit keys found in state, skipping bootstrap")
            return state

        schemas: Dict[str, Any] = {}
        for name in toolkit_names:
            schema_path = schema_loader.find_schema_file(name)
            if schema_path:
                try:
                    schemas[name] = schema_loader.load_schema(schema_path)
                except Exception as e:
                    logger.warning(f"[BOOTSTRAP] Failed to load schema for {name}: {e}")

        if not schemas:
            logger.debug("[BOOTSTRAP] No schemas loaded, skipping bootstrap")
            return state

        toolkit_summaries = extract_toolkit_summaries(schemas)
        if not toolkit_summaries:
            logger.debug("[BOOTSTRAP] No toolkit summaries extracted, skipping bootstrap")
            return state

        try:
            enriched = bootstrap_state(state, toolkit_summaries)
            logger.info(f"[BOOTSTRAP] State bootstrapped for session {session_id}")
            return enriched
        except Exception as e:
            logger.warning(f"[BOOTSTRAP] Failed to bootstrap state: {e}")
            return state

    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Get the latest state snapshot for a specific session."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT latest_state FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        if row and row["latest_state"] is not None:
            return json.loads(row["latest_state"])
        return {}

    def has_session_state(self, session_id: str) -> bool:
        """Return whether the session already has a persisted latest state."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT latest_state FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        return bool(row and row["latest_state"] is not None)
    
    def add_to_state(self, session_id: str, state_details: Dict[str, Any]):
        """Persist the latest state snapshot for the session."""
        conn = self._get_connection()

        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE sessions SET latest_state = ? WHERE session_id = ?",
                (json.dumps(state_details), session_id)
            )
            conn.commit()
        except Exception:
            self._rollback_quietly(conn, "persist session state")
            logger.exception("Failed to persist session state (session=%s)", session_id)
            raise
    
    def validate_session(self, session_id: str) -> bool:
        """Validate if a session ID exists."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
        return cursor.fetchone() is not None

    def get_health_status(self) -> Dict[str, Any]:
        """Return a lightweight health snapshot for the current Gecko process."""
        schema_loader = get_global_schema_loader()
        schema_dirs = []
        schema_cache_size = 0
        if schema_loader is not None:
            schema_dirs = list(getattr(schema_loader, "schema_dirs", []) or [])
            schema_cache_size = len(getattr(schema_loader, "schemas_cache", {}) or {})

        conn = self._get_connection()
        sessions_row = conn.execute("SELECT COUNT(*) AS count FROM sessions").fetchone()
        session_count = int(sessions_row["count"] or 0) if sessions_row else 0

        state_model = get_state_model()
        uptime_seconds = max(0.0, time.time() - self.started_at)
        try:
            workers = int(os.environ.get("GECKO_WORKERS", "1") or "1")
        except ValueError:
            workers = 1

        return {
            "status": "ok",
            "service": "gecko",
            "process_id": os.getpid(),
            "uptime_seconds": round(uptime_seconds, 3),
            "config": {
                "db_path": self.db_path,
                "schema_dirs": schema_dirs,
                "schema_cache_size": schema_cache_size,
                "response_model": get_response_model(),
                "validation_model": get_validation_model(),
                "state_model": state_model,
                "state_model_enabled": state_model is not None,
                "workers": workers,
            },
            "stats": {
                "session_count": session_count,
            },
        }

session_handler = SessionHandler()
session_router = session_handler.router 

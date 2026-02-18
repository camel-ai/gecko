import json
import uuid
import sqlite3
from typing import Any, Dict, List, Optional, Union
import threading

from fastapi import APIRouter, Request, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time

from ..utils.request_details import RequestDetails
from ..utils.config_updater import update_state as util_update_state
from ..utils.config_updater import bootstrap_state, extract_toolkit_summaries
from ..schemas.global_loader import get_global_schema_loader

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
        self.local = threading.local()
        self._init_db()
        self.router = APIRouter()
        self._register_routes()
    
    def _get_connection(self):
        """Get a thread-local database connection."""
        if not hasattr(self.local, "connection"):
            self.local.connection = sqlite3.connect(self.db_path)
            self.local.connection.row_factory = sqlite3.Row
        return self.local.connection
    
    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        # Hard reset old schema (`config` column) to the new `state` schema.
        table_info = conn.execute("PRAGMA table_info(sessions)").fetchall()
        if table_info:
            columns = {row[1] for row in table_info}
            if "state" not in columns:
                conn.execute("DROP TABLE IF EXISTS sessions")

        conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                history TEXT,
                state TEXT
            )
        ''')
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_usage_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                ts REAL NOT NULL,
                category TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_session_id ON llm_usage_events(session_id)")
        conn.commit()
        conn.close()
    
    def _register_routes(self):
        """Register the session routes."""
        @self.router.get("/session-id")
        async def get_session_id():
            """Get a new session ID."""
            session_id = str(uuid.uuid4())
            conn = self._get_connection()
            conn.execute(
                "INSERT INTO sessions (session_id, history, state) VALUES (?, ?, ?)",
                (session_id, json.dumps([]), json.dumps([]))
            )
            conn.commit()
            return JSONResponse(
                content={
                    "session_id": session_id,
                    "message": "New session ID generated. Include this in the X-Session-ID header for subsequent requests."
                }
            )
        
        @self.router.post("/set-session-state")
        async def set_session_state(request: Request):
            """Set the session state."""
            session_id = request.headers.get("X-Session-ID")
            if not session_id or not self.validate_session(session_id):
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid or missing session ID"}
                )
            
            body = await request.json()
            state = body.get("state")
            # Allow empty dict {}, but not None
            if state is None:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Missing state in request body"}
                )

            # Bootstrap runtime_state on first state set
            enriched_state = self._maybe_bootstrap(session_id, state)
            self.add_to_state(session_id, enriched_state)
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

            states = self.get_session_state(session_id)
            latest_state = states[-1] if len(states) > 0 else {}
            
            return JSONResponse(
                content={"state": latest_state}
            )

        @self.router.get("/get-session-history")
        async def get_session_history_endpoint(request: Request):
            """Get the session history."""
            session_id = request.headers.get("X-Session-ID")
            if not session_id or not self.validate_session(session_id):
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid or missing session ID"}
                )

            history = self.get_session_history(session_id)
            
            return JSONResponse(
                content={"history": history}
            )

        @self.router.post("/update_state", response_model=Dict[str, Any])
        async def update_state_endpoint(request_body: UpdateStateRequest, session_id: str = Header(alias="X-Session-ID")):
            """Update state based on task, previous_state, and tool_calls."""
            try:
                # util_update_state handles session update internally when session_id is provided.
                updated_state = util_update_state(
                    previous_state=request_body.previous_state,
                    tool_calls=request_body.tool_calls,
                    task=request_body.task,
                    execution_results=request_body.execution_results,
                    tool_descriptions=request_body.tool_descriptions,
                    session_id=session_id
                )
                return updated_state
            except Exception as e:
                logger.exception("Error in /update_state endpoint: %s", e)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.get("/get-session-llm-usage")
        async def get_session_llm_usage_endpoint(
            request: Request,
            include_events: bool = False,
            limit: int = 10000,
            since_id: int = 0,
        ):
            """Get token usage for all LLM calls recorded under this session.

            Parameters:
              - include_events: include per-call event list (can be large)
              - limit: max number of events returned when include_events is True
              - since_id: return only events with id > since_id (supports incremental pulls)
            """
            session_id = request.headers.get("X-Session-ID")
            if not session_id or not self.validate_session(session_id):
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid or missing session ID"},
                )

            usage = self.get_session_llm_usage(
                session_id=session_id,
                include_events=include_events,
                limit=limit,
                since_id=since_id,
            )
            return JSONResponse(content=usage)

        @self.router.post("/update-state-from-real")
        async def update_state_from_real_execution(request: Request):
            """Update state based on real tool execution results.

            This endpoint is called by RealToolWrapper to sync execution results
            to the mock server's session state. Supports batch updates.

            Request Body:
                {
                    "tool_call": {...}  // Single tool call (legacy)
                    OR
                    "tool_calls": [...] // List of tool calls (batch)
                }

            Returns:
                {
                    "success": true,
                    "updated_state": {...}
                }
            """
            # Get session ID from header
            session_id = request.headers.get("X-Session-ID")
            if not session_id or not self.validate_session(session_id):
                raise HTTPException(status_code=400, detail="Invalid or missing session ID")

            # Parse request body
            try:
                body = await request.json()
                tool_call = body.get("tool_call")
                tool_calls = body.get("tool_calls")

                if not tool_call and not tool_calls:
                    raise HTTPException(status_code=400, detail="Missing tool_call or tool_calls in request body")

                # Normalize to list
                calls_to_process = []
                if tool_calls:
                    if not isinstance(tool_calls, list):
                        raise HTTPException(status_code=400, detail="tool_calls must be a list")
                    calls_to_process.extend(tool_calls)
                if tool_call:
                    calls_to_process.append(tool_call)

                # Validate tool_call structure
                for tc in calls_to_process:
                    if not all(k in tc for k in ["name", "arguments", "result"]):
                        raise HTTPException(
                            status_code=400,
                            detail="tool_call must contain 'name', 'arguments', and 'result'"
                        )
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in request body")

            # Get current state
            states = self.get_session_state(session_id)
            current_state = states[-1] if len(states) > 0 else {}

            logger.info("[UPDATE STATE FROM REAL] Received %s tool calls", len(calls_to_process))

            # Use update_state_from_real_tool directly for batch efficiency
            try:
                from ..utils.config_updater import update_state_from_real_tool
                from ..utils.global_config import get_state_model

                updated_state = update_state_from_real_tool(
                    previous_state=current_state,
                    tool_call=calls_to_process,
                    session_id=session_id,
                    state_model=get_state_model()
                )

                logger.info(
                    "[UPDATE STATE FROM REAL] Updated state keys: %s",
                    list(updated_state.keys()) if isinstance(updated_state, dict) else type(updated_state),
                )
            except Exception as e:
                logger.exception("Failed to update state from real tools: %s", e)
                raise HTTPException(status_code=500, detail=f"State update failed: {str(e)}")

            # Optional: Record real tool calls in history for debugging
            try:
                for tc in calls_to_process:
                    self._record_real_tool_call(session_id, tc)
            except Exception as e:
                # Don't fail the request if history recording fails
                logger.warning("Failed to record real tool call in history: %s", e)

            return JSONResponse(
                content={
                    "success": True,
                    "updated_state": updated_state
                }
            )
        
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get the history for a specific session."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT history FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return []
    
    def add_to_history(self, session_id: str, request_details: RequestDetails, response_details: Dict[str, Any]):
        """Add a request/response pair to the session history."""
        conn = self._get_connection()
        
        # Get current history
        cursor = conn.execute("SELECT history FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if not row:
            history = []
        else:
            history = json.loads(row[0])
        
        # Add new entry
        history_entry = {
            "timestamp": self._get_timestamp(),
            "request": {
                "method": request_details.get("method"),
                "path": request_details.get("path"),
                "query_params": request_details.get("query_params"),
                "headers": request_details.get("headers"),
                "body": request_details.get("body")
            },
            "response": {
                "status_code": response_details.get("status_code"),
                "headers": response_details.get("headers"),
                "body": response_details.get("body")
            }
        }
        
        history.append(history_entry)
        
        # Update history
        conn.execute(
            "UPDATE sessions SET history = ? WHERE session_id = ?",
            (json.dumps(history), session_id)
        )
        conn.commit()
    
    def _maybe_bootstrap(self, session_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Bootstrap runtime_state on first state set for a session.

        If the session has no prior state entries, uses LLM to infer runtime
        state variables (e.g. current working directory) from the toolkit
        definitions and the initial state. Returns the enriched state
        or the original state if bootstrap is skipped/fails.
        """
        # Only bootstrap on first state set (empty state list).
        existing = self.get_session_state(session_id)
        if existing:
            return state

        # Need schemas to extract toolkit summaries
        schema_loader = get_global_schema_loader()
        if schema_loader is None:
            logger.debug("[BOOTSTRAP] No global schema loader available, skipping bootstrap")
            return state

        # Extract toolkit names from state top-level keys.
        # (skip non-toolkit meta keys like "runtime_state")
        toolkit_names = [
            k for k in (state or {}).keys()
            if k not in ("runtime_state",) and isinstance(state.get(k), dict)
        ]
        if not toolkit_names:
            logger.debug("[BOOTSTRAP] No toolkit keys found in state, skipping bootstrap")
            return state

        # Load schemas for each toolkit
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

    def get_session_state(self, session_id: str) -> List[Dict[str, Any]]:
        """Get the state snapshots for a specific session."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT state FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return []
    
    def add_to_state(self, session_id: str, state_details: Dict[str, Any]):
        """Append a state snapshot to the session."""
        conn = self._get_connection()

        try:
            # Serialize concurrent updates to prevent lost updates across workers.
            conn.execute("BEGIN IMMEDIATE")

            # Get current state snapshots.
            cursor = conn.execute("SELECT state FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if not row:
                states = []
            else:
                states = json.loads(row[0])

            states.append(state_details)

            # Persist state snapshots.
            conn.execute(
                "UPDATE sessions SET state = ? WHERE session_id = ?",
                (json.dumps(states), session_id)
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    def _get_timestamp(self) -> str:
        """Get the current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def validate_session(self, session_id: str) -> bool:
        """Validate if a session ID exists."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
        return cursor.fetchone() is not None
    
    def record_llm_usage(
        self,
        session_id: str,
        *,
        category: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        ts: Optional[float] = None,
    ) -> None:
        """Record one LLM call's token usage for a session."""
        if not session_id:
            return
        if ts is None:
            ts = time.time()

        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO llm_usage_events
                (session_id, ts, category, model, input_tokens, output_tokens, total_tokens)
            VALUES
                (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                float(ts),
                str(category),
                str(model),
                int(input_tokens or 0),
                int(output_tokens or 0),
                int(total_tokens or 0),
            ),
        )
        conn.commit()

    def get_session_llm_usage(
        self,
        session_id: str,
        *,
        include_events: bool = False,
        limit: int = 10000,
        since_id: int = 0,
    ) -> Dict[str, Any]:
        """Get aggregated token usage for a session, optionally including per-call events."""
        conn = self._get_connection()

        # Aggregation by model
        by_model: Dict[str, Dict[str, int]] = {}
        cursor = conn.execute(
            """
            SELECT model,
                   SUM(input_tokens) AS input_tokens,
                   SUM(output_tokens) AS output_tokens,
                   SUM(total_tokens) AS total_tokens
            FROM llm_usage_events
            WHERE session_id = ? AND id > ?
            GROUP BY model
            """,
            (session_id, int(since_id)),
        )
        for row in cursor.fetchall():
            by_model[str(row["model"])] = {
                "input_tokens": int(row["input_tokens"] or 0),
                "output_tokens": int(row["output_tokens"] or 0),
                "total_tokens": int(row["total_tokens"] or 0),
            }

        # Aggregation by category
        by_category: Dict[str, Dict[str, int]] = {}
        cursor = conn.execute(
            """
            SELECT category,
                   SUM(input_tokens) AS input_tokens,
                   SUM(output_tokens) AS output_tokens,
                   SUM(total_tokens) AS total_tokens
            FROM llm_usage_events
            WHERE session_id = ? AND id > ?
            GROUP BY category
            """,
            (session_id, int(since_id)),
        )
        for row in cursor.fetchall():
            by_category[str(row["category"])] = {
                "input_tokens": int(row["input_tokens"] or 0),
                "output_tokens": int(row["output_tokens"] or 0),
                "total_tokens": int(row["total_tokens"] or 0),
            }

        # Total
        cursor = conn.execute(
            """
            SELECT SUM(input_tokens) AS input_tokens,
                   SUM(output_tokens) AS output_tokens,
                   SUM(total_tokens) AS total_tokens
            FROM llm_usage_events
            WHERE session_id = ? AND id > ?
            """,
            (session_id, int(since_id)),
        )
        row = cursor.fetchone() or {}
        total = {
            "input_tokens": int((row["input_tokens"] or 0) if "input_tokens" in row.keys() else 0),
            "output_tokens": int((row["output_tokens"] or 0) if "output_tokens" in row.keys() else 0),
            "total_tokens": int((row["total_tokens"] or 0) if "total_tokens" in row.keys() else 0),
        }

        payload: Dict[str, Any] = {
            "session_id": session_id,
            "since_id": int(since_id),
            "by_model": by_model,
            "by_category": by_category,
            "total": total,
        }

        if include_events:
            cursor = conn.execute(
                """
                SELECT id, ts, category, model, input_tokens, output_tokens, total_tokens
                FROM llm_usage_events
                WHERE session_id = ? AND id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (session_id, int(since_id), int(limit)),
            )
            events = []
            max_id = int(since_id)
            for r in cursor.fetchall():
                rid = int(r["id"])
                max_id = max(max_id, rid)
                events.append(
                    {
                        "id": rid,
                        "ts": float(r["ts"]),
                        "category": str(r["category"]),
                        "model": str(r["model"]),
                        "input_tokens": int(r["input_tokens"] or 0),
                        "output_tokens": int(r["output_tokens"] or 0),
                        "total_tokens": int(r["total_tokens"] or 0),
                    }
                )
            payload["events"] = events
            payload["max_event_id"] = max_id

        return payload

    def _record_real_tool_call(self, session_id: str, tool_call: Dict[str, Any]):
        """Record a real tool call in session history.

        This helps maintain a complete history of both real and mock tool calls
        for debugging and example generation.

        Args:
            session_id: Session ID
            tool_call: Tool call dict with name, arguments, result
        """
        conn = self._get_connection()

        # Get current history
        cursor = conn.execute("SELECT history FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if not row:
            history = []
        else:
            history = json.loads(row[0])

        # Add new entry with "real_tool" type marker
        history_entry = {
            "timestamp": self._get_timestamp(),
            "type": "real_tool",  # Mark as real tool execution
            "tool_call": {
                "name": tool_call.get("name"),
                "arguments": tool_call.get("arguments"),
                "result": tool_call.get("result")
            }
        }

        history.append(history_entry)

        # Update history
        conn.execute(
            "UPDATE sessions SET history = ? WHERE session_id = ?",
            (json.dumps(history), session_id)
        )
        conn.commit()
    

# Create a singleton instance
session_handler = SessionHandler()
session_router = session_handler.router 

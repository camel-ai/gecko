from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ..handlers.session_handler import session_handler


class SessionMiddleware(BaseHTTPMiddleware):
    """Middleware for handling session management."""
    
    def __init__(self, app):
        """Initialize the session middleware.
        
        Args:
            app: The FastAPI application
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in {
            "/health",
            "/session-id",
            "/init-session-state",
            "/set-session-state",
        }:
            return await call_next(request)
        elif request.url.path == "/update-state-from-real":
            return await call_next(request)

        session_id = request.headers.get("X-Session-ID")
        if not session_id:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Missing session ID",
                    "message": "Include a valid session ID in the X-Session-ID header. Get a new session ID from /session-id endpoint."
                }
            )
        elif not session_handler.validate_session(session_id):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid session ID",
                    "message": "The provided session ID is not valid. Get a new session ID from /session-id endpoint."
                }
            )
        
        request.state.session_state = session_handler.get_session_state(session_id)
        request.state.session_has_state = session_handler.has_session_state(session_id)
        if (hasattr(request.state, 'request_handler') and 
            hasattr(request.state, 'matching_path') and 
            hasattr(request.state, 'matching_operation')):
            return await request.state.request_handler.handle_request(
                request, 
                request.state.matching_path, 
                request.state.matching_operation, 
                request.state.api_name
            )
        return await call_next(request)

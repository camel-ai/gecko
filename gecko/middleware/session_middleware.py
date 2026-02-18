from typing import Any, Callable, Dict, List
import json

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ..handlers.session_handler import session_handler
from ..utils.request_details import RequestDetails


class SessionMiddleware(BaseHTTPMiddleware):
    """Middleware for handling session management."""
    
    def __init__(self, app):
        """Initialize the session middleware.
        
        Args:
            app: The FastAPI application
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check if this is the session-id endpoint or state-sync endpoint.
        if request.url.path == "/session-id":
            return await call_next(request)
        elif request.url.path == "/set-session-state":
            return await call_next(request)
        elif request.url.path == "/update-state-from-real":
            # Real tool sync endpoint handles its own session validation
            return await call_next(request)

        # Require session ID in header for all other requests
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
        
        # Extract request details using the RequestDetails utility
        request_details = await RequestDetails.extract(request)
        
        request.state.session_history = session_handler.get_session_history(session_id)
        request.state.session_state = session_handler.get_session_state(session_id)
        # Call the next middleware/route handler
        # Check if this is an API request that needs handling
        if (hasattr(request.state, 'request_handler') and 
            hasattr(request.state, 'matching_path') and 
            hasattr(request.state, 'matching_operation')):
            # Handle API request
            response = await request.state.request_handler.handle_request(
                request, 
                request.state.matching_path, 
                request.state.matching_operation, 
                request.state.api_name
            )
        else:
            # Continue to next middleware
            response = await call_next(request)
        
        # Extract response body from JSONResponse
        response_body = None
        try:
            if isinstance(response, JSONResponse):
                # First check if we stored the content during response creation
                if hasattr(response, '_session_response_content'):
                    content = response._session_response_content
                    if isinstance(content, (dict, list)):
                        response_body = json.dumps(content, ensure_ascii=False)
                    else:
                        response_body = str(content)
                else:
                    # Fallback: try to access content through various means
                    content = None
                    for attr in ['content', '_content', 'body']:
                        if hasattr(response, attr):
                            attr_value = getattr(response, attr)
                            if attr_value is not None:
                                content = attr_value
                                break
                    
                    if content is not None:
                        if isinstance(content, bytes):
                            response_body = content.decode('utf-8')
                        elif isinstance(content, str):
                            response_body = content
                        elif isinstance(content, (dict, list)):
                            response_body = json.dumps(content, ensure_ascii=False)
                        else:
                            response_body = str(content)
                    else:
                        response_body = "[Response body not captured - JSONResponse content not accessible]"
                        
            elif hasattr(response, "body") and response.body:
                response_body = response.body.decode('utf-8')
        except Exception as e:
            response_body = f"Failed to extract response body: {str(e)}"
        
        # Add to session history
        session_handler.add_to_history(
            session_id,
            request_details,
            {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response_body
            }
        )
        
        return response

from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ..core.request_handler import RequestHandler
from ..core.route_matcher import RouteMatcher
from ..schemas.loader import SchemaLoader


class RouteMiddleware(BaseHTTPMiddleware):
    """Middleware for handling API routing and request processing."""
    
    def __init__(self, app, schema_loader: SchemaLoader,
                 response_model: str = "gpt-4.1-mini", state_model: str = "gpt-4.1-mini", validation_model: str = "gpt-4.1-mini"):
        """Initialize the route middleware.
        
        Args:
            app: The FastAPI application
            schema_loader: SchemaLoader instance for loading OpenAPI schemas
            response_model: LLM model for response generation
            state_model: LLM model for state update
            validation_model: LLM model for request validation
        """
        super().__init__(app)
        self.schema_loader = schema_loader
        self.route_matcher = RouteMatcher(schema_loader)
        self.response_model = response_model
        self.state_model = state_model
        self.validation_model = validation_model

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract API name and path
        api_name, remaining_path = self.route_matcher.extract_api_info(request)
        if not api_name:
            return await call_next(request)
        
        # Load schema
        schema = self.route_matcher.load_api_schema(api_name)
        if not schema:
            return JSONResponse(
                status_code=404,
                content={"detail": f"API schema not found for {api_name}"}
            )
        
        # Find matching endpoint (with api_name for fallback normalization)
        matching_path, matching_operation = self.route_matcher.find_matching_endpoint(
            remaining_path, schema, request.method.lower(), api_name
        )
        
        if not matching_path or not matching_operation:
            return JSONResponse(
                status_code=404,
                content={"detail": f"Endpoint not found: {request.method} {request.url.path}"}
            )
        
        # Handle the request
        request_handler = RequestHandler(schema, 
                                        response_model=self.response_model,
                                        state_model=self.state_model,
                                        validation_model=self.validation_model)
        
        # Store routing info in request state for other middlewares to use
        request.state.api_name = api_name
        request.state.matching_path = matching_path
        request.state.matching_operation = matching_operation
        request.state.request_handler = request_handler
        
        # Continue to next middleware (e.g., SessionMiddleware)
        response = await call_next(request)
        return response 

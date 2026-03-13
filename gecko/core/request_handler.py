import logging
from typing import Any, Dict

from fastapi import Request
from fastapi.responses import JSONResponse

from ..utils.request_validator import RequestValidator
from ..utils.response_generator import ResponseGenerator
from ..utils.response_validator import ResponseValidator


class RequestHandler:
    """Handles the processing of incoming requests and generation of responses."""
    
    def __init__(self, schema: Dict[str, Any],
                 response_model: str = "gpt-4.1-mini", state_model: str = "gpt-4.1-mini", validation_model: str = "gpt-4.1-mini"):
        """Initialize the request handler with an OpenAPI schema.
        Args:
            schema: Loaded OpenAPI schema
            response_model: LLM model for response generation
            state_model: LLM model for state update
            validation_model: LLM model for request validation
        """
        self.schema = schema
        self.validator = RequestValidator(schema, validation_model=validation_model)
        self.response_generator = ResponseGenerator(
            response_model=response_model,
            state_model=state_model,
            validation_model=validation_model,
        )
        self.response_validator = ResponseValidator(schema)
    
    async def handle_request(self, request: Request, matching_path: str, operation: Dict[str, Any], api_name: str) -> JSONResponse:
        """Process an incoming request and generate a response."""
        try:
            # Validate request
            is_valid, error_message = await self.validator.validate_request(request, matching_path, request.method.lower(), api_name)
            if not is_valid:
                logging.error(f"Request validation failed: {error_message}")
                return JSONResponse(
                    status_code=422,  # Use 422 Unprocessable Entity for validation errors
                    content={"detail": error_message}
                )
            
            # Generate response
            return await self._generate_response(request, operation)
                
        except Exception as e:
            logging.error(f"Request handling error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": f"Internal server error: {str(e)}"}
            )
    
    async def _generate_response(self, request: Request, operation: Dict[str, Any]) -> JSONResponse:
        """Generate a response based on the operation and schema."""
        responses = operation.get('responses', {})
        success_response = responses.get('200', responses.get('201', {}))
        
        if 'content' in success_response:
            content = success_response['content']
            if 'application/json' in content:
                response_schema = content['application/json'].get('schema', {})
                mock_response = await self.response_generator.generate_response(response_schema, self.schema, request, operation)
                
                # Validate the generated response
                is_valid, error_message = await self.response_validator.validate_response(request, mock_response)
                if not is_valid:
                    logging.error(f"Generated response validation failed: {error_message}")
                    return JSONResponse(
                        status_code=500,
                        content={"detail": f"Failed to generate valid response: {error_message}"}
                    )
                
                response = JSONResponse(content=mock_response)
                # Store the content for session middleware to access
                response._session_response_content = mock_response
                return response
        
        response = JSONResponse(content={})
        # Store the content for session middleware to access
        response._session_response_content = {}
        return response 

from fastapi import FastAPI

from ..handlers.session_handler import session_router
from ..middleware.route_middleware import RouteMiddleware
from ..middleware.session_middleware import SessionMiddleware
from ..schemas.loader import SchemaLoader
from ..schemas.global_loader import set_global_schema_loader
from ..utils.global_config import set_model_config


class GeckoServer:
    """Gecko server that processes requests according to OpenAPI schemas."""
    
    def __init__(
        self,
        schemas_dir: str = "data/openapi",
        response_model: str = "gpt-4.1-mini",
        state_model: str = "gpt-4.1-mini",
        validation_model: str = "gpt-4.1-mini",
    ):
        """Initialize Gecko with a directory containing OpenAPI schemas.
        Args:
            schemas_dir: Directory containing OpenAPI schemas
            response_model: LLM model for response generation (default: gpt-4.1-mini)
            state_model: LLM model for state update (default: gpt-4.1-mini)
            validation_model: LLM model for request validation (default: gpt-4.1-mini)
        """
        self.app = FastAPI()
        # Support providing multiple schema roots via list (from state/CLI settings).
        self.schema_loader = SchemaLoader(schemas_dir)
        # Expose globally for utilities that need schema context (introspection)
        try:
            set_global_schema_loader(self.schema_loader)
        except Exception:
            pass
        self.response_model = response_model
        self.state_model = state_model
        self.validation_model = validation_model

        # Set global model configuration for all components
        set_model_config(
            state_model=state_model,
            response_model=response_model,
            validation_model=validation_model,
        )
        
        # Include the session router
        self.app.include_router(session_router)
        
        # Add middlewares - order matters! SessionMiddleware must be added FIRST to execute LAST
        self.app.add_middleware(SessionMiddleware)
        self.app.add_middleware(RouteMiddleware, 
                               schema_loader=self.schema_loader, 
                               response_model=self.response_model,
                               state_model=self.state_model,
                               validation_model=self.validation_model)

    def run(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
        """Run Gecko.

        Args:
            host: Host to bind to
            port: Port to bind to
            workers: Number of worker processes for parallel request handling
        """
        import uvicorn
        uvicorn.run(self.app, host=host, port=port, workers=workers) 

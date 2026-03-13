"""
Tool Registry for managing real tools and mock tools.

This module provides a unified interface for registering and managing
both real tools (wrapped with RealToolWrapper) and mock tools (from OpenAPI).
"""

import importlib
import inspect
import logging
import re
import threading
from typing import Any, Callable, Dict, List, Optional

from .context import SessionContext
from .wrapper import RealToolWrapper, wrap_real_tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing real tools and mock tools.

    This registry provides a unified interface for:
    - Registering real tools (from toolkit instances or module paths)
    - Registering mock tools (from OpenAPI schemas)
    - Getting combined tool lists for agents

    Example:
        >>> registry = ToolRegistry()
        >>> # Register real tool from instance
        >>> registry.register_real_tool(airline_tools, "search_direct_flight")
        >>> # Register mock tools from OpenAPI
        >>> registry.register_mock_tools_from_openapi("schemas/airline.json")
        >>> # Get all tools for agent
        >>> tools = registry.get_all_tools()
    """

    def __init__(self):
        """Initialize empty tool registry."""
        self.real_tools: Dict[str, Callable] = {}
        self.mock_tools: Dict[str, Callable] = {}
        self._openapi_toolkits: List[Any] = []  # Store toolkit instances for session management
        self._session_lock = threading.RLock()  # Thread-safe session ID updates
        self._bound_session_context: Optional[SessionContext] = None

    def register_real_tool(
        self,
        tool_instance: Any,
        method_name: str,
        function_name: Optional[str] = None,
        sync_config: bool = True,
    ) -> None:
        """Register a real tool from a toolkit instance.

        Args:
            tool_instance: Toolkit instance (e.g., AirlineTools(db))
            method_name: Method name on the toolkit
            function_name: Name to register under (default: method_name)
            sync_config: Whether to sync config to mock server (default: True)

        Raises:
            AttributeError: If method doesn't exist on instance
            ValueError: If tool name already exists as mock tool

        Example:
            >>> from tau2.domains.airline.tools import AirlineTools
            >>> airline_tools = AirlineTools(db)
            >>> registry.register_real_tool(
            ...     airline_tools,
            ...     "search_direct_flight"
            ... )
        """
        # Get method from instance
        if not hasattr(tool_instance, method_name):
            raise AttributeError(
                f"Method '{method_name}' not found on {type(tool_instance).__name__}"
            )

        method = getattr(tool_instance, method_name)

        # Determine registration name
        name = function_name or method_name

        # Check for mock tool conflict
        if name in self.mock_tools:
            logger.info(
                f"Overriding mock tool '{name}' with real tool from {type(tool_instance).__name__}"
            )
            del self.mock_tools[name]

        # Wrap and register
        wrapped = wrap_real_tool(
            method,
            function_name=name,
            sync_config=sync_config,
            session_context=self._bound_session_context,
        )
        self.real_tools[name] = wrapped

    def register_real_tool_from_path(
        self,
        tool_path: str,
        tool_instance_args: Optional[tuple] = None,
        tool_instance_kwargs: Optional[Dict] = None,
        function_name: Optional[str] = None,
        sync_config: bool = True,
    ) -> None:
        """Register a real tool from a module path.

        Path format: "module.path:ClassName.method_name"

        Args:
            tool_path: Path to tool (e.g., "tau2.domains.airline.tools:AirlineTools.search_direct_flight")
            tool_instance_args: Args for toolkit instantiation (if needed)
            tool_instance_kwargs: Kwargs for toolkit instantiation (if needed)
            function_name: Name to register under (default: from path)
            sync_config: Whether to sync config to mock server (default: True)

        Raises:
            ValueError: If path format is invalid
            ImportError: If module cannot be imported
            AttributeError: If class/method not found

        Example:
            >>> registry.register_real_tool_from_path(
            ...     "tau2.domains.airline.tools:AirlineTools.search_direct_flight",
            ...     tool_instance_kwargs={"db": airline_db}
            ... )

        Note:
            This method requires the toolkit class to be instantiable.
            For complex initialization, use register_real_tool() instead.
        """
        # Parse path
        if ":" not in tool_path:
            raise ValueError(
                f"Invalid tool path format: {tool_path}. "
                "Expected 'module.path:ClassName.method_name'"
            )

        module_path, obj_path = tool_path.split(":", 1)

        # Validate object path format before importing
        if "." not in obj_path:
            raise ValueError(
                f"Invalid object path: {obj_path}. Expected 'ClassName.method_name'"
            )

        # Import module
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(f"Failed to import module {module_path}: {e}")

        parts = obj_path.split(".")
        class_name = parts[0]
        method_name = ".".join(parts[1:])

        # Get class
        if not hasattr(module, class_name):
            raise AttributeError(f"Class '{class_name}' not found in {module_path}")

        tool_class = getattr(module, class_name)

        # Instantiate toolkit
        instance_args = tool_instance_args or ()
        instance_kwargs = tool_instance_kwargs or {}

        try:
            tool_instance = tool_class(*instance_args, **instance_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate {class_name}: {e}. "
                "Consider using register_real_tool() with pre-instantiated toolkit."
            )

        # Register using instance method
        self.register_real_tool(
            tool_instance, method_name, function_name=function_name, sync_config=sync_config
        )

    def register_mock_tools_from_openapi(
        self, openapi_schema_path: str, base_url: Optional[str] = None
    ) -> None:
        """Register mock tools from OpenAPI schema.

        Creates tools using CAMEL's OpenAPIToolkit. The toolkit instance is stored
        internally to allow dynamic session_id updates via set_session_id().

        Args:
            openapi_schema_path: Path to OpenAPI schema JSON file
            base_url: Base URL for the API (optional)

        Raises:
            ImportError: If CAMEL is not installed
            FileNotFoundError: If schema file not found

        Example:
            >>> registry.register_mock_tools_from_openapi(
            ...     "data/bfcl_v4/openapi/gorilla_file_system.json",
            ...     base_url="http://localhost:8000"
            ... )
        """
        try:
            from camel.toolkits import OpenAPIToolkit
        except ImportError:
            raise ImportError(
                "CAMEL is required for OpenAPI tool registration. "
                "Install with: pip install camel-ai"
            )

        # Create OpenAPI toolkit and parse spec
        try:
            from camel.toolkits import FunctionTool

            toolkit = OpenAPIToolkit()
            if base_url:
                toolkit.set_override_server_url(base_url)

            # Parse the OpenAPI specification
            openapi_spec = toolkit.parse_openapi_file(openapi_schema_path)
            if openapi_spec is None:
                raise RuntimeError("Failed to parse OpenAPI specification")

            # Get API name from spec and sanitize it for OpenAI tool names
            raw_api_name = openapi_spec.get('info', {}).get('title', 'API')
            api_name = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_api_name)

            # Generate functions and schemas
            funcs = toolkit.generate_openapi_funcs(api_name, openapi_spec)
            schemas = toolkit.openapi_spec_to_openai_schemas(api_name, openapi_spec)

            # Create FunctionTool objects
            tool_functions = [
                FunctionTool(func=func, openai_tool_schema=schema)
                for func, schema in zip(funcs, schemas)
            ]

            # Wrap mock tools to flush buffer before execution
            def create_flush_wrapper(func):
                def wrapper(*args, **kwargs):
                    # Flush any pending real tool updates before executing mock tool
                    context = self._bound_session_context
                    if context:
                        context.flush_buffer()
                    return func(*args, **kwargs)
                return wrapper

            for tool in tool_functions:
                # Wrap the underlying function
                original_func = tool.func
                tool.func = create_flush_wrapper(original_func)

            # Store toolkit instance for session management (thread-safe)
            with self._session_lock:
                self._openapi_toolkits.append(toolkit)
            logger.debug(f"Stored OpenAPIToolkit instance for {api_name}")

        except Exception as e:
            raise RuntimeError(f"Failed to create OpenAPIToolkit: {e}")

        # Register all tools
        try:
            for tool_func in tool_functions:
                # Get tool name from FunctionTool
                try:
                    tool_name = tool_func.get_function_name()
                except Exception:
                    # Fallback: try to extract from schema
                    try:
                        schema = tool_func.openai_tool_schema
                        if 'function' in schema:
                            tool_name = schema['function']['name']
                        else:
                            tool_name = schema.get('name')
                    except Exception:
                        logger.warning(f"Cannot determine name for tool: {type(tool_func)}")
                        continue

                # Skip if real tool already registered with same name
                if tool_name in self.real_tools:
                    logger.debug(
                        f"Skipping mock tool '{tool_name}' (real tool already registered)"
                    )
                    continue

                self.mock_tools[tool_name] = tool_func
                logger.debug(f"Registered mock tool: {tool_name}")

        except Exception as e:
            logger.warning(f"Failed to extract tools from OpenAPIToolkit: {e}")

    def get_real_tools(self) -> List[Callable]:
        """Get list of all registered real tools.

        Returns:
            List of wrapped real tool functions
        """
        return list(self.real_tools.values())

    def get_mock_tools(self) -> List[Callable]:
        """Get list of all registered mock tools.

        Returns:
            List of mock tool functions
        """
        return list(self.mock_tools.values())

    def get_all_tools(self) -> List[Callable]:
        """Get combined list of all tools (real + mock).

        Real tools take precedence over mock tools with the same name.

        Returns:
            List of all tool functions (real + mock)

        Example:
            >>> tools = registry.get_all_tools()
            >>> agent = ChatAgent(system_message, tools=tools)
        """
        all_tools = []
        all_tools.extend(self.real_tools.values())
        all_tools.extend(self.mock_tools.values())
        return all_tools

    def get_tool_names(self) -> Dict[str, str]:
        """Get mapping of tool names to their types.

        Returns:
            Dict mapping tool name to type ("real" or "mock")

        Example:
            >>> names = registry.get_tool_names()
            >>> print(names)
            {'search_flight': 'real', 'book_reservation': 'mock'}
        """
        names = {}
        for name in self.real_tools:
            names[name] = "real"
        for name in self.mock_tools:
            names[name] = "mock"
        return names

    def set_session_id(self, session_id: str) -> None:
        """Set session ID for all OpenAPI toolkits (thread-safe).

        This updates the session_id on all registered OpenAPIToolkit instances.
        The session ID will be automatically included in the X-Session-ID header
        for all subsequent mock tool calls.

        Thread-safe: Uses internal lock to prevent concurrent modifications.

        Args:
            session_id: Session ID to set

        Example:
            >>> registry.set_session_id("abc-123-def")
            >>> # All mock tool calls will now include X-Session-ID: abc-123-def
        """
        with self._session_lock:
            for toolkit in self._openapi_toolkits:
                toolkit.set_session_id(session_id)
            logger.debug(f"Set session_id={session_id} for {len(self._openapi_toolkits)} OpenAPI toolkits")

    def _bind_context_to_real_tools(self, session_context: Optional[SessionContext]) -> None:
        """Bind sync context to all registered real wrappers."""
        for tool in self.real_tools.values():
            if isinstance(tool, RealToolWrapper):
                tool.bind_session_context(session_context)

    def bind_session_context(
        self,
        session_id: str,
        mock_server_url: str,
        task_id: Optional[str] = None,
        buffer_updates: bool = True,
    ) -> SessionContext:
        """Create and bind one explicit session context for all real wrappers."""
        with self._session_lock:
            context = SessionContext(
                session_id=session_id,
                mock_server_url=mock_server_url,
                task_id=task_id,
                buffer_updates=buffer_updates,
            )
            self._bound_session_context = context
            self._bind_context_to_real_tools(context)
            return context

    def get_session_context(self) -> Optional[SessionContext]:
        """Get currently bound explicit session context."""
        return self._bound_session_context

    def flush_session_buffer(self) -> Optional[Dict[str, Any]]:
        """Flush buffered updates from bound session context if present."""
        context = self._bound_session_context
        if not context:
            return None
        if hasattr(context, "maybe_flush_buffer"):
            return context.maybe_flush_buffer()
        return context.flush_buffer()

    def clear_session_context(self) -> None:
        """Unbind session context from all real wrappers."""
        with self._session_lock:
            self._bound_session_context = None
            self._bind_context_to_real_tools(None)

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool exists (real or mock)
        """
        return tool_name in self.real_tools or tool_name in self.mock_tools

    def get_tool_type(self, tool_name: str) -> Optional[str]:
        """Get the type of a registered tool.

        Args:
            tool_name: Name of the tool

        Returns:
            "real", "mock", or None if not found
        """
        if tool_name in self.real_tools:
            return "real"
        elif tool_name in self.mock_tools:
            return "mock"
        return None

    def clear(self) -> None:
        """Clear all registered tools."""
        self.clear_session_context()
        self.real_tools.clear()
        self.mock_tools.clear()
        logger.debug("Cleared all tools from registry")

    def __len__(self) -> int:
        """Get total number of registered tools."""
        return len(self.real_tools) + len(self.mock_tools)

    def __repr__(self) -> str:
        """String representation of the registry."""
        return (
            f"ToolRegistry(real_tools={len(self.real_tools)}, "
            f"mock_tools={len(self.mock_tools)})"
        )

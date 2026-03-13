"""
Real Tool Wrapper

Wraps real tool functions to automatically sync execution results to mock server.
Now inherits from CAMEL's FunctionTool for compatibility with ChatAgent.
"""

import inspect
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import requests

from camel.toolkits import FunctionTool

from .context import SessionContext

logger = logging.getLogger(__name__)


def load_predefined_schema(
    function_name: str, schema_dir: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Load pre-defined OpenAI function schema for a function.

    Checks for pre-defined OpenAPI schema files and extracts the function schema
    if available. This allows bypassing auto-generation for known problematic cases.

    Args:
        function_name: Name of the function to load schema for
        schema_dir: Directory to search for schema files.
            If not provided, searches:
            1) data/tau2/openapi
            2) data/bfcl_v4/openapi/multi_turn/real

    Returns:
        OpenAI function schema dict if found, None otherwise
    """
    if schema_dir is None:
        repo_root = Path(__file__).parent.parent.parent
        schema_dirs = [
            repo_root / "data" / "tau2" / "openapi",
            repo_root / "data" / "bfcl_v4" / "openapi" / "multi_turn" / "real",
        ]
    else:
        schema_dirs = [Path(schema_dir)]

    for current_dir in schema_dirs:
        if not current_dir.exists():
            continue

        # Look for schema files (could be named after domain or specific to function)
        schema_files = sorted(current_dir.glob("*RealAPI.json"))

        for schema_file in schema_files:
            try:
                with open(schema_file, "r") as f:
                    openapi_spec = json.load(f)

                # Extract function schema from OpenAPI spec
                paths = openapi_spec.get("paths", {})
                endpoint = f"/{function_name}"

                if endpoint in paths:
                    # Convert OpenAPI endpoint to OpenAI function schema
                    path_item = paths[endpoint]
                    operation = path_item.get("post", {})

                    # Extract parameters from requestBody
                    request_body = operation.get("requestBody", {})
                    content = request_body.get("content", {})
                    json_schema = content.get("application/json", {}).get("schema", {})

                    # Build OpenAI function schema
                    schema = {
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "description": operation.get(
                                "description", operation.get("summary", "")
                            ),
                            "strict": True,
                            "parameters": json_schema,
                        },
                    }

                    logger.debug(
                        "[SCHEMA LOADER] Loaded pre-defined schema for %s from %s",
                        function_name,
                        schema_file.name,
                    )
                    return schema

            except Exception as e:
                logger.debug(
                    "[SCHEMA LOADER] Failed to load schema from %s: %s",
                    schema_file,
                    e,
                )
                continue

    return None


def fix_openai_schema(schema: Dict[str, Any], depth: int = 0, path: str = "") -> Dict[str, Any]:
    """Fix OpenAPI schema to comply with OpenAI's strict requirements.

    OpenAI requires all object types to have 'additionalProperties' set to false.
    This function recursively adds this field to all object types in the schema.

    Args:
        schema: Original schema from CAMEL's get_openai_tool_schema
        depth: Current recursion depth (for debugging)
        path: Current path in schema tree (for debugging)

    Returns:
        Fixed schema with additionalProperties added to all objects
    """
    if not isinstance(schema, dict):
        return schema

    # Create a copy to avoid modifying the original
    fixed = schema.copy()

    # OpenAI rejects array schemas without explicit items.
    if fixed.get("type") == "array" and "items" not in fixed:
        logger.debug(
            "[SCHEMA FIX] Adding fallback items schema at path: %s",
            path or "root",
        )
        fixed["items"] = {
            "anyOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "boolean"},
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ]
        }

    # If this is an object type, ensure it has additionalProperties: false
    # An object can be identified by either:
    # 1. Explicit "type": "object"
    # 2. Having "properties" field (implicit object)
    is_object = (
        fixed.get("type") == "object" or
        ("properties" in fixed and isinstance(fixed.get("properties"), dict))
    )

    # OpenAI strict mode requires additionalProperties: false for ALL objects
    # We need to both add missing fields AND correct any true values to false
    if is_object:
        current_value = fixed.get("additionalProperties")
        if current_value is None:
            logger.debug(
                "[SCHEMA FIX] Adding additionalProperties=false at path: %s (type=%s)",
                path or "root",
                fixed.get("type", "implicit"),
            )
            fixed["additionalProperties"] = False
        elif current_value is True:
            logger.debug(
                "[SCHEMA FIX] Correcting additionalProperties from true to false at path: %s (type=%s)",
                path or "root",
                fixed.get("type", "implicit"),
            )
            fixed["additionalProperties"] = False

        # OpenAI strict validation also requires required[] to include every key in properties.
        # If optional fields exist in source OpenAPI, we promote them to required for strict mode.
        props = fixed.get("properties")
        if isinstance(props, dict):
            prop_keys = list(props.keys())
            required = fixed.get("required")
            if not isinstance(required, list):
                fixed["required"] = prop_keys
            else:
                # Preserve existing order, then append missing keys deterministically.
                merged_required = list(required)
                for key in prop_keys:
                    if key not in merged_required:
                        merged_required.append(key)
                fixed["required"] = merged_required

    # Recursively fix nested schemas
    # Handle 'function' field (OpenAI tool wrapper)
    if "function" in fixed and isinstance(fixed["function"], dict):
        fixed["function"] = fix_openai_schema(fixed["function"], depth+1, f"{path}.function")

    # Handle '$defs' field (JSON Schema definitions)
    if "$defs" in fixed and isinstance(fixed["$defs"], dict):
        logger.debug(
            "[SCHEMA FIX] Processing $defs at path: %s with %s definitions",
            path or "root",
            len(fixed["$defs"]),
        )
        fixed["$defs"] = {
            k: fix_openai_schema(v, depth+1, f"{path}.$defs.{k}")
            for k, v in fixed["$defs"].items()
        }

    # Handle 'definitions' field (alternative name)
    if "definitions" in fixed and isinstance(fixed["definitions"], dict):
        fixed["definitions"] = {
            k: fix_openai_schema(v, depth+1, f"{path}.definitions.{k}")
            for k, v in fixed["definitions"].items()
        }

    # Handle 'parameters' field (function parameters)
    if "parameters" in fixed and isinstance(fixed["parameters"], dict):
        fixed["parameters"] = fix_openai_schema(fixed["parameters"], depth+1, f"{path}.parameters")

    # Handle 'properties' field (object properties)
    if "properties" in fixed and isinstance(fixed["properties"], dict):
        fixed["properties"] = {
            k: fix_openai_schema(v, depth+1, f"{path}.properties.{k}")
            for k, v in fixed["properties"].items()
        }

    # Handle 'items' field (array items)
    if "items" in fixed:
        fixed["items"] = fix_openai_schema(fixed["items"], depth+1, f"{path}.items")

    # Handle 'anyOf' field (union types)
    if "anyOf" in fixed and isinstance(fixed["anyOf"], list):
        logger.debug(
            "[SCHEMA FIX] Processing anyOf at path: %s with %s options",
            path or "root",
            len(fixed["anyOf"]),
        )
        fixed["anyOf"] = [
            fix_openai_schema(item, depth+1, f"{path}.anyOf[{i}]")
            for i, item in enumerate(fixed["anyOf"])
        ]

    # Handle 'allOf' field
    if "allOf" in fixed and isinstance(fixed["allOf"], list):
        fixed["allOf"] = [
            fix_openai_schema(item, depth+1, f"{path}.allOf[{i}]")
            for i, item in enumerate(fixed["allOf"])
        ]

    # Handle 'oneOf' field
    if "oneOf" in fixed and isinstance(fixed["oneOf"], list):
        fixed["oneOf"] = [
            fix_openai_schema(item, depth+1, f"{path}.oneOf[{i}]")
            for i, item in enumerate(fixed["oneOf"])
        ]

    return fixed


class RealToolWrapper(FunctionTool):
    """Wrapper for real tool functions.

    Wraps a real tool function to:
    1. Execute the real function
    2. Serialize the result
    3. Sync config to mock server
    4. Return the original result

    Inherits from CAMEL's FunctionTool for compatibility with ChatAgent.
    """

    def __init__(
        self,
        real_function: Callable,
        function_name: Optional[str] = None,
        sync_config: bool = True,
        openai_tool_schema: Optional[Dict[str, Any]] = None,
        argument_normalizer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        expected_session_id: Optional[str] = None,
        strict_sync: bool = False,
        session_context: Optional[SessionContext] = None,
    ):
        """Initialize the wrapper.

        Args:
            real_function: The real tool function to wrap
            function_name: Function name for config sync (default: from function)
            sync_config: Whether to sync config to mock server (default: True)
            openai_tool_schema: Optional OpenAPI tool schema (auto-generated if None)
            expected_session_id: Optional bound session id expected for sync
            strict_sync: Whether sync failures should fail tool execution
            session_context: Explicit bound session context for sync
        """
        # Store RealToolWrapper-specific attributes BEFORE calling super().__init__
        self.real_function = real_function
        self.function_name = function_name or real_function.__name__
        self.sync_config = sync_config
        self.argument_normalizer = argument_normalizer
        self.expected_session_id = expected_session_id
        self.strict_sync = strict_sync
        self._session_context: Optional[SessionContext] = session_context

        # If no schema provided, load pre-defined schema
        if openai_tool_schema is None:
            openai_tool_schema = load_predefined_schema(self.function_name)
            if openai_tool_schema is None:
                raise ValueError(
                    f"No predefined schema found for real tool '{self.function_name}'. "
                    "Add it to a *RealAPI.json file under data/tau2/openapi or "
                    "data/bfcl_v4/openapi/multi_turn/real, or pass openai_tool_schema explicitly."
                )

        # Ensure schemas comply with OpenAI strict requirements
        openai_tool_schema = fix_openai_schema(openai_tool_schema)

        # Create wrapper function that includes sync logic
        def _wrapped_function(*args, **kwargs):
            return self._execute_with_sync(*args, **kwargs)

        # IMPORTANT: Manually copy safe attributes instead of using @functools.wraps
        # to avoid issues with complex type annotations (Union, Optional, etc.)
        # that cause issubclass() errors in CAMEL's FunctionTool
        _wrapped_function.__name__ = real_function.__name__
        _wrapped_function.__doc__ = real_function.__doc__
        # Do NOT copy __annotations__ or __wrapped__ to avoid signature inspection issues

        # Sanitize schema to satisfy OpenAI function requirements
        if openai_tool_schema and isinstance(openai_tool_schema, dict):
            fn_schema = openai_tool_schema.setdefault("function", {})
            parameters = fn_schema.setdefault("parameters", {"type": "object"})
            if not isinstance(parameters, dict):
                parameters = {"type": "object"}
                fn_schema["parameters"] = parameters
            parameters.setdefault("type", "object")
            parameters.setdefault("properties", {})

        # Initialize FunctionTool parent with the pre-generated schema
        # This prevents FunctionTool from inspecting the wrapper's signature
        super().__init__(
            func=_wrapped_function,
            openai_tool_schema=openai_tool_schema,
        )

        # Preserve metadata so inspect/signature tooling keeps working
        self.__name__ = _wrapped_function.__name__
        self.__doc__ = _wrapped_function.__doc__
        try:
            self.__signature__ = inspect.signature(real_function)
        except (TypeError, ValueError):
            # Some callables (e.g., builtins) may not expose signatures
            pass

    def bind_session_context(self, session_context: Optional[SessionContext]) -> None:
        """Bind an explicit sync context to this wrapper instance."""
        self._session_context = session_context

    def _execute_with_sync(self, *args, **kwargs) -> Any:
        """Execute the wrapped function with config sync.

        Flow:
        1. Call real function
        2. Serialize result
        3. Sync to mock server (if enabled)
        4. Return original result

        Returns:
            The result from the real function

        Raises:
            Any exception from the real function (propagated to agent)
        """
        normalized_args = args
        normalized_kwargs = kwargs
        if self.argument_normalizer and not args and isinstance(kwargs, dict):
            normalized_kwargs = self.argument_normalizer(dict(kwargs))

        # Step 1: Execute real function
        try:
            result = self.real_function(*normalized_args, **normalized_kwargs)
        except Exception as e:
            # Let exceptions propagate to agent
            logger.debug(f"Real tool {self.function_name} raised exception: {e}")
            raise

        # Normalize successful no-content returns to an explicit success signal.
        # Many BFCL real tools return None on success (e.g., echo/write-like ops),
        # which can make agent-side completion detection ambiguous.
        if result is None:
            result = {"success": True}

        # Step 2 & 3: Sync config (if enabled and context available)
        if self.sync_config:
            try:
                self._sync_config_to_mock_server(
                    arguments=self._extract_arguments(normalized_args, normalized_kwargs),
                    result=result,
                )
            except Exception as e:
                if self.strict_sync:
                    raise
                logger.warning(
                    f"Failed to sync config for {self.function_name}: {e}",
                    exc_info=True,
                )

        # Step 4: Return original result
        return result

    def _extract_arguments(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract function arguments from args and kwargs.

        Uses inspect.signature to bind arguments correctly.

        Args:
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Dictionary of argument names to values
        """
        # CAMEL may invoke wrapped tools with envelope kwargs like:
        # {"_tool_name": "...", "kwargs": {...}}.
        # Normalize this shape so sync payload carries real function arguments.
        if isinstance(kwargs, dict):
            nested_kwargs = kwargs.get("kwargs")
            if isinstance(nested_kwargs, dict):
                return dict(nested_kwargs)

        if len(args) == 1 and isinstance(args[0], dict):
            arg0 = args[0]
            nested_kwargs = arg0.get("kwargs")
            if isinstance(nested_kwargs, dict):
                return dict(nested_kwargs)

        try:
            sig = inspect.signature(self.real_function)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            return dict(bound.arguments)
        except Exception as e:
            logger.warning(f"Failed to extract arguments: {e}")
            # Fallback: just use kwargs
            return kwargs

    def _serialize_result(self, result: Any) -> Any:
        """Serialize result for JSON transmission.

        Handles:
        - Pydantic v2 models (model_dump)
        - Pydantic v1 models (dict)
        - Dataclasses (asdict)
        - Lists/dicts (recursive)
        - Primitive types
        - Fallback: str() for unknown types

        Args:
            result: Result to serialize

        Returns:
            JSON-serializable result
        """
        # Handle None
        if result is None:
            return None

        # Primitive types (int, str, float, bool)
        if isinstance(result, (int, str, float, bool)):
            return result

        # Pydantic v2
        if hasattr(result, "model_dump"):
            try:
                return result.model_dump()
            except Exception as e:
                logger.warning(f"model_dump() failed: {e}, trying dict()")

        # Pydantic v1
        if hasattr(result, "dict") and callable(getattr(result, "dict")):
            try:
                return result.dict()
            except Exception as e:
                logger.warning(f"dict() method failed: {e}")

        # Dataclass
        if hasattr(result, "__dataclass_fields__"):
            try:
                from dataclasses import asdict
                return asdict(result)
            except Exception as e:
                logger.warning(f"asdict() failed: {e}")

        # List
        if isinstance(result, list):
            return [self._serialize_result(item) for item in result]

        # Tuple (convert to list)
        if isinstance(result, tuple):
            return [self._serialize_result(item) for item in result]

        # Dict
        if isinstance(result, dict):
            return {k: self._serialize_result(v) for k, v in result.items()}

        # Fallback: convert to string
        # This handles any unknown types gracefully
        logger.warning(
            f"Unknown type {type(result).__name__} for result, converting to string"
        )
        return str(result)

    def _sync_config_to_mock_server(
        self, arguments: Dict[str, Any], result: Any
    ) -> None:
        """Sync execution result to mock server.

        Posts tool call to mock server's /update-state-from-real endpoint.

        Args:
            arguments: Function arguments
            result: Function result

        Raises:
            RuntimeError: If config sync fails
        """
        # Use explicitly bound session context (no global lookup).
        context = self._session_context
        if not context:
            raise RuntimeError(
                f"Missing bound session context for real tool sync: {self.function_name}"
            )
        if (
            self.expected_session_id
            and context.session_id != self.expected_session_id
        ):
            raise RuntimeError(
                f"Session mismatch for {self.function_name}: "
                f"expected={self.expected_session_id}, actual={context.session_id}"
            )
        logger.debug(f"[WRAPPER DEBUG] Syncing {self.function_name} to {context.mock_server_url}, session={context.session_id}")

        # Serialize result
        try:
            serialized_result = self._serialize_result(result)
        except Exception as e:
            logger.warning(f"Failed to serialize result: {e}")
            serialized_result = str(result)

        # Construct tool call for Real Tool sync
        tool_call = {
            "name": self.function_name,
            "arguments": arguments,
            "result": serialized_result,
        }

        # Check if buffering is enabled
        if context.buffer_updates:
            context.add_to_buffer(tool_call)
            logger.debug(f"[WRAPPER DEBUG] Buffered config update for {self.function_name}")
            return

        # Post to mock server
        url = f"{context.mock_server_url}/update-state-from-real"

        try:
            payload = {"tool_call": tool_call}
            response = requests.post(
                url,
                json=payload,
                headers={"X-Session-ID": context.session_id},
                timeout=60,  # increased from 30s to reduce sync timeouts
            )
            response.raise_for_status()
            logger.debug(
                f"Successfully synced config for {self.function_name} "
                f"(session: {context.session_id})"
            )
        except requests.RequestException as e:
            try:
                payload_len = len(json.dumps(payload))
                args_len = len(json.dumps(tool_call.get("arguments", {})))
                result_len = len(json.dumps(tool_call.get("result", {})))
                task_id = getattr(context, "task_id", None) if context else None
                logger.warning(
                    "Sync to mock server timed out for %s (task_id=%s, payload_len=%s, args_len=%s, result_len=%s)",
                    self.function_name,
                    task_id,
                    payload_len,
                    args_len,
                    result_len,
                )
            except Exception:
                logger.warning("Sync to mock server timed out for %s (failed to compute payload size)", self.function_name)
            logger.error("Sync failed for %s: %s", self.function_name, e)
            raise RuntimeError(f"Failed to sync config to mock server: {e}")


def wrap_real_tool(
    real_function: Callable,
    function_name: Optional[str] = None,
    sync_config: bool = True,
    openai_tool_schema: Optional[Dict[str, Any]] = None,
    argument_normalizer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    expected_session_id: Optional[str] = None,
    strict_sync: bool = False,
    session_context: Optional[SessionContext] = None,
) -> Callable:
    """Wrap a real tool function for config synchronization.

    Args:
        real_function: The real tool function to wrap
        function_name: Function name for config sync (optional)
        sync_config: Whether to sync config (default: True)
        openai_tool_schema: Optional OpenAI tool schema override
        argument_normalizer: Optional callable to normalize kwargs before execution/sync
        expected_session_id: Optional bound session id expected for sync
        strict_sync: Whether sync failures should fail tool execution
        session_context: Explicit bound session context for sync

    Returns:
        Wrapped function that can be used by agents

    Example:
        >>> from tau2.domains.airline.tools import AirlineTools
        >>> airline_tools = AirlineTools(db)
        >>>
        >>> # Wrap a real tool
        >>> wrapped_search = wrap_real_tool(
        ...     airline_tools.search_direct_flight,
        ...     function_name="search_direct_flight"
        ... )
        >>>
        >>> # Use in agent
        >>> tools = [wrapped_search, mock_book_tool, ...]
        >>> agent = ChatAgent(system_message, tools=tools)
    """
    return RealToolWrapper(
        real_function,
        function_name,
        sync_config,
        openai_tool_schema=openai_tool_schema,
        argument_normalizer=argument_normalizer,
        expected_session_id=expected_session_id,
        strict_sync=strict_sync,
        session_context=session_context,
    )

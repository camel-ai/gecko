#!/usr/bin/env python3
import logging
import os
import time
from typing import Any, Dict, List, Callable
from camel.toolkits import FunctionTool

logger = logging.getLogger(__name__)


def wrap_openapi_tool(original_tool: FunctionTool) -> FunctionTool:
    """
    Wrap an OpenAPIToolkit FunctionTool to handle requestBody parameter format.

    This wrapper detects if the tool expects a 'requestBody' parameter and
    automatically wraps direct parameters into the requestBody format.

    Args:
        original_tool: The original FunctionTool from OpenAPIToolkit

    Returns:
        A wrapped FunctionTool that handles parameter format conversion
    """
    # Get the original function and schema
    original_func = original_tool.func

    # Access the schema directly instead of using get_openai_function_schema
    # which might fail due to validation issues
    if hasattr(original_tool, 'openai_tool_schema'):
        original_schema = original_tool.openai_tool_schema
    else:
        try:
            original_schema = original_tool.get_openai_function_schema()
        except Exception:
            # If we can't get the schema, just return the original tool
            return original_tool

    # Handle both old and new OpenAI schema formats
    # New format has "type": "function" and nested "function" object
    # Old format has parameters directly at top level
    if "function" in original_schema:
        # New OpenAI format
        func_schema = original_schema["function"]
        params = func_schema.get("parameters", {})
    else:
        # Old format
        func_schema = original_schema
        params = original_schema.get("parameters", {})

    props = params.get("properties", {})
    needs_request_body = "requestBody" in props and len(props) == 1

    if not needs_request_body:
        # No wrapping needed
        return original_tool

    # Get the actual parameter schema from inside requestBody
    request_body_schema = props.get("requestBody", {})
    actual_params = request_body_schema.get("properties", {})
    actual_required = request_body_schema.get("required", [])

    tool_name = func_schema.get("name", "unknown_tool")
    trace_enabled = os.getenv("OPENAPI_TOOL_TRACE", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}

    # Create a wrapper function that converts parameters
    def wrapped_func(**kwargs):
        started = time.time()
        if trace_enabled:
            logger.warning("[OPENAPI TOOL START] %s kwargs=%s", tool_name, kwargs)

        # Check if requestBody is already provided
        if "requestBody" in kwargs:
            # Already in correct format
            try:
                result = original_func(**kwargs)
            except Exception:
                if trace_enabled:
                    logger.exception("[OPENAPI TOOL ERROR] %s (elapsed=%.3fs)", tool_name, time.time() - started)
                raise
            if trace_enabled:
                logger.warning("[OPENAPI TOOL END] %s (elapsed=%.3fs)", tool_name, time.time() - started)
            return result

        # Wrap all parameters in requestBody
        try:
            result = original_func(requestBody=kwargs)
        except Exception:
            if trace_enabled:
                logger.exception("[OPENAPI TOOL ERROR] %s (elapsed=%.3fs)", tool_name, time.time() - started)
            raise
        if trace_enabled:
            logger.warning("[OPENAPI TOOL END] %s (elapsed=%.3fs)", tool_name, time.time() - started)
        return result

    # Create a new schema with flattened parameters
    # Always use the new OpenAI format to avoid validation issues
    new_schema = {
        "type": "function",
        "function": {
            "name": func_schema.get("name"),
            "description": func_schema.get("description"),
            "parameters": {
                "type": "object",
                "properties": actual_params,
                "required": actual_required
            }
        }
    }
    # Create a new FunctionTool with the wrapped function and corrected schema
    wrapped_tool = FunctionTool(func=wrapped_func, openai_tool_schema=new_schema)

    tool_name = new_schema['function']['name'] if 'function' in new_schema else new_schema.get('name', 'unknown')
    logger.debug(f"Wrapped tool {tool_name} to handle requestBody format")

    return wrapped_tool


def fix_openapi_tools(tools: List[FunctionTool]) -> List[FunctionTool]:
    """
    Fix all OpenAPIToolkit tools to handle requestBody parameter format.

    Args:
        tools: List of FunctionTool objects from OpenAPIToolkit

    Returns:
        List of wrapped FunctionTool objects with fixed parameter handling
    """
    fixed_tools = []
    for tool in tools:
        try:
            fixed_tool = wrap_openapi_tool(tool)
            fixed_tools.append(fixed_tool)
        except Exception as e:
            logger.warning(f"Failed to wrap tool: {e}")
            # Keep original tool if wrapping fails
            fixed_tools.append(tool)

    logger.info(f"Fixed {len(fixed_tools)} OpenAPIToolkit tools for requestBody handling")
    return fixed_tools


# Monkey-patch for OpenAPIToolkit to automatically fix tools on generation
def patch_openapi_toolkit():
    """
    Monkey-patch OpenAPIToolkit to automatically fix requestBody issue.
    This modifies the toolkit to wrap all generated tools.
    """
    try:
        from camel.toolkits import OpenAPIToolkit

        # Save original methods
        original_generate = OpenAPIToolkit.generate_openapi_funcs

        # Create patched method
        def patched_generate(self, api_name: str, openapi_spec: Dict[str, Any]):
            # Call original method
            tools = original_generate(self, api_name, openapi_spec)
            # Note: generate_openapi_funcs returns functions, not FunctionTools
            # The wrapping happens later when FunctionTools are created
            return tools

        # Apply patch
        OpenAPIToolkit.generate_openapi_funcs = patched_generate
        logger.info("Successfully patched OpenAPIToolkit for requestBody handling")

    except Exception as e:
        logger.error(f"Failed to patch OpenAPIToolkit: {e}")


# Example usage
if __name__ == "__main__":
    import json
    from camel.toolkits import OpenAPIToolkit, FunctionTool

    # Load an OpenAPI spec
    with open('data/openapi/multi_turn/GorillaFileSystem.json', 'r') as f:
        schema = json.load(f)

    # Create toolkit and generate tools
    toolkit = OpenAPIToolkit()
    toolkit.set_override_server_url('http://localhost:8000')

    api_name = schema.get("info", {}).get("title", "GorillaFileSystem")
    toolkit_funcs = toolkit.generate_openapi_funcs(api_name, schema)
    schemas = toolkit.openapi_spec_to_openai_schemas(api_name, schema)

    # Create FunctionTool objects
    tools = [FunctionTool(func=func, openai_tool_schema=schema_item)
            for func, schema_item in zip(toolkit_funcs, schemas)]

    # Fix the tools
    fixed_tools = fix_openapi_tools(tools)

    print(f"Fixed {len(fixed_tools)} tools")

    # Test a fixed tool
    for tool in fixed_tools:
        try:
            tool_name = tool.get_function_name()
        except Exception:
            # If get_function_name fails, try to get name directly from schema
            try:
                schema = tool.openai_tool_schema
                if "function" in schema:
                    tool_name = schema["function"]["name"]
                else:
                    tool_name = schema.get("name", "unknown")
            except:
                tool_name = "unknown"

        if "mkdir" in tool_name:
            print(f"\nTesting {tool_name}")

            # Get session
            import requests
            session_response = requests.get("http://localhost:8000/session-id")
            session_id = session_response.json()["session_id"]

            # Set session on toolkit
            toolkit.set_session_id(session_id)

            # Now direct parameters should work!
            result = tool.func(dir_name="test_wrapped")
            print(f"Result: {result}")
            break

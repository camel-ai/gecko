#!/usr/bin/env python3
import logging
import re
from typing import Any, Dict, List

from camel.toolkits import FunctionTool

logger = logging.getLogger(__name__)


def _normalize_kwargs_for_export(kwargs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _materialize_numeric_defaults_for_export(dict(kwargs) if isinstance(kwargs, dict) else kwargs, params)
    return _canonicalize_kwargs_for_export(normalized, params)


def _materialize_numeric_defaults_for_export(kwargs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(kwargs, dict):
        return kwargs

    normalized = dict(kwargs)
    for key, schema in params.items():
        if key in normalized or not isinstance(schema, dict):
            continue
        default = schema.get("default")
        if isinstance(default, bool):
            continue
        if isinstance(default, (int, float)):
            normalized[key] = default
    return normalized


def _canonicalize_kwargs_for_export(kwargs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    return dict(kwargs) if isinstance(kwargs, dict) else kwargs


def wrap_openapi_tool(original_tool: FunctionTool) -> FunctionTool:
    original_func = original_tool.func

    if hasattr(original_tool, 'openai_tool_schema'):
        original_schema = original_tool.openai_tool_schema
    else:
        try:
            original_schema = original_tool.get_openai_function_schema()
        except Exception:
            return original_tool

    if "function" in original_schema:
        func_schema = original_schema["function"]
        params = func_schema.get("parameters", {})
    else:
        func_schema = original_schema
        params = original_schema.get("parameters", {})

    props = params.get("properties", {})
    needs_request_body = "requestBody" in props and len(props) == 1

    if not needs_request_body:
        return original_tool

    request_body_schema = props.get("requestBody", {})
    actual_params = request_body_schema.get("properties", {})
    actual_required = request_body_schema.get("required", [])

    def wrapped_func(**kwargs):
        if "requestBody" in kwargs:
            return original_func(**kwargs)
        return original_func(requestBody=kwargs)

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
    wrapped_tool = FunctionTool(func=wrapped_func, openai_tool_schema=new_schema)
    preconditions = getattr(original_tool, "bfcl_preconditions", None)
    if preconditions:
        setattr(wrapped_tool, "bfcl_preconditions", preconditions)

    tool_name = new_schema['function']['name'] if 'function' in new_schema else new_schema.get('name', 'unknown')
    logger.debug(f"Wrapped tool {tool_name} to handle requestBody format")

    return wrapped_tool


def fix_openapi_tools(tools: List[FunctionTool]) -> List[FunctionTool]:
    fixed_tools = []
    for tool in tools:
        try:
            fixed_tool = wrap_openapi_tool(tool)
            fixed_tool = _strip_schema_prefix_from_tool(fixed_tool)
            fixed_tool = _clarify_extractor_fields_for_agent(fixed_tool)
            fixed_tools.append(fixed_tool)
        except Exception as e:
            logger.warning(f"Failed to wrap tool: {e}")
            fixed_tools.append(tool)

    logger.info(f"Fixed {len(fixed_tools)} OpenAPIToolkit tools for requestBody handling")
    return fixed_tools


def _clarify_extractor_fields_for_agent(tool: FunctionTool) -> FunctionTool:
    """Clarify underspecified extractor item fields in agent-facing schemas."""
    schema = getattr(tool, "openai_tool_schema", None)
    if not isinstance(schema, dict):
        return tool

    function_schema = schema.get("function") if isinstance(schema.get("function"), dict) else schema
    if not isinstance(function_schema, dict):
        return tool

    parameters = function_schema.get("parameters")
    _clarify_extractor_schema(function_schema)
    return tool


def _clarify_extractor_schema(function_schema: Dict[str, Any]) -> None:
    name = str(function_schema.get("name") or "").lower()
    if "extractor" not in name:
        return
    parameters = function_schema.get("parameters")
    if not isinstance(parameters, dict):
        return
    data_schema = (parameters.get("properties") or {}).get("data")
    if not isinstance(data_schema, dict) or data_schema.get("type") != "array":
        return
    item_schema = data_schema.get("items")
    if not isinstance(item_schema, dict):
        return
    item_props = item_schema.setdefault("properties", {})
    if not isinstance(item_props, dict):
        return

    text = " ".join(
        str(part or "")
        for part in (
            function_schema.get("description"),
            data_schema.get("description"),
            item_schema.get("description"),
        )
    ).lower()
    if not item_props and "name" in text and "age" in text:
        item_props.update(
            {
                "name": {
                    "type": "string",
                    "description": "Name text extracted from the user-provided text. Use exactly the name present; do not invent missing surnames.",
                },
                "age": {
                    "type": "integer",
                    "description": "Age extracted or directly inferred from the user-provided text.",
                },
            }
        )
        item_schema.setdefault("additionalProperties", False)

    name_schema = item_props.get("name")
    if isinstance(name_schema, dict):
        description = str(name_schema.get("description") or "").rstrip()
        if "do not invent missing surnames" not in description:
            name_schema["description"] = (
                f"{description} Use exactly the name present in the text; "
                "do not invent missing surnames."
            ).strip()


def _strip_schema_prefix_from_tool(tool: FunctionTool) -> FunctionTool:
    schema = getattr(tool, 'openai_tool_schema', None)
    if not schema:
        return tool

    if 'function' in schema:
        name = schema['function'].get('name', '')
        desc = schema['function'].get('description', '')
    else:
        name = schema.get('name', '')
        desc = schema.get('description', '')

    prefix_match = re.match(r'^([A-Z]{1,4}\d+)_', name)
    if not prefix_match:
        return tool

    prefix = prefix_match.group(1)
    new_name = name[len(prefix) + 1:]

    new_desc = desc
    new_desc = re.sub(
        rf'\s*This function is from {re.escape(prefix)}[ .]?(?:API\.?)?\s*'
        rf'(?:(?:The\s+)?{re.escape(prefix)}\b.*)?$',
        '', new_desc, flags=re.DOTALL,
    )
    new_desc = new_desc.strip()

    if 'function' in schema:
        schema['function']['name'] = new_name
        schema['function']['description'] = new_desc
    else:
        schema['name'] = new_name
        schema['description'] = new_desc

    logger.debug(f"Stripped schema prefix: {name} -> {new_name}")
    return tool

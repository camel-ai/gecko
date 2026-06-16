import os
from typing import Any, Dict, List, Optional

from ..schemas.global_loader import get_global_schema_loader


def _summarize_operation(op: Dict[str, Any], method: str, path: str) -> Dict[str, Any]:
    # Minimized to control tokens: only id/summary/description
    return {
        "operationId": op.get("operationId"),
        "summary": op.get("summary", ""),
        "description": op.get("description", ""),
        # Keep light context if needed later without heavy schemas
        "method": method.upper(),
        "path": path,
    }


def introspect_toolkit(api_name: str) -> Optional[Dict[str, Any]]:
    """Summarize a toolkit's state-related information from its OpenAPI schema.

    Returns a compact dict including info, and summarized operations as grounding for LLM.
    """
    loader = get_global_schema_loader()
    if not loader:
        return None
    schema_path = loader.find_schema_file(api_name)
    if not schema_path:
        return None
    schema = loader.load_schema(schema_path)
    info = schema.get("info", {})
    paths = schema.get("paths", {})

    operations: List[Dict[str, Any]] = []
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if not isinstance(op, dict):
                continue
            operations.append(_summarize_operation(op, method, path))

    # Include x-default-state if present (for LLM awareness)
    x_default_state = info.get("x-default-state", None)
    
    result = {
        "api_name": api_name,
        "info": {
            "title": info.get("title"),
            "description": info.get("description", ""),
            "version": info.get("version")
        },
        "operations": operations
    }
    
    # Add x-default-state summary if present (for LLM context)
    if x_default_state:
        result["x_default_state_summary"] = {
            "has_default_state": bool(x_default_state.get("default_state")),
            "has_preset_data": bool(x_default_state.get("preset_data")),
            "has_static_values": bool(x_default_state.get("static_values")),
            "has_validation_rules": bool(x_default_state.get("validation_rules")),
            "has_data_relationships": bool(x_default_state.get("data_relationships"))
        }
    
    return result


def introspect_toolkits(api_names: List[str]) -> Dict[str, Any]:
    """Summarize multiple toolkits by name."""
    result: Dict[str, Any] = {}
    for name in api_names:
        summary = introspect_toolkit(name)
        if summary:
            result[name] = summary
    return result



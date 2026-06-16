from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from benchmarks.bfcl.utils import (
    compress_single_turn_function_name,
    derive_single_turn_endpoint_name,
    derive_single_turn_schema_name,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TASK_DIR = _PROJECT_ROOT / "data" / "bfcl" / "task"
_SINGLE_OPENAPI_DIR = _PROJECT_ROOT / "data" / "bfcl" / "openapi" / "single_turn"
_SOURCE_CACHE: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None


def align_bfcl_single_turn_openapi_schema(
    schema_path: str | Path,
    openapi_schema: Dict[str, Any],
) -> None:
    """Align generated single-turn OpenAPI request schemas with BFCL source.

    The compact OpenAPI files are generated artifacts. In a few cases the
    converter drifts from BFCL's canonical ``function`` schema, for example by
    dropping a default enum value such as ``dontcare``. Gecko and the agent
    should both consume the canonical BFCL tool contract, so this function
    patches the loaded schema in memory without rewriting data files.
    """
    path = Path(schema_path)
    try:
        if path.resolve().parent != _SINGLE_OPENAPI_DIR.resolve():
            return
    except OSError:
        return

    source_by_operation = _load_single_turn_source_schemas().get(path.stem)
    if not source_by_operation:
        return

    paths = openapi_schema.get("paths")
    if not isinstance(paths, dict):
        return

    for methods in paths.values():
        if not isinstance(methods, dict):
            continue
        for operation in methods.values():
            if not isinstance(operation, dict):
                continue
            operation_id = operation.get("operationId")
            source_params = source_by_operation.get(str(operation_id))
            if not source_params:
                continue
            request_schema = (
                operation.setdefault("requestBody", {})
                .setdefault("content", {})
                .setdefault("application/json", {})
                .setdefault("schema", {})
            )
            if isinstance(request_schema, dict):
                _replace_request_schema_from_bfcl_source(request_schema, source_params)


def _load_single_turn_source_schemas() -> Dict[str, Dict[str, Dict[str, Any]]]:
    global _SOURCE_CACHE
    if _SOURCE_CACHE is not None:
        return _SOURCE_CACHE

    result: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for task_file in sorted(_TASK_DIR.glob("BFCL_v4_*.json")):
        with task_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    task = json.loads(line)
                except json.JSONDecodeError:
                    continue
                task_id = str(task.get("id") or "")
                functions = task.get("function")
                if not task_id or not isinstance(functions, list):
                    continue
                schema_name = derive_single_turn_schema_name(task_id)
                operation_map = result.setdefault(schema_name, {})
                for function in functions:
                    if not isinstance(function, dict):
                        continue
                    raw_name = str(function.get("name") or "")
                    params = function.get("parameters")
                    if not raw_name or not isinstance(params, dict):
                        continue
                    operation_map[compress_single_turn_function_name(raw_name)] = params
                    operation_map[derive_single_turn_endpoint_name(raw_name)] = params

    _SOURCE_CACHE = result
    return result


def _replace_request_schema_from_bfcl_source(
    request_schema: Dict[str, Any],
    source_params: Dict[str, Any],
) -> None:
    source_schema = _convert_bfcl_schema(source_params)
    _merge_source_schema(request_schema, source_schema)
    request_schema.setdefault("type", "object")


def _merge_source_schema(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Fill source BFCL contract gaps without overwriting curated OpenAPI text.

    The compact OpenAPI files are allowed to carry clearer descriptions and
    benchmark-specific schema clarifications. BFCL source schemas remain useful
    for missing enum/default/type information, but replacing the full request
    schema discards those clarifications and makes live_simple prompt behavior
    worse.
    """
    for key in ("type", "default"):
        if key in source and key not in target:
            target[key] = copy.deepcopy(source[key])

    if isinstance(source.get("enum"), list):
        target_enum = target.get("enum")
        if isinstance(target_enum, list):
            merged = list(target_enum)
            for value in source["enum"]:
                if value not in merged:
                    merged.append(value)
            target["enum"] = merged
        elif "enum" not in target:
            target["enum"] = copy.deepcopy(source["enum"])

    if source.get("additionalProperties") is False and "additionalProperties" not in target:
        target["additionalProperties"] = False

    source_required = source.get("required")
    if isinstance(source_required, list) and "required" not in target:
        target["required"] = copy.deepcopy(source_required)

    source_props = source.get("properties")
    if isinstance(source_props, dict):
        target_props = target.setdefault("properties", {})
        if isinstance(target_props, dict):
            for name, source_prop in source_props.items():
                if not isinstance(source_prop, dict):
                    continue
                target_prop = target_props.get(name)
                if isinstance(target_prop, dict):
                    _merge_source_schema(target_prop, source_prop)
                else:
                    target_props[name] = copy.deepcopy(source_prop)

    if isinstance(source.get("items"), dict):
        target_items = target.get("items")
        if isinstance(target_items, dict):
            _merge_source_schema(target_items, source["items"])
        elif "items" not in target:
            target["items"] = copy.deepcopy(source["items"])


def _convert_bfcl_schema(source_schema: Dict[str, Any]) -> Dict[str, Any]:
    converted: Dict[str, Any] = {}
    for key, value in source_schema.items():
        if key == "type":
            converted_type = _convert_bfcl_type(value)
            if converted_type is not None:
                converted[key] = converted_type
        elif key == "properties" and isinstance(value, dict):
            converted[key] = {
                str(name): _convert_bfcl_schema(prop)
                for name, prop in value.items()
                if isinstance(prop, dict)
            }
        elif key == "items" and isinstance(value, dict):
            converted[key] = _convert_bfcl_schema(value)
        else:
            converted[key] = copy.deepcopy(value)

    if converted.get("type") == "object" and "properties" in converted:
        converted.setdefault("additionalProperties", False)

    if "default" not in converted:
        inferred_default = _infer_quoted_default(converted.get("description"))
        if inferred_default is not None:
            converted["default"] = inferred_default

    enum = converted.get("enum")
    default = converted.get("default")
    if isinstance(enum, list) and default is not None and default not in enum:
        converted["enum"] = list(enum) + [default]

    return converted


def _convert_bfcl_type(value: Any) -> Any:
    if value == "dict":
        return "object"
    if value == "float":
        return "number"
    if value == "tuple":
        return "array"
    if value == "any":
        return None
    return value


def _infer_quoted_default(description: Any) -> Optional[str]:
    if not isinstance(description, str):
        return None
    match = re.search(
        r"\bdefaults?\s+(?:value\s+)?(?:is|to|=)\s*(['\"])(.*?)\1",
        description,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(2)

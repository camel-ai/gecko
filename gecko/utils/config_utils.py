"""Utilities for config transformation, validation, and comparison."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def deep_merge_configs(base_config: Dict[str, Any], update_config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-merged copy of two config dictionaries."""
    result = deepcopy(base_config)

    def _merge_dict(target: Dict[str, Any], source: Dict[str, Any]) -> None:
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                _merge_dict(target[key], value)
            else:
                target[key] = deepcopy(value)

    _merge_dict(result, update_config)
    return result


def extract_config_changes(old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
    """Compute field-level changes from old_config to new_config."""
    changes: Dict[str, Any] = {}

    def _extract_changes(old: Dict[str, Any], new: Dict[str, Any], path: str = "") -> None:
        for key, new_value in new.items():
            current_path = f"{path}.{key}" if path else key
            if key not in old:
                changes[current_path] = {"action": "added", "value": new_value}
            elif isinstance(old[key], dict) and isinstance(new_value, dict):
                _extract_changes(old[key], new_value, current_path)
            elif old[key] != new_value:
                changes[current_path] = {
                    "action": "modified",
                    "old_value": old[key],
                    "new_value": new_value,
                }

        for key in old:
            if key not in new:
                current_path = f"{path}.{key}" if path else key
                changes[current_path] = {"action": "removed", "value": old[key]}

    _extract_changes(old_config, new_config)
    return changes


def validate_config_schema(config: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a config object against a simplified JSON schema structure."""
    errors: List[str] = []

    def _validate_value(value: Any, schema_def: Dict[str, Any], path: str = "") -> None:
        schema_type = schema_def.get("type")

        if schema_type == "object" and isinstance(value, dict):
            properties = schema_def.get("properties", {})
            required = schema_def.get("required", [])
            for req_field in required:
                if req_field not in value:
                    errors.append(f"Missing required field: {path}.{req_field}")
            for prop_name, prop_value in value.items():
                if prop_name in properties:
                    prop_path = f"{path}.{prop_name}" if path else prop_name
                    _validate_value(prop_value, properties[prop_name], prop_path)
            return

        if schema_type == "array" and isinstance(value, list):
            items_schema = schema_def.get("items", {})
            for index, item in enumerate(value):
                _validate_value(item, items_schema, f"{path}[{index}]")
            return

        if schema_type == "string" and not isinstance(value, str):
            errors.append(f"Expected string at {path}, got {type(value).__name__}")
        elif schema_type == "integer" and not isinstance(value, int):
            errors.append(f"Expected integer at {path}, got {type(value).__name__}")
        elif schema_type == "number" and not isinstance(value, (int, float)):
            errors.append(f"Expected number at {path}, got {type(value).__name__}")
        elif schema_type == "boolean" and not isinstance(value, bool):
            errors.append(f"Expected boolean at {path}, got {type(value).__name__}")

    try:
        _validate_value(config, schema)
        return len(errors) == 0, errors
    except Exception as exc:
        logger.error("Config validation error: %s", exc)
        return False, [str(exc)]


def sanitize_config(
    config: Dict[str, Any],
    allowed_keys: Optional[List[str]] = None,
    sensitive_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return a sanitized copy of a config dictionary."""
    result = deepcopy(config)

    if sensitive_keys:
        for key in sensitive_keys:
            if key not in result:
                continue
            if isinstance(result[key], str) and len(result[key]) > 4:
                result[key] = result[key][:2] + "*" * (len(result[key]) - 4) + result[key][-2:]
            else:
                result[key] = "***"

    if allowed_keys:
        result = {k: v for k, v in result.items() if k in allowed_keys}

    return result


def convert_config_types(config: Dict[str, Any], type_mapping: Dict[str, type]) -> Dict[str, Any]:
    """Convert top-level config keys to requested Python types when possible."""
    result = deepcopy(config)

    def _convert_value(value: Any, target_type: type) -> Any:
        try:
            if target_type == bool and isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")
            if target_type == int and isinstance(value, str):
                return int(value)
            if target_type == float and isinstance(value, str):
                return float(value)
            if target_type == str:
                return str(value)
            return target_type(value)
        except (ValueError, TypeError) as exc:
            logger.warning("Failed to convert %s to %s: %s", value, target_type, exc)
            return value

    for key, target_type in type_mapping.items():
        if key in result:
            result[key] = _convert_value(result[key], target_type)

    return result


def flatten_config(config: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
    """Flatten nested config into a single-level dotted-key dict."""
    result: Dict[str, Any] = {}

    def _flatten(obj: Dict[str, Any], prefix: str = "") -> None:
        for key, value in obj.items():
            new_key = f"{prefix}{separator}{key}" if prefix else key
            if isinstance(value, dict):
                _flatten(value, new_key)
            else:
                result[new_key] = value

    _flatten(config)
    return result


def unflatten_config(flat_config: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
    """Rebuild nested config from flattened dotted-key representation."""
    result: Dict[str, Any] = {}
    for key, value in flat_config.items():
        keys = key.split(separator)
        current = result
        for item in keys[:-1]:
            if item not in current:
                current[item] = {}
            current = current[item]
        current[keys[-1]] = value
    return result


def get_config_diff(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
    """Return added/removed/modified/unchanged diff between two configs."""
    flat1 = flatten_config(config1)
    flat2 = flatten_config(config2)
    all_keys = set(flat1.keys()) | set(flat2.keys())

    diff = {"added": {}, "removed": {}, "modified": {}, "unchanged": {}}
    for key in all_keys:
        if key not in flat1:
            diff["added"][key] = flat2[key]
        elif key not in flat2:
            diff["removed"][key] = flat1[key]
        elif flat1[key] != flat2[key]:
            diff["modified"][key] = {"old": flat1[key], "new": flat2[key]}
        else:
            diff["unchanged"][key] = flat1[key]
    return diff


def apply_config_template(config: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
    """Apply defaults from template and override with user config values."""
    return deep_merge_configs(deepcopy(template), config)

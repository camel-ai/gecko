"""Schema normalization, sanitization, and protected-path merging."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Set, Tuple

from .models import (
    SCHEMA_KEYWORDS,
    dedup_string_list,
    infer_scalar_type_from_literal,
    join_pointer,
    normalize_type_name,
)


class SchemaProcessor:
    """Pure schema operations -- no LLM dependency."""

    def normalize(
        self,
        raw_schema: Any,
        *,
        default_description: str,
        pointer: str,
        protect_explicit: bool,
    ) -> Tuple[Dict[str, Any], Set[str]]:
        protected_paths: Set[str] = set()

        if not isinstance(raw_schema, dict):
            inferred_type = infer_scalar_type_from_literal(raw_schema)
            schema: Dict[str, Any] = {
                "type": inferred_type,
                "description": default_description,
            }
            if raw_schema is not None and inferred_type != "object":
                schema["default"] = raw_schema
            return schema, protected_paths

        schema = {}
        explicit_keys = set(raw_schema.keys())
        raw_type = raw_schema.get("type")

        if raw_type is None:
            if "properties" in raw_schema:
                raw_type = "object"
            elif "items" in raw_schema:
                raw_type = "array"
            elif "oneOf" in raw_schema or "anyOf" in raw_schema:
                raw_type = "object"
            else:
                raw_type = "string"

        normalized_type = normalize_type_name(raw_type)
        schema["type"] = normalized_type
        if protect_explicit and "type" in explicit_keys:
            protected_paths.add(join_pointer(pointer, "type"))

        # Preserve float origin: when original type was "float"/"double" and
        # normalized to "number", tag with format so downstream descriptions
        # can hint at decimal notation (important for BFCL eval strictness).
        _raw_lower = str(raw_type).strip().lower() if raw_type else ""
        if normalized_type == "number" and _raw_lower in ("float", "double"):
            schema.setdefault("format", "double")

        description = raw_schema.get("description")
        if isinstance(description, str) and description.strip():
            schema["description"] = description.strip()
            if protect_explicit:
                protected_paths.add(join_pointer(pointer, "description"))
        else:
            schema["description"] = default_description

        # When format is "double", append decimal-notation hint to description
        # so that callers (and LLM agents) know to use 1.0 instead of 1.
        if schema.get("format") == "double":
            _hint = "Provide as a decimal number (e.g., 1.0, not 1)."
            if _hint not in schema["description"]:
                schema["description"] = f'{schema["description"]}. {_hint}'

        for key in (
            "format",
            "enum",
            "default",
            "nullable",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "minLength",
            "maxLength",
            "pattern",
            "minItems",
            "maxItems",
            "uniqueItems",
        ):
            if key in raw_schema:
                schema[key] = deepcopy(raw_schema[key])
                if protect_explicit:
                    protected_paths.add(join_pointer(pointer, key))

        # oneOf / anyOf
        for compound_key in ("oneOf", "anyOf"):
            raw_list = raw_schema.get(compound_key)
            if not isinstance(raw_list, list):
                continue
            items: List[Any] = []
            for idx, item in enumerate(raw_list):
                child, child_protected = self.normalize(
                    item,
                    default_description=f"{default_description} option {idx + 1}",
                    pointer=join_pointer(pointer, compound_key) + f"/{idx}",
                    protect_explicit=protect_explicit,
                )
                items.append(child)
                protected_paths.update(child_protected)
            schema[compound_key] = items
            if protect_explicit:
                protected_paths.add(join_pointer(pointer, compound_key))

        # array items
        if normalized_type == "array":
            items_pointer = join_pointer(pointer, "items")
            if "items" in raw_schema:
                child, child_protected = self.normalize(
                    raw_schema.get("items"),
                    default_description="Array item",
                    pointer=items_pointer,
                    protect_explicit=protect_explicit,
                )
                schema["items"] = child
                protected_paths.update(child_protected)
                if protect_explicit:
                    protected_paths.add(items_pointer)
            else:
                schema["items"] = {"type": "string", "description": "Array item"}

        # object properties
        if normalized_type == "object":
            properties: Dict[str, Any] = {}
            raw_props = raw_schema.get("properties")
            if isinstance(raw_props, dict):
                for prop_name, prop_schema in raw_props.items():
                    child_pointer = join_pointer(join_pointer(pointer, "properties"), prop_name)
                    child, child_protected = self.normalize(
                        prop_schema,
                        default_description=f"Parameter {prop_name}",
                        pointer=child_pointer,
                        protect_explicit=protect_explicit,
                    )
                    properties[prop_name] = child
                    protected_paths.update(child_protected)
            schema["properties"] = properties

            required_raw = raw_schema.get("required", [])
            if isinstance(required_raw, list):
                required = [name for name in dedup_string_list(required_raw) if name in properties]
                if required:
                    schema["required"] = required
                if protect_explicit and "required" in explicit_keys:
                    protected_paths.add(join_pointer(pointer, "required"))

            if "additionalProperties" in raw_schema:
                schema["additionalProperties"] = deepcopy(raw_schema["additionalProperties"])
                if protect_explicit:
                    protected_paths.add(join_pointer(pointer, "additionalProperties"))
            elif properties:
                # Has defined properties → lock down to prevent fabrication
                schema["additionalProperties"] = False
            # else: no properties defined → map/dict type, leave open

        return schema, protected_paths

    def sanitize_fragment(self, fragment: Any) -> Dict[str, Any]:
        if not isinstance(fragment, dict):
            return {}
        cleaned: Dict[str, Any] = {}
        for key, value in fragment.items():
            if key not in SCHEMA_KEYWORDS:
                continue
            if key == "properties" and isinstance(value, dict):
                cleaned["properties"] = {
                    prop_name: self.sanitize_fragment(prop_schema)
                    for prop_name, prop_schema in value.items()
                }
            elif key == "items" and isinstance(value, dict):
                cleaned["items"] = self.sanitize_fragment(value)
            elif key in ("oneOf", "anyOf") and isinstance(value, list):
                cleaned[key] = [
                    self.sanitize_fragment(item) for item in value if isinstance(item, dict)
                ]
            elif key == "required" and isinstance(value, list):
                cleaned["required"] = dedup_string_list(value)
            else:
                cleaned[key] = deepcopy(value)
        return cleaned

    def merge_with_protection(
        self,
        base_schema: Dict[str, Any],
        updates: Dict[str, Any],
        protected_paths: Set[str],
        pointer: str = "",
    ) -> Dict[str, Any]:
        if not isinstance(base_schema, dict):
            return deepcopy(base_schema)
        if not isinstance(updates, dict):
            return deepcopy(base_schema)

        merged = deepcopy(base_schema)
        for key, update_value in updates.items():
            child_pointer = join_pointer(pointer, key)
            if child_pointer in protected_paths:
                continue

            if key not in merged:
                merged[key] = deepcopy(update_value)
                continue

            current_value = merged.get(key)
            if isinstance(current_value, dict) and isinstance(update_value, dict):
                merged[key] = self.merge_with_protection(
                    current_value,
                    update_value,
                    protected_paths,
                    child_pointer,
                )
                continue

            if isinstance(current_value, list) and isinstance(update_value, list):
                if key == "required":
                    merged[key] = dedup_string_list(update_value)
                else:
                    merged[key] = deepcopy(update_value)
                continue

            merged[key] = deepcopy(update_value)

        return merged

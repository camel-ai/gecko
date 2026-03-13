"""Data models, constants, and pure utility functions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

SCHEMA_KEYWORDS: Set[str] = {
    "type",
    "description",
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
    "properties",
    "required",
    "items",
    "additionalProperties",
    "oneOf",
    "anyOf",
}

TYPE_ALIASES: Dict[str, str] = {
    "dict": "object",
    "map": "object",
    "str": "string",
    "text": "string",
    "int": "integer",
    "long": "integer",
    "float": "number",
    "double": "number",
    "bool": "boolean",
    "list": "array",
    "tuple": "array",
}


@dataclass
class ToolIR:
    name: str
    description: str
    request_schema: Dict[str, Any]
    protected_request_paths: Set[str] = field(default_factory=set)
    output_hint_schema: Optional[Dict[str, Any]] = None
    context_hints: List[str] = field(default_factory=list)


@dataclass
class EndpointEnrichment:
    summary: str
    description: str
    request_schema_updates: Dict[str, Any]
    success_schema: Optional[Dict[str, Any]]
    validation_rules: List[Dict[str, Any]]
    behavior_hints: List[str]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def pointer_escape(segment: str) -> str:
    return segment.replace("~", "~0").replace("/", "~1")


def join_pointer(pointer: str, key: str) -> str:
    escaped = pointer_escape(key)
    return f"{pointer}/{escaped}" if pointer else f"/{escaped}"


def sanitize_identifier(raw: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", (raw or "").strip())
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        return ""
    if value[0].isdigit():
        value = f"_{value}"
    return value


def humanize_identifier(identifier: str) -> str:
    text = re.sub(r"[_\\-]+", " ", identifier).strip()
    text = re.sub(r"\s+", " ", text)
    return text.capitalize() if text else "Execute operation"


def dedup_string_list(values: List[Any]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def normalize_type_name(raw_type: Any) -> str:
    if not isinstance(raw_type, str):
        return "string"
    key = raw_type.strip()
    if not key:
        return "string"
    return TYPE_ALIASES.get(key.lower(), key.lower())


def infer_scalar_type_from_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def extract_text_blobs(value: Any) -> List[str]:
    texts: List[str] = []
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            texts.append(cleaned)
        return texts
    if isinstance(value, list):
        for item in value:
            texts.extend(extract_text_blobs(item))
        return texts
    if isinstance(value, dict):
        for key in ("content", "text", "question", "prompt", "instruction", "task"):
            if key in value:
                texts.extend(extract_text_blobs(value.get(key)))
        return texts
    return texts


def strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

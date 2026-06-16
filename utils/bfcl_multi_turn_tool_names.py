from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


# Canonical toolkit names are aligned with BFCL multi-turn OpenAPI schema files.
BFCL_MULTI_TURN_TOOLKIT_ALIASES: Dict[str, str] = {
    "GorillaFileSystem": "GorillaFileSystem",
    "MathAPI": "MathAPI",
    "MessageAPI": "MessageAPI",
    "TwitterAPI": "TwitterAPI",
    "PostingAPI": "TwitterAPI",
    "TicketAPI": "TicketAPI",
    "TradingBot": "TradingBot",
    "TravelAPI": "TravelAPI",
    "TravelBooking": "TravelAPI",
    "VehicleControlAPI": "VehicleControlAPI",
    "VehicleControl": "VehicleControlAPI",
}

X_BFCL_TOOLKIT_KEY = "x-bfcl-toolkit"


def canonicalize_bfcl_multi_turn_toolkit(toolkit_name: Optional[str]) -> Optional[str]:
    if not isinstance(toolkit_name, str):
        return None
    stripped = toolkit_name.strip()
    if not stripped:
        return None
    return BFCL_MULTI_TURN_TOOLKIT_ALIASES.get(stripped)


def extract_bfcl_multi_turn_toolkit(tool_name: Optional[str]) -> Optional[str]:
    if not isinstance(tool_name, str):
        return None
    stripped = tool_name.strip()
    if not stripped:
        return None

    for prefix, canonical in BFCL_MULTI_TURN_TOOLKIT_ALIASES.items():
        if stripped.startswith(f"{prefix}_"):
            return canonical
    return None


def normalize_bfcl_multi_turn_tool_name(
    tool_name: Optional[str],
    *,
    default_toolkit: Optional[str] = None,
) -> str:
    if not isinstance(tool_name, str):
        return ""
    stripped = tool_name.strip()
    if not stripped:
        return ""

    explicit_toolkit = canonicalize_bfcl_multi_turn_toolkit(default_toolkit)
    if explicit_toolkit and stripped.startswith(f"{explicit_toolkit}_"):
        return stripped[len(explicit_toolkit) + 1 :]

    for prefix in BFCL_MULTI_TURN_TOOLKIT_ALIASES:
        marker = f"{prefix}_"
        if stripped.startswith(marker):
            return stripped[len(marker) :]

    return stripped


def resolve_bfcl_multi_turn_tool_identity(
    tool_name: Optional[str],
    *,
    default_toolkit: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    canonical_toolkit = (
        extract_bfcl_multi_turn_toolkit(tool_name)
        or canonicalize_bfcl_multi_turn_toolkit(default_toolkit)
    )
    bare_name = normalize_bfcl_multi_turn_tool_name(
        tool_name,
        default_toolkit=canonical_toolkit,
    )
    return bare_name, canonical_toolkit


def normalize_bfcl_multi_turn_tool_schema(
    schema: Dict[str, Any],
    *,
    default_toolkit: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    if not isinstance(schema, dict):
        return "", canonicalize_bfcl_multi_turn_toolkit(default_toolkit)

    func_schema = schema.get("function", schema)
    if not isinstance(func_schema, dict):
        return "", canonicalize_bfcl_multi_turn_toolkit(default_toolkit)

    raw_name = str(func_schema.get("name") or "").strip()
    bare_name, canonical_toolkit = resolve_bfcl_multi_turn_tool_identity(
        raw_name,
        default_toolkit=default_toolkit,
    )
    if bare_name:
        func_schema["name"] = bare_name
    if canonical_toolkit:
        schema[X_BFCL_TOOLKIT_KEY] = canonical_toolkit
        func_schema[X_BFCL_TOOLKIT_KEY] = canonical_toolkit
    return bare_name, canonical_toolkit


def normalize_bfcl_multi_turn_function_tool(
    tool: Any,
    *,
    default_toolkit: Optional[str] = None,
) -> Any:
    schema = getattr(tool, "openai_tool_schema", None)
    if not isinstance(schema, dict):
        return tool

    bare_name, canonical_toolkit = normalize_bfcl_multi_turn_tool_schema(
        schema,
        default_toolkit=default_toolkit,
    )

    func = getattr(tool, "func", None)
    if callable(func) and bare_name:
        try:
            func.__name__ = bare_name
        except Exception:
            pass

    if canonical_toolkit:
        try:
            setattr(tool, "bfcl_toolkit", canonical_toolkit)
        except Exception:
            pass

    return tool


def normalize_bfcl_multi_turn_tool_call(
    tool_call: Dict[str, Any],
    *,
    default_toolkit: Optional[str] = None,
    rename: bool = True,
    preserve_original: bool = False,
) -> Dict[str, Any]:
    if not isinstance(tool_call, dict):
        return tool_call

    normalized = dict(tool_call)
    raw_toolkit = normalized.get("toolkit")
    current_name = (
        normalized.get("function")
        or normalized.get("name")
        or normalized.get("function_name")
        or ""
    )
    bare_name, canonical_toolkit = resolve_bfcl_multi_turn_tool_identity(
        str(current_name),
        default_toolkit=raw_toolkit or default_toolkit,
    )

    if canonical_toolkit:
        normalized["toolkit"] = canonical_toolkit

    if rename and bare_name and current_name:
        if preserve_original and bare_name != current_name:
            normalized.setdefault("original_function", current_name)
        if "function" in normalized or "name" not in normalized:
            normalized["function"] = bare_name
        if "name" in normalized:
            normalized["name"] = bare_name
        if "function_name" in normalized:
            normalized["function_name"] = bare_name

    return normalized

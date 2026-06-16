"""Shared state-update hint derivation.

These helpers convert a write-call's contract (``state_hints``), its observed
arguments/result, and the previous persistent state into concrete, contract-
grounded hints that can be attached to the state-updater prompt:

- ``matched_read_scalar_candidates``: scalar candidates from matched context-only
  read calls aligned to the write's persisted fields.
- ``helper_static_scalar_candidates``: scalar candidates derived from helper
  static lookup tables / branch rules declared on the write.
- ``dynamic_key_shape_hints``: canonical append/remove shape for dynamic-key
  collections.
- ``set_membership_hints``: scalar-membership persistence for generated IDs.
- ``latest_match_hints``: "latest/most-recent" selector policy for updates.
- ``resolved_persistent_records``: exact current record(s) the write will mutate.
- ``resolved_effect_hints``: concrete path/value mutations derived from
  natural-language ``state_effects`` plus the previous state.

The module keeps state-hint derivation shared and deterministic across updater
paths.
"""

from __future__ import annotations

import ast
import copy
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


_LIST_FILTER_RE = re.compile(
    r"^\?\(@\.(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s*==\s*(?P<value>.+)\)$"
)


def _parse_selector_literal(token: str) -> Any:
    stripped = str(token).strip()
    if not stripped:
        return stripped
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1]
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if re.fullmatch(r"-?\d+", stripped):
        try:
            return int(stripped)
        except Exception:
            return stripped
    return stripped


def _find_list_item_index_by_field(items: List[Any], field_name: str, expected_value: Any) -> Optional[int]:
    matches: List[int] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict) or field_name not in item:
            continue
        item_value = item.get(field_name)
        if item_value == expected_value:
            matches.append(idx)
            continue
        if isinstance(item_value, (int, str, bool)) or item_value is None:
            if str(item_value) == str(expected_value):
                matches.append(idx)
    if len(matches) == 1:
        return matches[0]
    return None


def resolve_direct_logical_path(
    state: Dict[str, Any],
    logical_path: str,
    *,
    prefer_append_for_missing_leaf: bool = False,
) -> Tuple[str, Any]:
    """Walk a ``Toolkit/field/.../leaf`` path into ``state``.

    Resolve filesystem-style ``contents`` wrappers, list-index and
    ``?(@.field==value)`` selectors, and return a normalized JSON-pointer plus
    the resolved node (or ``None``).
    """
    segments = [s for s in logical_path.strip("/").split("/") if s]
    pointer_parts: List[str] = []
    current: Any = state

    for idx, seg in enumerate(segments):
        is_leaf = idx == len(segments) - 1
        if isinstance(current, dict):
            if seg in current:
                pointer_parts.append(seg)
                current = current[seg]
            elif "contents" in current and isinstance(current.get("contents"), dict):
                pointer_parts.append("contents")
                pointer_parts.append(seg)
                current = current["contents"].get(seg)
            else:
                pointer_parts.append(seg)
                current = None
            continue

        if isinstance(current, list):
            if seg == "-":
                pointer_parts.append(seg)
                current = None
                continue
            match_idx: Optional[int] = None
            try:
                candidate_idx: Optional[int] = int(seg)
            except ValueError:
                candidate_idx = None
            if candidate_idx is not None and 0 <= candidate_idx < len(current):
                match_idx = candidate_idx
            else:
                filter_match = _LIST_FILTER_RE.match(seg)
                if filter_match:
                    match_idx = _find_list_item_index_by_field(
                        current,
                        filter_match.group("field"),
                        _parse_selector_literal(filter_match.group("value")),
                    )
                elif candidate_idx is not None:
                    match_idx = _find_list_item_index_by_field(current, "id", candidate_idx)
                else:
                    match_idx = _find_list_item_index_by_field(current, "id", _parse_selector_literal(seg))
            if match_idx is None:
                if prefer_append_for_missing_leaf and is_leaf:
                    pointer_parts.append("-")
                    current = None
                    continue
                pointer_parts.append(seg)
                current = None
                continue
            pointer_parts.append(str(match_idx))
            current = current[match_idx] if 0 <= match_idx < len(current) else None
            continue

        pointer_parts.append(seg)
        current = None

    return "/" + "/".join(pointer_parts), current


def _target_persistent_field_names(entry: Dict[str, Any]) -> List[str]:
    fields: List[str] = []
    seen: set = set()
    for shape in entry.get("persistent_state_shapes", []) or []:
        if not isinstance(shape, dict):
            continue
        path = shape.get("path")
        if isinstance(path, str) and path.strip():
            leaf = path.strip().split("/")[-1].strip()
            if leaf and leaf not in seen:
                seen.add(leaf)
                fields.append(leaf)
        shape_fields = shape.get("fields")
        if not isinstance(shape_fields, dict):
            continue
        for field_name in shape_fields.keys():
            if not isinstance(field_name, str):
                continue
            normalized = field_name.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                fields.append(normalized)
    return fields


def _normalize_hint_token(value: Any) -> str:
    text = str(value or "").strip().strip("`")
    if not text:
        return ""
    return re.sub(r"[^A-Za-z0-9_]+", "", text).lower()


def _unwrap_single_scalar(value: Any) -> Optional[Any]:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list) and len(value) == 1:
        item = value[0]
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
    if isinstance(value, dict):
        candidates = [
            nested
            for nested in (
                _unwrap_single_scalar(item)
                for item in value.values()
            )
            if nested is not None
        ]
        if len(candidates) == 1:
            return candidates[0]
    return None


def _scalar_candidates_from_value(value: Any, field_name: str = "") -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    if isinstance(value, (str, int, float, bool)) or value is None:
        if field_name:
            candidates.append({"field": field_name, "value": value})
        return candidates
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], (str, int, float, bool)):
            if field_name:
                candidates.append({"field": field_name, "value": value[0]})
                if field_name.endswith("_list") and len(field_name) > len("_list"):
                    candidates.append({"field": field_name[: -len("_list")], "value": value[0]})
        return candidates
    if isinstance(value, dict):
        for nested_name, nested_value in value.items():
            if not isinstance(nested_name, str):
                continue
            candidates.extend(_scalar_candidates_from_value(nested_value, nested_name))
        return candidates
    return candidates


def _lookup_toolkit_scalar(previous_state: Dict[str, Any], toolkit: str, token: str) -> Optional[Any]:
    toolkit_state = previous_state.get(toolkit)
    if not isinstance(toolkit_state, dict):
        return None
    direct = toolkit_state.get(token)
    if isinstance(direct, (str, int, float, bool)) or direct is None:
        return direct
    normalized = _normalize_hint_token(token)
    for key, value in toolkit_state.items():
        if _normalize_hint_token(key) != normalized:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
    return None


def _resolve_scalar_reference(
    token: str,
    entry: Dict[str, Any],
    previous_state: Dict[str, Any],
) -> Optional[Any]:
    if not isinstance(token, str) or not token.strip():
        return None
    stripped = token.strip().strip("`")
    normalized = _normalize_hint_token(stripped)
    args = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
    result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
    toolkit = entry.get("toolkit") if isinstance(entry.get("toolkit"), str) else ""

    if isinstance(args, dict):
        for key, value in args.items():
            if _normalize_hint_token(key) != normalized:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            single = _unwrap_single_scalar(value)
            if single is not None:
                return single

    if isinstance(result, dict):
        for key, value in result.items():
            if _normalize_hint_token(key) != normalized:
                continue
            single = _unwrap_single_scalar(value)
            if single is not None:
                return single
        for candidate in _scalar_candidates_from_value(result):
            field_name = candidate.get("field")
            if _normalize_hint_token(field_name) != normalized:
                continue
            return candidate.get("value")

    for bucket_name in (
        "matched_read_scalar_candidates",
        "helper_static_scalar_candidates",
    ):
        bucket = entry.get(bucket_name)
        if not isinstance(bucket, list):
            continue
        for candidate in bucket:
            if not isinstance(candidate, dict):
                continue
            field_name = candidate.get("field") or candidate.get("source_key")
            if _normalize_hint_token(field_name) != normalized:
                continue
            value = candidate.get("value")
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value

    for resolved in entry.get("resolved_persistent_records", []) or []:
        if not isinstance(resolved, dict):
            continue
        record = resolved.get("record")
        if not isinstance(record, dict):
            continue
        for key, value in record.items():
            if _normalize_hint_token(key) != normalized:
                continue
            single = _unwrap_single_scalar(value)
            if single is not None:
                return single

    if toolkit:
        toolkit_scalar = _lookup_toolkit_scalar(previous_state, toolkit, stripped)
        if toolkit_scalar is not None:
            return toolkit_scalar

    return None


def _split_effect_target_path(path_expr: str) -> Optional[List[Tuple[str, Optional[str]]]]:
    if not isinstance(path_expr, str):
        return None
    cleaned = path_expr.strip().strip("`").rstrip(".")
    if not cleaned:
        return None
    parts: List[Tuple[str, Optional[str]]] = []
    for raw_part in cleaned.split("."):
        part = raw_part.strip()
        if not part:
            return None
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[([A-Za-z_][A-Za-z0-9_]*)\])?", part)
        if not match:
            return None
        parts.append((match.group(1), match.group(2)))
    return parts


def _resolve_effect_target_path(
    toolkit: str,
    target_expr: str,
    entry: Dict[str, Any],
    previous_state: Dict[str, Any],
) -> Optional[str]:
    parts = _split_effect_target_path(target_expr)
    if not parts:
        return None
    segments: List[str] = [toolkit]
    for field_name, selector_name in parts:
        segments.append(field_name)
        if not selector_name:
            continue
        selector_value = _resolve_scalar_reference(selector_name, entry, previous_state)
        if selector_value is None:
            return None
        segments.append(str(selector_value))
    return "/".join(segments)


def _collection_add_hint(
    previous_state: Dict[str, Any],
    logical_path: str,
    value: Any,
) -> Optional[Dict[str, Any]]:
    _, node = resolve_direct_logical_path(previous_state, logical_path)
    if isinstance(node, list):
        return {"op": "add", "path": f"{logical_path}/-", "value": value}
    if node is None:
        return {"op": "set", "path": logical_path, "value": [value]}
    return None


_DYNAMIC_KEY_APPEND_RE = re.compile(
    r"^append an object with dynamic key (?P<key>[A-Za-z_][A-Za-z0-9_]*) "
    r"and value (?P<value>[A-Za-z_][A-Za-z0-9_]*) to (?P<path>[A-Za-z_][A-Za-z0-9_\.]*)\.?$",
    re.IGNORECASE,
)
_KEYED_APPEND_RE = re.compile(
    r"^append (?P<value>[A-Za-z_][A-Za-z0-9_]*) to "
    r"(?P<path>[A-Za-z_][A-Za-z0-9_]*)\[(?P<key>[A-Za-z_][A-Za-z0-9_]*)\]\.?$",
    re.IGNORECASE,
)
_SET_MEMBERSHIP_RE = re.compile(
    r"^add (?:generated )?(?P<value>[A-Za-z_][A-Za-z0-9_]*) to "
    r"(?P<path>[A-Za-z_][A-Za-z0-9_\.]*)\.?$",
    re.IGNORECASE,
)
_COUNTER_DELTA_RE = re.compile(
    r"^(?P<verb>increment|increase|decrement|decrease)\s+(?P<field>[A-Za-z_][A-Za-z0-9_\.]*)\s+by\s+(?P<amount>\d+)\.?$",
    re.IGNORECASE,
)
_ARITHMETIC_DELTA_RE = re.compile(
    r"^(?P<verb>increase|decrease)\s+(?P<target>[A-Za-z_][A-Za-z0-9_\[\]\.]*)\s+by\s+(?P<operand>[A-Za-z_][A-Za-z0-9_]*)\.?$",
    re.IGNORECASE,
)
_REFUND_RE = re.compile(
    r"^refund (?P<operand>[A-Za-z_][A-Za-z0-9_]*) to (?P<container>[A-Za-z_][A-Za-z0-9_]*) balance "
    r"for the (?P<selector>[A-Za-z_][A-Za-z0-9_]*) used in (?P<context>[A-Za-z_][A-Za-z0-9_]*)\.?$",
    re.IGNORECASE,
)


def resolve_effect_hints(
    previous_state: Dict[str, Any],
    entry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    toolkit = entry.get("toolkit") if isinstance(entry.get("toolkit"), str) else ""
    if not toolkit:
        return []

    effects = entry.get("state_effects_on_success")
    if not isinstance(effects, list) or not effects:
        effects = entry.get("state_effects")
    if not isinstance(effects, list):
        return []

    hints: List[Dict[str, Any]] = []
    seen: set = set()

    def _dedup_append(hint: Dict[str, Any], source_effect: str) -> None:
        hint["source_effect"] = source_effect
        marker = json.dumps(hint, ensure_ascii=False, sort_keys=True, default=str)
        if marker not in seen:
            seen.add(marker)
            hints.append(hint)

    for raw_effect in effects:
        if not isinstance(raw_effect, str):
            continue
        effect = " ".join(raw_effect.split()).strip()
        if not effect:
            continue

        dynamic_match = _DYNAMIC_KEY_APPEND_RE.match(effect)
        if dynamic_match:
            key_value = _resolve_scalar_reference(dynamic_match.group("key"), entry, previous_state)
            value = _resolve_scalar_reference(dynamic_match.group("value"), entry, previous_state)
            if key_value is not None and value is not None:
                logical_path = f"{toolkit}/{dynamic_match.group('path').rstrip('.')}"
                hint = _collection_add_hint(previous_state, logical_path, {str(key_value): value})
                if hint:
                    _dedup_append(hint, effect)
            continue

        keyed_append_match = _KEYED_APPEND_RE.match(effect)
        if keyed_append_match:
            container_path = _resolve_effect_target_path(
                toolkit,
                f"{keyed_append_match.group('path')}[{keyed_append_match.group('key')}]",
                entry,
                previous_state,
            )
            value = _resolve_scalar_reference(keyed_append_match.group("value"), entry, previous_state)
            if container_path and value is not None:
                hint = _collection_add_hint(previous_state, container_path, value)
                if hint:
                    _dedup_append(hint, effect)
            continue

        membership_match = _SET_MEMBERSHIP_RE.match(effect)
        if membership_match:
            logical_path = f"{toolkit}/{membership_match.group('path').rstrip('.')}"
            value = _resolve_scalar_reference(membership_match.group("value"), entry, previous_state)
            if value is not None:
                hint = _collection_add_hint(previous_state, logical_path, value)
                if hint:
                    _dedup_append(hint, effect)
            continue

        counter_match = _COUNTER_DELTA_RE.match(effect)
        if counter_match:
            logical_path = _resolve_effect_target_path(
                toolkit,
                counter_match.group("field"),
                entry,
                previous_state,
            )
            if logical_path:
                _, current = resolve_direct_logical_path(previous_state, logical_path)
                if isinstance(current, (int, float)):
                    amount = float(counter_match.group("amount"))
                    if counter_match.group("verb").lower() in {"increment", "increase"}:
                        next_value: Any = current + amount
                    else:
                        next_value = current - amount
                    if isinstance(current, int) and float(next_value).is_integer():
                        next_value = int(next_value)
                    _dedup_append({"op": "set", "path": logical_path, "value": next_value}, effect)
            continue

        arithmetic_match = _ARITHMETIC_DELTA_RE.match(effect)
        if arithmetic_match:
            logical_path = _resolve_effect_target_path(
                toolkit,
                arithmetic_match.group("target"),
                entry,
                previous_state,
            )
            operand_value = _resolve_scalar_reference(
                arithmetic_match.group("operand"),
                entry,
                previous_state,
            )
            if logical_path and isinstance(operand_value, (int, float)):
                _, current = resolve_direct_logical_path(previous_state, logical_path)
                if isinstance(current, (int, float)):
                    if arithmetic_match.group("verb").lower() == "increase":
                        next_value = current + operand_value
                    else:
                        next_value = current - operand_value
                    if isinstance(current, int) and float(next_value).is_integer():
                        next_value = int(next_value)
                    _dedup_append({"op": "set", "path": logical_path, "value": next_value}, effect)
            continue

        refund_match = _REFUND_RE.match(effect)
        if refund_match:
            logical_path = _resolve_effect_target_path(
                toolkit,
                f"{refund_match.group('container')}[{refund_match.group('selector')}].balance",
                entry,
                previous_state,
            )
            operand_value = _resolve_scalar_reference(
                refund_match.group("operand"),
                entry,
                previous_state,
            )
            if logical_path and isinstance(operand_value, (int, float)):
                _, current = resolve_direct_logical_path(previous_state, logical_path)
                if isinstance(current, (int, float)):
                    next_value = current + operand_value
                    if isinstance(current, int) and float(next_value).is_integer():
                        next_value = int(next_value)
                    _dedup_append({"op": "set", "path": logical_path, "value": next_value}, effect)
            continue

    return hints


def _safe_eval_static_expression(expression: str, local_env: Dict[str, Any]) -> Any:
    parsed = ast.parse(expression, mode="eval")
    safe_globals = {
        "__builtins__": {},
        "sum": sum,
        "int": int,
        "float": float,
        "str": str,
        "len": len,
        **local_env,
    }
    return eval(  # noqa: S307
        compile(parsed, "<static_data_expr>", "eval"),
        safe_globals,
        {},
    )


def _scalar_like_arg_values(args: Dict[str, Any]) -> List[str]:
    return [
        str(value)
        for value in args.values()
        if isinstance(value, (str, int, float, bool)) and value is not None
    ]


def _match_numeric_lookup_candidate(lookup: Dict[str, Any], args: Dict[str, Any]) -> Optional[Any]:
    arg_values = _scalar_like_arg_values(args)
    for raw_key, raw_value in lookup.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, (int, float)):
            continue
        parts = [part for part in raw_key.split("|") if part]
        if not parts:
            continue
        if len(parts) == 1 and parts[0] in arg_values:
            return raw_value
        if len(parts) > 1:
            for start in range(len(arg_values) - len(parts) + 1):
                if arg_values[start : start + len(parts)] == parts:
                    return raw_value
    return None


def _evaluate_numeric_rule_candidate(rule: Dict[str, Any], args: Dict[str, Any]) -> Optional[Any]:
    condition = rule.get("condition")
    if_true = rule.get("if_true")
    if_false = rule.get("if_false")
    if not isinstance(condition, str) or not isinstance(if_true, (int, float)) or not isinstance(if_false, (int, float)):
        return None
    local_env = {key: value for key, value in args.items() if isinstance(value, (str, int, float, bool))}
    referenced = rule.get("referenced_expressions")
    if isinstance(referenced, dict):
        for name, expression in referenced.items():
            if not isinstance(name, str) or not isinstance(expression, str):
                continue
            try:
                local_env[name] = _safe_eval_static_expression(expression, local_env)
            except Exception:
                return None
    try:
        result = _safe_eval_static_expression(condition, local_env)
    except Exception:
        return None
    return if_true if bool(result) else if_false


def _derive_matched_read_scalar_candidates(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    target_fields = set(_target_persistent_field_names(entry))
    if not target_fields:
        return []
    candidates: List[Dict[str, Any]] = []
    seen: set = set()
    for matched in entry.get("matched_context_only_read_calls", []) or []:
        if not isinstance(matched, dict):
            continue
        for candidate in _scalar_candidates_from_value(matched.get("result")):
            field_name = candidate.get("field")
            if not isinstance(field_name, str) or field_name not in target_fields:
                continue
            marker = (
                matched.get("name"),
                field_name,
                json.dumps(candidate.get("value"), ensure_ascii=False, sort_keys=True, default=str),
            )
            if marker in seen:
                continue
            seen.add(marker)
            candidates.append(
                {
                    "source_tool": matched.get("name"),
                    "field": field_name,
                    "value": candidate.get("value"),
                    "usage_policy": (
                        "Use this matched read-derived scalar as preferred evidence for the same persistent field "
                        "instead of recomputing or substituting a placeholder."
                    ),
                }
            )
    return candidates


def _derive_helper_static_scalar_candidates(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    called_method_static_data = entry.get("called_method_static_data")
    if not isinstance(called_method_static_data, dict):
        return []
    target_fields = _target_persistent_field_names(entry)
    args = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
    if not target_fields or not isinstance(args, dict):
        return []

    candidates: List[Dict[str, Any]] = []
    numeric_components: List[float] = []
    seen: set = set()
    for helper_name, helper_static in called_method_static_data.items():
        if not isinstance(helper_static, dict):
            continue
        for static_key, static_value in helper_static.items():
            derived_value: Optional[Any] = None
            if isinstance(static_value, dict):
                if {"condition", "if_true", "if_false"}.issubset(static_value.keys()):
                    derived_value = _evaluate_numeric_rule_candidate(static_value, args)
                else:
                    derived_value = _match_numeric_lookup_candidate(static_value, args)
            if not isinstance(derived_value, (int, float)):
                continue
            numeric_components.append(float(derived_value))
            marker = (helper_name, static_key, derived_value)
            if marker in seen:
                continue
            seen.add(marker)
            candidates.append(
                {
                    "source_tool": helper_name,
                    "source_key": static_key,
                    "value": derived_value,
                    "usage_policy": "Use this helper-derived scalar candidate when populating a missing computed write field.",
                }
            )

    cost_fields = [field for field in target_fields if "cost" in field.lower()]
    if len(cost_fields) == 1 and len(numeric_components) >= 2:
        product_value = 1.0
        for component in numeric_components:
            product_value *= component
        marker = ("derived_product", cost_fields[0], product_value)
        if marker not in seen:
            candidates.append(
                {
                    "field": cost_fields[0],
                    "value": product_value,
                    "source_tool": "called_method_static_data",
                    "source_key": "derived_product",
                    "usage_policy": "Use this helper-derived computed scalar for the matching persisted cost field instead of defaulting to 0 or a partial base lookup.",
                }
            )
    return candidates


def _extract_dynamic_key_shape_hints(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    hints: List[Dict[str, Any]] = []
    persistent_state_shapes = entry.get("persistent_state_shapes")
    if not isinstance(persistent_state_shapes, list):
        return hints
    for shape in persistent_state_shapes:
        if not isinstance(shape, dict):
            continue
        fields = shape.get("fields")
        path = shape.get("path")
        if not isinstance(fields, dict) or not isinstance(path, str) or not path.strip():
            continue
        for raw_key, raw_value in fields.items():
            if not isinstance(raw_key, str):
                continue
            match = re.fullmatch(r"<dynamic_key:([^>]+)>", raw_key.strip())
            if not match:
                continue
            hints.append(
                {
                    "path": path.strip(),
                    "dynamic_key_param": match.group(1).strip(),
                    "value_template": raw_value,
                    "collection_policy": (
                        "Append/remove whole top-level collection items using this dynamic-key object shape. "
                        "Do not merge into an existing same-key item's nested value unless the value_template itself is a list or object."
                    ),
                }
            )
    return hints


def _extract_set_membership_hints(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    hints: List[Dict[str, Any]] = []
    persistent_state_shapes = entry.get("persistent_state_shapes")
    result = entry.get("result")
    if not isinstance(persistent_state_shapes, list) or not isinstance(result, dict):
        return hints

    scalar_result_candidates: List[Tuple[str, Any]] = []
    for key, value in result.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, (str, int, float, bool)) or isinstance(value, bool):
            continue
        lowered = key.lower()
        if lowered == "id" or lowered.endswith("_id"):
            scalar_result_candidates.append((key, value))

    if not scalar_result_candidates:
        return hints

    for shape in persistent_state_shapes:
        if not isinstance(shape, dict):
            continue
        key_rule = shape.get("key_rule")
        path = shape.get("path")
        if not isinstance(key_rule, str) or "set membership" not in key_rule.lower():
            continue
        if not isinstance(path, str) or not path.strip():
            continue
        source_field, value = scalar_result_candidates[0]
        hints.append(
            {
                "path": path.strip(),
                "source_field": source_field,
                "value": value,
                "collection_policy": (
                    "Persist this scalar value as a collection member/item. Do not convert it into a boolean-valued mapping entry."
                ),
            }
        )
    return hints


def _extract_latest_match_hints(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    hints: List[Dict[str, Any]] = []
    persistent_state_shapes = entry.get("persistent_state_shapes")
    arguments = entry.get("arguments")
    if not isinstance(persistent_state_shapes, list) or not isinstance(arguments, dict):
        return hints

    text_fragments: List[str] = []
    for hint in entry.get("behavior_hints", []) or []:
        if isinstance(hint, str):
            text_fragments.append(hint)
    success_templates = entry.get("success_string_templates")
    if isinstance(success_templates, dict):
        for values in success_templates.values():
            if isinstance(values, list):
                text_fragments.extend(value for value in values if isinstance(value, str))
    combined_text = " ".join(text_fragments).lower()
    if not any(token in combined_text for token in ("latest", "most recent", "last ")):
        return hints

    for shape in persistent_state_shapes:
        if not isinstance(shape, dict):
            continue
        path = shape.get("path")
        fields = shape.get("fields")
        if not isinstance(path, str) or not path.strip() or not isinstance(fields, dict):
            continue
        match_keys = [key for key in arguments if isinstance(key, str) and key in fields]
        if not match_keys:
            match_keys = [key for key in arguments if isinstance(key, str) and (key == "id" or key.endswith("_id"))]
        hints.append(
            {
                "path": path.strip(),
                "match_policy": "latest",
                "match_key_candidates": match_keys,
                "selection_policy": "Choose the last matching collection item in current state when applying this removal/update.",
            }
        )
    return hints


def _resolve_shape_container_node(previous_state: Dict[str, Any], toolkit: str, shape_path: str) -> Any:
    logical_path = f"{toolkit}/{shape_path}" if not shape_path.startswith(f"{toolkit}/") else shape_path
    _, node = resolve_direct_logical_path(previous_state, logical_path)
    return node


def _resolved_record_from_shape(
    previous_state: Dict[str, Any],
    toolkit: str,
    shape: Dict[str, Any],
    args: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    path = shape.get("path")
    key_rule = shape.get("key_rule")
    if not isinstance(path, str) or not path.strip() or not isinstance(key_rule, str) or not key_rule.strip():
        return None
    container = _resolve_shape_container_node(previous_state, toolkit, path.strip())
    key_name = key_rule.strip()
    if key_name not in args:
        return None
    key_value = args.get(key_name)
    if isinstance(container, dict):
        lookup_key = str(key_value)
        if lookup_key not in container or not isinstance(container[lookup_key], dict):
            return None
        return {
            "path": path.strip(),
            "lookup_field": key_name,
            "lookup_value": key_value,
            "record": copy.deepcopy(container[lookup_key]),
        }
    if isinstance(container, list):
        for item in container:
            if not isinstance(item, dict):
                continue
            item_value = item.get(key_name)
            if item_value == key_value or str(item_value) == str(key_value):
                return {
                    "path": path.strip(),
                    "lookup_field": key_name,
                    "lookup_value": key_value,
                    "record": copy.deepcopy(item),
                }
        return None
    return None


def resolved_persistent_records_from_entry(
    previous_state: Dict[str, Any],
    entry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    toolkit = entry.get("toolkit") if isinstance(entry.get("toolkit"), str) else ""
    args = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
    if not toolkit or not isinstance(args, dict):
        return []
    records: List[Dict[str, Any]] = []
    seen: set = set()
    for shape in entry.get("persistent_state_shapes", []) or []:
        if not isinstance(shape, dict):
            continue
        resolved = _resolved_record_from_shape(previous_state, toolkit, shape, args)
        if not resolved:
            continue
        marker = (resolved["path"], resolved["lookup_field"], str(resolved["lookup_value"]))
        if marker in seen:
            continue
        seen.add(marker)
        resolved["usage_policy"] = (
            "Use this exact current-state record as the canonical source for derived mutations tied to the same lookup key."
        )
        records.append(resolved)
    return records


def annotate_prompt_tool_calls_with_hints(
    prompt_tool_calls: Sequence[Dict[str, Any]],
    previous_state: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Enrich each write entry with contract-grounded derived hints.

    The caller is responsible for attaching (where available):
    ``toolkit``, ``arguments``, ``result``, ``persistent_state_shapes``,
    ``state_effects`` / ``state_effects_on_success``, ``method_calls``,
    ``called_method_static_data``, ``behavior_hints``,
    ``success_string_templates``, ``matched_context_only_read_calls``.

    Context-only reads (entries whose ``state_persistence_role`` is
    ``context_only_read``) are passed through unchanged.
    """
    annotated_calls: List[Dict[str, Any]] = []
    for entry in prompt_tool_calls:
        if not isinstance(entry, dict):
            annotated_calls.append(entry)
            continue
        annotated_entry = copy.deepcopy(entry)
        is_read_only = annotated_entry.get("state_persistence_role") == "context_only_read"
        if not is_read_only:
            derived_candidates = _derive_matched_read_scalar_candidates(annotated_entry)
            if derived_candidates:
                annotated_entry["matched_read_scalar_candidates"] = derived_candidates
            helper_static_candidates = _derive_helper_static_scalar_candidates(annotated_entry)
            if helper_static_candidates:
                annotated_entry["helper_static_scalar_candidates"] = helper_static_candidates
            dynamic_key_shape_hints = _extract_dynamic_key_shape_hints(annotated_entry)
            if dynamic_key_shape_hints:
                annotated_entry["dynamic_key_shape_hints"] = dynamic_key_shape_hints
            set_membership_hints = _extract_set_membership_hints(annotated_entry)
            if set_membership_hints:
                annotated_entry["set_membership_hints"] = set_membership_hints
            latest_match_hints = _extract_latest_match_hints(annotated_entry)
            if latest_match_hints:
                annotated_entry["latest_match_hints"] = latest_match_hints
            if isinstance(previous_state, dict):
                resolved_records = resolved_persistent_records_from_entry(previous_state, annotated_entry)
                if resolved_records:
                    annotated_entry["resolved_persistent_records"] = resolved_records
                resolved_effects = resolve_effect_hints(previous_state, annotated_entry)
                if resolved_effects:
                    annotated_entry["resolved_effect_hints"] = resolved_effects
        annotated_calls.append(annotated_entry)
    return annotated_calls

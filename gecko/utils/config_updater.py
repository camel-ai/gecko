import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from camel.agents import ChatAgent
from utils.model_utils import create_model, sanitize_llm_json_text
from .global_config import get_state_model, get_agent_timeout
import json_repair

logger = logging.getLogger(__name__)


def deep_merge(target: Any, source: Any) -> Any:
    """Apply an RFC 7396 JSON Merge Patch object to a target value.

    NOTE:
    - If ``source`` is not a dict, it replaces ``target`` entirely.
    - If ``source`` is a dict (JSON object), keys with value ``None`` (JSON null)
      delete the key from ``target`` (if present).
    - Nested dicts are patched recursively; non-dict values overwrite.
    """
    if not isinstance(source, dict):
        return source
    if not isinstance(target, dict):
        # If the target is not an object, the patch replaces it with a new object.
        target = {}

    import copy

    for key, value in source.items():
        if value is None:
            target.pop(key, None)
            continue

        if isinstance(value, dict):
            existing = target.get(key)
            if isinstance(existing, dict):
                target[key] = deep_merge(existing, value)
            else:
                target[key] = copy.deepcopy(value)
            continue

        target[key] = value

    return target


def _json_pointer_unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _split_json_pointer(path: str) -> List[str]:
    if path == "":
        return []
    if not path.startswith("/"):
        raise ValueError(f"Invalid JSON Pointer (must start with '/'): {path}")
    parts = path.lstrip("/").split("/")
    return [_json_pointer_unescape(p) for p in parts]


def _parse_list_index(token: str, length: int, allow_end: bool = False) -> int:
    if token == "-" and allow_end:
        return length
    try:
        idx = int(token)
    except Exception as exc:
        raise ValueError(f"Invalid list index token: {token}") from exc
    if idx < 0:
        raise IndexError(f"Negative list index not allowed: {idx}")
    if allow_end:
        if idx > length:
            raise IndexError(f"List index out of range: {idx} > {length}")
    else:
        if idx >= length:
            raise IndexError(f"List index out of range: {idx} >= {length}")
    return idx


def _get_container(target: Any, pointer: str, create_missing: bool = False) -> Tuple[Any, str]:
    parts = _split_json_pointer(pointer)
    if not parts:
        return None, ""
    *parent_parts, last = parts
    current = target
    for part in parent_parts:
        if isinstance(current, list):
            idx = _parse_list_index(part, len(current), allow_end=False)
            current = current[idx]
        elif isinstance(current, dict):
            if part not in current:
                if create_missing:
                    current[part] = {}
                else:
                    raise KeyError(f"Missing parent path segment: {part}")
            current = current[part]
        else:
            raise TypeError(f"Cannot traverse into non-container node at segment '{part}'")
    return current, last


def _get_by_pointer(target: Any, pointer: str) -> Any:
    parts = _split_json_pointer(pointer)
    current = target
    for part in parts:
        if isinstance(current, list):
            idx = _parse_list_index(part, len(current), allow_end=False)
            current = current[idx]
        elif isinstance(current, dict):
            current = current[part]
        else:
            raise TypeError(f"Cannot traverse into non-container node at segment '{part}'")
    return current


def _add_by_pointer(target: Any, pointer: str, value: Any, create_missing: bool = True) -> Any:
    if pointer == "":
        return value
    container, key = _get_container(target, pointer, create_missing=create_missing)
    if isinstance(container, list):
        idx = _parse_list_index(key, len(container), allow_end=True)
        if idx == len(container):
            container.append(value)
        else:
            container.insert(idx, value)
    elif isinstance(container, dict):
        container[key] = value
    else:
        raise TypeError(f"Cannot add under non-container at path: {pointer}")
    return target


def _remove_by_pointer(target: Any, pointer: str) -> Any:
    if pointer == "":
        raise ValueError("Cannot remove document root")
    container, key = _get_container(target, pointer, create_missing=False)
    if isinstance(container, list):
        idx = _parse_list_index(key, len(container), allow_end=False)
        del container[idx]
    elif isinstance(container, dict):
        if key not in container:
            raise KeyError(f"Path does not exist for remove: {pointer}")
        del container[key]
    else:
        raise TypeError(f"Cannot remove under non-container at path: {pointer}")
    return target


def _replace_by_pointer(target: Any, pointer: str, value: Any) -> Any:
    if pointer == "":
        return value
    container, key = _get_container(target, pointer, create_missing=False)
    if isinstance(container, list):
        idx = _parse_list_index(key, len(container), allow_end=False)
        container[idx] = value
    elif isinstance(container, dict):
        if key not in container:
            raise KeyError(f"Path does not exist for replace: {pointer}")
        container[key] = value
    else:
        raise TypeError(f"Cannot replace under non-container at path: {pointer}")
    return target


def apply_json_patch(target: Any, patch_ops: List[Dict[str, Any]]) -> Any:
    """Apply an RFC 6902 JSON Patch list to the target with LLM tolerance.

    Tolerance features (handles common LLM mistakes):
    - add: auto-creates missing parent objects along the path.
    - replace on missing path: falls back to add (LLMs often confuse the two).
    - remove on missing path: silently skips (idempotent delete).
    - move/copy destination: auto-creates missing parents for the target path.
    """
    import copy

    if not isinstance(patch_ops, list):
        raise ValueError("JSON Patch payload must be a list of operations")

    doc = copy.deepcopy(target)
    for idx, op in enumerate(patch_ops):
        if not isinstance(op, dict):
            raise ValueError(f"Patch op at index {idx} must be an object")
        op_name = op.get("op")
        path = op.get("path")
        if not isinstance(path, str):
            raise ValueError(f"Patch op at index {idx} must include string 'path'")
        try:
            if op_name == "add":
                if "value" not in op:
                    raise ValueError(f"JSON Patch 'add' at index {idx} requires 'value'")
                doc = _add_by_pointer(doc, path, op.get("value"), create_missing=True)
            elif op_name == "remove":
                try:
                    doc = _remove_by_pointer(doc, path)
                except (KeyError, IndexError):
                    logger.debug(f"Patch op {idx}: remove at '{path}' — path not found, skipping")
            elif op_name == "replace":
                if "value" not in op:
                    raise ValueError(f"JSON Patch 'replace' at index {idx} requires 'value'")
                try:
                    doc = _replace_by_pointer(doc, path, op.get("value"))
                except (KeyError, IndexError):
                    # Fallback: treat as add when path doesn't exist
                    logger.debug(f"Patch op {idx}: replace at '{path}' — path not found, falling back to add")
                    doc = _add_by_pointer(doc, path, op.get("value"), create_missing=True)
            elif op_name == "move":
                from_path = op.get("from")
                if not isinstance(from_path, str):
                    raise ValueError(f"JSON Patch 'move' at index {idx} requires string 'from'")
                try:
                    value = _get_by_pointer(doc, from_path)
                    doc = _remove_by_pointer(doc, from_path)
                    doc = _add_by_pointer(doc, path, value, create_missing=True)
                except (KeyError, IndexError):
                    logger.debug(f"Patch op {idx}: move from '{from_path}' — source not found, skipping")
            elif op_name == "copy":
                from_path = op.get("from")
                if not isinstance(from_path, str):
                    raise ValueError(f"JSON Patch 'copy' at index {idx} requires string 'from'")
                value = _get_by_pointer(doc, from_path)
                doc = _add_by_pointer(doc, path, copy.deepcopy(value), create_missing=True)
            elif op_name == "test":
                expected = op.get("value")
                actual = _get_by_pointer(doc, path)
                if actual != expected:
                    raise ValueError(f"JSON Patch 'test' failed at path {path}")
            else:
                raise ValueError(f"Unsupported JSON Patch op at index {idx}: {op_name}")
        except (KeyError, IndexError, TypeError) as exc:
            logger.warning(f"Patch op {idx} ({op_name} at '{path}') failed: {exc} — skipping")
    return doc


# ---------------------------------------------------------------------------
# Function-calling state update
# ---------------------------------------------------------------------------

_FC_SYSTEM_PROMPT = """You are a state tracker. Given the previous state and a sequence of tool calls with their results, determine what state mutations occurred and call the provided tools to apply them.

Rules:
1. Read/query operations → no_state_change(). Write-like operations (post/create/update/delete/add/remove/send/set/move/...) that succeed MUST produce state mutations. no_state_change() is valid ONLY for truly read-only operations or explicit error responses.
2. Use tool results as source of truth. Process calls in listed order.
3. If a "Required state effects" section is provided, you MUST realize every listed effect. no_state_change() is invalid for those calls.
4. State location: top-level /<ToolkitName>/... is canonical for domain data (tweets, orders, balances, counters, collections). runtime_state is ONLY for transient context (current_working_directory). Never put domain data in runtime_state.
5. When a successful write operation creates or returns an entity (with fields like id, content, tags, etc.), persist it in the appropriate canonical collection using add_entry, and update related counters/balances consistently.
6. When a successful response returns fields matching existing canonical state keys, update those state values to match (e.g. if response has {"remainingUnlockedDoors": 0} and state has "remainingUnlockedDoors": 4, emit replace_entry).
7. On auth/login success, persist authentication flags in canonical toolkit state (not runtime-only).
8. For filesystem toolkits: context-changing ops (cd/navigation) → set_runtime_field() with absolute paths; move/rename → move_entry() (atomic, never duplicate); scope from earlier calls affects subsequent ones.

Path format for entry tools (add/remove/move/replace):
- Use absolute paths from config root: '<ToolkitName>/<key>/...' (e.g. 'TwitterAPI/tweets', 'TradingBot/orders/12345', 'GorillaFileSystem/root/workspace/document').
- Paths can also be relative to the current working scope when a filesystem toolkit is active.
- For filesystem toolkits, omit structural 'contents' wrappers — the resolver handles them.
"""


# -- Tool functions (signatures only; never actually executed) ---------------

def _fc_set_runtime_field(toolkit: str, key: str, value: str) -> str:
    """Set a runtime state field for a toolkit.
    Use ONLY for transient context changes (e.g. current_working_directory).
    Never use this for domain/business entities or counters (e.g. tweets, comments, orders, balances).
    value must be the definitive form (e.g. absolute path, not relative).
    """
    return "ok"


def _fc_add_entry(parent_path: str, name: str, value_json: str) -> str:
    """Add a new child entry under parent_path.
    parent_path: path with '/' separators (e.g. 'TwitterAPI/tweets', 'GorillaFileSystem/root/workspace').
    name: key name for the new entry (e.g. '1', 'report.txt').
    value_json: JSON string of the value to insert — any valid JSON (object, array, string, number, etc.).
    """
    return "ok"


def _fc_remove_entry(entry_path: str) -> str:
    """Remove the entry at entry_path.
    entry_path: logical path with '/' separators (e.g. 'TradingBot/watch_list/0', 'GorillaFileSystem/root/workspace/file.txt').
    """
    return "ok"


def _fc_move_entry(source_path: str, dest_parent_path: str, new_name: str = "") -> str:
    """Move an entry to a new parent, optionally renaming it.
    source_path: logical path of the entry to move.
    dest_parent_path: logical path of the destination parent.
    new_name: optional new name for move+rename semantics.
    """
    return "ok"


def _fc_replace_entry(entry_path: str, new_value_json: str) -> str:
    """Replace the value at entry_path.
    entry_path: logical path with '/' separators (e.g. 'TwitterAPI/authenticated', 'TradingBot/account_info/balance').
    new_value_json: JSON string of the replacement value.
    """
    return "ok"


def _fc_no_state_change() -> str:
    """Call when all operations are read-only and no state mutation is needed."""
    return "ok"


def _build_fc_tools():
    """Build FunctionTool list for function-calling state update mode."""
    from camel.toolkits import FunctionTool
    return [
        FunctionTool(_fc_set_runtime_field),
        FunctionTool(_fc_add_entry),
        FunctionTool(_fc_remove_entry),
        FunctionTool(_fc_move_entry),
        FunctionTool(_fc_replace_entry),
        FunctionTool(_fc_no_state_change),
    ]


def _result_indicates_error(result: Any) -> bool:
    """Heuristic: determine whether a tool-call result indicates failure."""
    status, _ = classify_tool_call_status(result)
    return status == "error"


def classify_tool_call_status(result: Any) -> Tuple[str, Optional[str]]:
    """Classify tool call outcome from result payload.

    Returns:
        (status, reason)
        - status: "success" | "error"
        - reason: optional textual error reason
    """
    if isinstance(result, dict):
        err = result.get("error")
        if err:
            return "error", str(err)

        if result.get("success") is False:
            msg = result.get("message") or "success=false"
            return "error", str(msg)

        detail = result.get("detail")
        if isinstance(detail, dict):
            em = detail.get("error_message")
            if em:
                return "error", str(em)
        elif isinstance(detail, str) and detail:
            return "error", detail

    if isinstance(result, str) and "error" in result.lower():
        return "error", result

    return "success", None


def _has_error_state_effects(
    tool_name: str,
    tool_descriptions: Optional[Dict[str, Any]],
) -> bool:
    """Whether schema hints explicitly allow state changes on error for this tool."""
    if not isinstance(tool_descriptions, dict):
        return False
    desc_entry = tool_descriptions.get(tool_name)
    if not isinstance(desc_entry, dict):
        return False
    state_hints = desc_entry.get("state_hints")
    if not isinstance(state_hints, dict):
        return False

    error_effects = state_hints.get("state_effects_on_error")
    always_effects = state_hints.get("state_effects_always")
    return bool(error_effects) or bool(always_effects)


def _prepare_tool_calls_for_state_update(
    tool_calls: List[Dict[str, Any]],
    tool_descriptions: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Normalize tool-call status and filter out unsupported error calls."""
    prepared: List[Dict[str, Any]] = []
    skipped_error_calls = 0

    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        call_name = str(call.get("name") or call.get("function") or "").strip()
        if not call_name:
            continue

        status, reason = classify_tool_call_status(call.get("result"))
        normalized = dict(call)
        normalized.setdefault("execution_status", status)
        if reason and "error_reason" not in normalized:
            normalized["error_reason"] = reason

        # Default: error calls should not mutate state unless explicitly modeled.
        if status == "error" and not _has_error_state_effects(call_name, tool_descriptions):
            skipped_error_calls += 1
            continue

        prepared.append(normalized)

    return prepared, skipped_error_calls


def _extract_required_state_effects(
    tool_calls: List[Dict[str, Any]],
    tool_descriptions: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract required state effects for successful calls from tool descriptions."""
    if not isinstance(tool_descriptions, dict):
        return []

    requirements: List[Dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        call_name = call.get("name")
        if not isinstance(call_name, str) or not call_name:
            continue
        if _result_indicates_error(call.get("result")):
            continue

        desc_entry = tool_descriptions.get(call_name)
        if not isinstance(desc_entry, dict):
            continue
        state_hints = desc_entry.get("state_hints")
        if not isinstance(state_hints, dict):
            continue

        # Prefer branch-aware success effects; fall back to legacy state_effects.
        # Also include always-effects on successful calls.
        success_effects = state_hints.get("state_effects_on_success")
        always_effects = state_hints.get("state_effects_always")
        if isinstance(success_effects, list) and success_effects:
            state_effects = list(success_effects)
        else:
            fallback_effects = state_hints.get("state_effects")
            state_effects = list(fallback_effects) if isinstance(fallback_effects, list) else []
        if isinstance(always_effects, list) and always_effects:
            state_effects.extend(always_effects)
        if not isinstance(state_effects, list) or not state_effects:
            continue

        normalized_effects: List[str] = []
        seen_effects: set[str] = set()
        for effect in state_effects:
            if not isinstance(effect, str) or not effect.strip():
                continue
            normalized = " ".join(effect.split())
            if normalized in seen_effects:
                continue
            seen_effects.add(normalized)
            normalized_effects.append(normalized)
        if not normalized_effects:
            continue

        toolkit_name = ""
        toolkit_info = desc_entry.get("toolkit")
        if isinstance(toolkit_info, dict):
            name = toolkit_info.get("name")
            if isinstance(name, str):
                toolkit_name = name

        requirements.append(
            {
                "call_name": call_name,
                "toolkit": toolkit_name,
                "state_effects": normalized_effects,
            }
        )

    return requirements


def _extract_fc_request_names(fc_requests: List[Any]) -> List[str]:
    """Extract function-call tool names from FC requests."""
    req_names: List[str] = []
    for req in fc_requests:
        if hasattr(req, "tool_name") and isinstance(req.tool_name, str):
            req_names.append(req.tool_name)
        elif isinstance(req, dict):
            func = req.get("function")
            if isinstance(func, dict) and isinstance(func.get("name"), str):
                req_names.append(func["name"])
            elif isinstance(req.get("tool_name"), str):
                req_names.append(req["tool_name"])
            elif isinstance(req.get("name"), str):
                req_names.append(req["name"])
    return req_names


def _parse_counter_delta(effect: str) -> Optional[Tuple[str, int]]:
    """Parse counter delta effects like 'increment `tweet_counter` by 1'."""
    if not isinstance(effect, str):
        return None
    patterns = [
        (r"increment\s+`([^`]+)`\s+by\s+(\d+)", +1),
        (r"increase\s+`([^`]+)`\s+by\s+(\d+)", +1),
        (r"decrement\s+`([^`]+)`\s+by\s+(\d+)", -1),
        (r"decrease\s+`([^`]+)`\s+by\s+(\d+)", -1),
    ]
    lowered = effect.lower()
    for pattern, sign in patterns:
        m = re.search(pattern, lowered)
        if not m:
            continue
        field = m.group(1).strip()
        amount = int(m.group(2))
        if not field:
            return None
        return field, sign * amount
    return None


def _apply_counter_effect_guards(
    previous_state: Dict[str, Any],
    state_result: Dict[str, Any],
    required_state_effects: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply deterministic guard for counter delta effects if LLM missed them."""
    import copy as _copy

    guard_ops: List[Dict[str, Any]] = []
    for requirement in required_state_effects:
        if not isinstance(requirement, dict):
            continue
        toolkit = requirement.get("toolkit")
        if not isinstance(toolkit, str) or not toolkit:
            continue
        effects = requirement.get("state_effects")
        if not isinstance(effects, list):
            continue

        prev_toolkit = previous_state.get(toolkit)
        curr_toolkit = state_result.get(toolkit)
        if not isinstance(prev_toolkit, dict) or not isinstance(curr_toolkit, dict):
            continue

        for effect in effects:
            parsed = _parse_counter_delta(effect if isinstance(effect, str) else "")
            if not parsed:
                continue
            field, delta = parsed
            prev_value = prev_toolkit.get(field)
            curr_value = curr_toolkit.get(field)
            if not isinstance(prev_value, int):
                continue
            expected = prev_value + delta
            if curr_value == expected:
                continue

            op = "replace" if field in curr_toolkit else "add"
            guard_ops.append(
                {"op": op, "path": f"/{toolkit}/{field}", "value": expected}
            )

    if not guard_ops:
        return state_result

    logger.info(
        "[FC STATE GUARD] Applying %d counter-effect guard ops: %s",
        len(guard_ops),
        json.dumps(guard_ops, ensure_ascii=False),
    )
    return apply_json_patch(_copy.deepcopy(state_result), guard_ops)


def _get_toolkit_name_for_call(
    call_name: str,
    tool_descriptions: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Extract toolkit name for a tool call from tool_descriptions."""
    if not isinstance(tool_descriptions, dict):
        return None
    desc_entry = tool_descriptions.get(call_name)
    if not isinstance(desc_entry, dict):
        return None
    toolkit_info = desc_entry.get("toolkit")
    if isinstance(toolkit_info, dict):
        name = toolkit_info.get("name")
        if isinstance(name, str) and name:
            return name
    return None


def _auto_merge_response_fields(
    state_result: Dict[str, Any],
    tool_calls: List[Dict[str, Any]],
    tool_descriptions: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Auto-merge successful tool response fields into matching canonical state keys.

    For each successful tool call, if the response contains fields whose names
    match existing keys in the toolkit's canonical state, overwrite the state
    values with the response values.  This ensures response-state consistency
    even when the LLM state_model misses some updates.

    Also syncs merged values to runtime_state if the same key exists there.
    """
    import copy as _copy

    result = _copy.deepcopy(state_result)
    merged_count = 0

    for call in tool_calls:
        call_result = call.get("result")
        if not isinstance(call_result, dict):
            continue
        # Skip error responses
        if "error" in call_result:
            continue

        call_name = call.get("name", "")
        toolkit_name = _get_toolkit_name_for_call(call_name, tool_descriptions)

        # Fallback: if tool_descriptions doesn't provide toolkit name,
        # try to infer from state keys (skip runtime_state)
        if not toolkit_name:
            candidate_toolkits = [
                k for k in result.keys()
                if k != "runtime_state" and isinstance(result.get(k), dict)
            ]
            if len(candidate_toolkits) == 1:
                toolkit_name = candidate_toolkits[0]

        if not toolkit_name:
            continue

        toolkit_state = result.get(toolkit_name)
        if not isinstance(toolkit_state, dict):
            continue

        runtime_toolkit = (
            result.get("runtime_state", {})
            .get("toolkits", {})
            .get(toolkit_name)
        )

        for key, value in call_result.items():
            # Merge into canonical state if key already exists
            if key in toolkit_state:
                old_val = toolkit_state[key]
                if isinstance(value, dict) and isinstance(old_val, dict):
                    old_val.update(value)
                else:
                    toolkit_state[key] = value
                merged_count += 1

            # Sync to runtime_state copy if same key exists there
            if isinstance(runtime_toolkit, dict) and key in runtime_toolkit:
                old_rt_val = runtime_toolkit[key]
                if isinstance(value, dict) and isinstance(old_rt_val, dict):
                    old_rt_val.update(value)
                else:
                    runtime_toolkit[key] = value

    if merged_count > 0:
        logger.info(
            "[AUTO-MERGE] Merged %d response fields into canonical state",
            merged_count,
        )

    return result


# -- Path resolver -----------------------------------------------------------

def resolve_logical_path(state: Dict[str, Any], logical_path: str) -> Tuple[str, Any]:
    """Resolve a logical path (without 'contents' wrappers) to a JSON Pointer.

    Walks the actual config tree, auto-detecting nodes that use a
    ``{"type": ..., "contents": {...}}`` directory pattern and inserting
    ``/contents`` in the pointer as needed.

    Args:
        state: The current config dict.
        logical_path: Slash-separated key names, e.g.
            ``GorillaFileSystem/root/workspace/document/report.txt``

    Returns:
        (json_pointer, node_value) — the resolved JSON Pointer string and the
        value at that location (or ``None`` if the path is new / not found).
    """
    segments = [s for s in logical_path.strip("/").split("/") if s]
    pointer_parts: List[str] = []
    current: Any = state

    for seg in segments:
        if not isinstance(current, dict):
            # Past known tree — append remaining segments literally.
            pointer_parts.append(seg)
            current = None
            continue

        if seg in current:
            pointer_parts.append(seg)
            current = current[seg]
        elif "contents" in current and isinstance(current.get("contents"), dict):
            # Auto-insert /contents wrapper.
            pointer_parts.append("contents")
            pointer_parts.append(seg)
            current = current["contents"].get(seg)
        else:
            # New entry — check if parent uses contents pattern.
            pointer_parts.append(seg)
            current = None

    return "/" + "/".join(pointer_parts), current


def _child_pointer(state: Dict[str, Any], parent_logical_path: str, child_name: str) -> str:
    """Get JSON Pointer for adding a child under a parent logical path."""
    parent_pointer, parent_node = resolve_logical_path(state, parent_logical_path)
    if (
        isinstance(parent_node, dict)
        and "contents" in parent_node
        and isinstance(parent_node.get("contents"), dict)
    ):
        return f"{parent_pointer}/contents/{child_name}"
    return f"{parent_pointer}/{child_name}"


# -- CWD-relative path resolution helpers ------------------------------------

def _find_cwd_prefix(running: Dict[str, Any]) -> Optional[List[str]]:
    """Find [toolkit_name, root_key, *cwd_segments] from runtime_state.

    Returns None if no toolkit with a current_working_directory is found.
    """
    rt = running.get("runtime_state", {}).get("toolkits", {})
    for toolkit_name, toolkit_rt in rt.items():
        if not isinstance(toolkit_rt, dict):
            continue
        cwd = toolkit_rt.get("current_working_directory")
        if not isinstance(cwd, str) or not cwd:
            continue

        toolkit_data = running.get(toolkit_name)
        if not isinstance(toolkit_data, dict):
            continue

        # Find the root key (first dict child, usually "root").
        root_key = None
        for k, v in toolkit_data.items():
            if isinstance(v, dict):
                root_key = k
                break
        if root_key is None:
            continue

        cwd_segments = [s for s in cwd.strip("/").split("/") if s]
        return [toolkit_name, root_key] + cwd_segments

    return None


def _collapse_redundant_root_suffix(path_str: str) -> str:
    """Collapse duplicated toolkit-root suffixes in malformed logical paths.

    Example:
    ``GorillaFileSystem/root/workspace/root/workspace`` ->
    ``GorillaFileSystem/root/workspace``
    """
    segments = [s for s in path_str.strip("/").split("/") if s]
    if len(segments) < 5:
        return path_str

    # Expected shape starts with "<toolkit>/<root>/..."
    root_idx = 1
    if len(segments) <= root_idx:
        return path_str
    root_key = segments[root_idx]

    # Detect: <toolkit>/<root>/<A...>/<root>/<A...>
    for split in range(root_idx + 2, len(segments)):
        if segments[split] != root_key:
            continue
        left = segments[root_idx + 1 : split]
        right = segments[split + 1 :]
        if left and left == right:
            return "/".join(segments[:split])

    return path_str


def _resolve_entry_path(running: Dict[str, Any], raw_path: str) -> str:
    """Resolve a potentially ambiguous path to a full logical path.

    Handles three styles that weak models produce:
    1. Full absolute: ``GorillaFileSystem/root/workspace/document``
    2. Partial absolute (from root key): ``workspace/document``
    3. CWD-relative: ``final_report.pdf``, ``temp``

    Uses tree-based disambiguation: builds candidate paths and picks the
    one that resolves to an existing node (or whose parent resolves).
    """
    segments = [s for s in raw_path.strip("/").split("/") if s]

    # Already absolute if first segment is a top-level config key.
    if segments:
        top_keys = set(running.keys()) - {"runtime_state"}
        if segments[0] in top_keys:
            return raw_path

    prefix = _find_cwd_prefix(running)
    if prefix is None:
        # No runtime_state — try to match the first segment against toolkit
        # root keys so that "root/project/file.txt" resolves to
        # "GorillaFileSystem/root/project/file.txt".
        if segments:
            for tk_name, tk_data in running.items():
                if tk_name == "runtime_state" or not isinstance(tk_data, dict):
                    continue
                if segments[0] in tk_data:
                    candidate = "/".join([tk_name] + segments)
                    _, node = resolve_logical_path(running, candidate)
                    if node is not None:
                        logger.debug(
                            "[FC PATH] '%s' → '%s' (toolkit-root inferred, no runtime_state)",
                            raw_path, candidate,
                        )
                        return candidate
                    # Also check if parent resolves (for new entries)
                    parent_candidate = "/".join([tk_name] + segments[:-1])
                    _, parent_node = resolve_logical_path(running, parent_candidate)
                    if parent_node is not None:
                        logger.debug(
                            "[FC PATH] '%s' → '%s' (toolkit-root inferred, parent exists)",
                            raw_path, candidate,
                        )
                        return candidate
        return raw_path

    toolkit_name, root_key = prefix[:2]
    toolkit_root = [toolkit_name, root_key]

    # Build candidate paths from most specific to least specific.
    candidates: List[Tuple[str, List[str]]] = []
    if segments:
        # If path starts at toolkit root key (e.g. "root/workspace/..."),
        # treat it as toolkit-root absolute instead of CWD-relative.
        if segments[0] == root_key:
            candidates.append(("toolkit_root_absolute", [toolkit_name] + segments))
            candidates.append(("cwd_relative", prefix + segments))
        else:
            candidates.append(("cwd_relative", prefix + segments))
        candidates.append(("root_relative", toolkit_root + segments))
    else:
        # Empty path = CWD itself.
        candidates.append(("cwd_self", list(prefix)))

    # Round 1: pick the first candidate whose leaf resolves to an existing node.
    for label, parts in candidates:
        path_str = "/".join(parts)
        _, node = resolve_logical_path(running, path_str)
        if node is not None:
            logger.debug("[FC PATH] '%s' → '%s' (%s, leaf exists)", raw_path, path_str, label)
            return path_str

    # Round 2: pick the first candidate whose parent resolves.
    for label, parts in candidates:
        if len(parts) <= 1:
            continue
        path_str = "/".join(parts)
        parent_str = "/".join(parts[:-1])
        _, parent_node = resolve_logical_path(running, parent_str)
        if parent_node is not None:
            logger.debug("[FC PATH] '%s' → '%s' (%s, parent exists)", raw_path, path_str, label)
            return path_str

    # Default: CWD-relative (first candidate).
    default = "/".join(candidates[0][1])
    logger.debug("[FC PATH] '%s' → '%s' (default)", raw_path, default)
    return default


def _normalize_cwd_value(running: Dict[str, Any], toolkit: str, raw_value: str) -> str:
    """Normalize a current_working_directory value to an absolute path.

    If *raw_value* already starts with ``/`` it is returned as-is.
    Otherwise it is resolved relative to the toolkit's current CWD.
    """
    if raw_value.startswith("/"):
        return raw_value

    current_cwd = (
        running
        .get("runtime_state", {})
        .get("toolkits", {})
        .get(toolkit, {})
        .get("current_working_directory", "/")
    )
    return current_cwd.rstrip("/") + "/" + raw_value


# -- Convert FC tool calls → JSON Patch ops ---------------------------------

def _fc_calls_to_patch(
    state: Dict[str, Any],
    fc_requests: List[Any],
) -> List[Dict[str, Any]]:
    """Translate function-calling tool requests into RFC 6902 JSON Patch ops.

    Maintains a running copy of the state so that later operations can resolve
    paths through entries created by earlier ones (e.g. mkdir then mv into it).

    Individual op errors are logged and skipped rather than raising, so that
    valid ops from the same batch still get applied (graceful degradation).

    Args:
        state: The config dict *before* this update (used for path resolution
            and for reading source values on move).
        fc_requests: List of tool-call request objects / dicts from ChatAgent.

    Returns:
        List of JSON Patch operation dicts ready for ``apply_json_patch()``.
    """
    import copy as _copy
    ops: List[Dict[str, Any]] = []
    # Running state: apply each op incrementally so subsequent resolutions
    # can see entries created by earlier ops (e.g. mkdir then mv into it).
    running = _copy.deepcopy(state)

    for req in fc_requests:
        # Normalize access — handle both object attrs and dict formats.
        if hasattr(req, "tool_name"):
            name = req.tool_name
            if not isinstance(name, str) or not name:
                logger.warning("[FC STATE] Invalid tool name in request object: %r — skipping", name)
                continue
            if not isinstance(req.args, dict):
                logger.warning("[FC STATE] Tool args must be dict for '%s', got %s — skipping", name, type(req.args).__name__)
                continue
            args = req.args
        elif isinstance(req, dict):
            func = req.get("function") or {}
            if isinstance(func, dict):
                name = func.get("name", "")
                raw_args = func.get("arguments", "{}")
            else:
                name = req.get("tool_name", req.get("name", ""))
                raw_args = req.get("args", req.get("arguments", "{}"))
            if not isinstance(name, str) or not name:
                logger.warning("[FC STATE] Missing tool name in request: %s — skipping", req)
                continue
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except Exception as exc:
                    logger.warning("[FC STATE] Invalid JSON arguments for tool '%s': %r — skipping (%s)", name, raw_args, exc)
                    continue
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                logger.warning("[FC STATE] Unsupported args type for tool '%s': %s — skipping", name, type(raw_args).__name__)
                continue
        else:
            logger.warning("[FC STATE] Unsupported FC request type: %s — skipping", type(req).__name__)
            continue

        step_ops: List[Dict[str, Any]] = []
        if name == "_fc_no_state_change":
            continue

        try:
            _fc_process_single_request(name, args, running, step_ops)
        except (ValueError, FileNotFoundError, TypeError, KeyError, IndexError) as exc:
            logger.warning("[FC STATE] Skipping failed op '%s': %s", name, exc)
            continue

        # Apply this step's ops to running state and collect them.
        for op in step_ops:
            ops.append(op)
            running = apply_json_patch(running, [op])

    return ops


def _fc_process_single_request(
    name: str,
    args: Dict[str, Any],
    running: Dict[str, Any],
    step_ops: List[Dict[str, Any]],
) -> None:
    """Process a single FC tool request into JSON Patch ops.

    Raises on error so the caller can catch and skip gracefully.
    """
    import copy as _copy

    if name == "_fc_set_runtime_field":
        toolkit = args.get("toolkit", "")
        key = args.get("key", "")
        value = args.get("value", "")
        if not isinstance(toolkit, str) or not toolkit:
            raise ValueError("[FC STATE] _fc_set_runtime_field requires non-empty 'toolkit'")
        if not isinstance(key, str) or not key:
            raise ValueError("[FC STATE] _fc_set_runtime_field requires non-empty 'key'")
        # Normalize CWD values: relative → absolute.
        if key == "current_working_directory" and isinstance(value, str):
            value = _normalize_cwd_value(running, toolkit, value)
        pointer = f"/runtime_state/toolkits/{toolkit}/{key}"
        step_ops.append({"op": "replace", "path": pointer, "value": value})

    elif name == "_fc_add_entry":
        parent_path = _resolve_entry_path(running, args.get("parent_path", ""))
        child_name = args.get("name", "")
        if not isinstance(child_name, str) or not child_name:
            raise ValueError("[FC STATE] _fc_add_entry requires non-empty 'name'")
        parent_pointer, parent_node = resolve_logical_path(running, parent_path)
        if not isinstance(parent_node, (dict, list)):
            fallback_path = _collapse_redundant_root_suffix(parent_path)
            if fallback_path != parent_path:
                fallback_pointer, fallback_node = resolve_logical_path(running, fallback_path)
                if isinstance(fallback_node, (dict, list)):
                    logger.debug(
                        "[FC PATH] repaired duplicated-root path: '%s' -> '%s'",
                        parent_path,
                        fallback_path,
                    )
                    parent_path = fallback_path
                    parent_pointer, parent_node = fallback_pointer, fallback_node
        if not isinstance(parent_node, (dict, list)):
            logger.warning(
                "[FC STATE] _fc_add_entry parent path not found: '%s' (pointer=%s), auto-creating empty object",
                parent_path,
                parent_pointer,
            )
            running = apply_json_patch(
                running,
                [{"op": "add", "path": parent_pointer, "value": {}}],
            )
            parent_pointer, parent_node = resolve_logical_path(running, parent_path)
            if not isinstance(parent_node, (dict, list)):
                raise FileNotFoundError(
                    f"[FC STATE] _fc_add_entry parent path not found after auto-create: '{parent_path}'"
                )
        value_json = args.get("value_json", "{}")
        if isinstance(value_json, str):
            try:
                value = json.loads(value_json)
            except Exception as exc:
                raise ValueError(
                    f"[FC STATE] _fc_add_entry invalid value_json: {value_json!r}"
                ) from exc
        else:
            value = value_json
        if isinstance(parent_node, list):
            # Allow list append semantics for collection-like states.
            if child_name in {"-", "append"}:
                pointer = f"{parent_pointer}/-"
            elif child_name.isdigit():
                pointer = f"{parent_pointer}/{child_name}"
            else:
                pointer = f"{parent_pointer}/-"
        else:
            pointer = _child_pointer(running, parent_path, child_name)
        step_ops.append({"op": "add", "path": pointer, "value": value})

    elif name == "_fc_remove_entry":
        entry_path = _resolve_entry_path(running, args.get("entry_path", ""))
        pointer, node = resolve_logical_path(running, entry_path)
        if node is None:
            raise FileNotFoundError(
                f"[FC STATE] _fc_remove_entry path not found: '{entry_path}'"
            )
        step_ops.append({"op": "remove", "path": pointer})

    elif name == "_fc_move_entry":
        source_path = _resolve_entry_path(running, args.get("source_path", ""))
        dest_parent_path = _resolve_entry_path(running, args.get("dest_parent_path", ""))
        new_name_raw = args.get("new_name", "")
        if new_name_raw is None:
            new_name_raw = ""
        if not isinstance(new_name_raw, str):
            raise ValueError(
                f"[FC STATE] _fc_move_entry new_name must be string, got {type(new_name_raw).__name__}"
            )
        new_name = new_name_raw.strip()
        source_pointer, source_value = resolve_logical_path(running, source_path)
        if source_value is None:
            raise FileNotFoundError(
                f"[FC STATE] _fc_move_entry source not found: '{source_path}'"
            )
        _dest_parent_pointer, dest_parent_node = resolve_logical_path(running, dest_parent_path)
        basename = source_path.rstrip("/").rsplit("/", 1)[-1]

        if new_name:
            if "/" in new_name:
                raise ValueError(
                    f"[FC STATE] _fc_move_entry new_name must be basename, got '{new_name}'"
                )
            if not isinstance(dest_parent_node, dict):
                raise FileNotFoundError(
                    f"[FC STATE] _fc_move_entry destination parent not found: '{dest_parent_path}'"
                )
            basename = new_name
        elif isinstance(dest_parent_node, dict):
            # Standard move: destination is an existing directory.
            pass
        else:
            # Compatibility path: model may pass full destination path
            # (including target basename) in dest_parent_path.
            dest_parts = [p for p in dest_parent_path.strip("/").split("/") if p]
            if len(dest_parts) <= 1:
                raise FileNotFoundError(
                    f"[FC STATE] _fc_move_entry destination parent not found: '{dest_parent_path}'"
                )
            candidate_parent = "/".join(dest_parts[:-1])
            candidate_name = dest_parts[-1]
            _candidate_parent_ptr, candidate_parent_node = resolve_logical_path(running, candidate_parent)
            if not isinstance(candidate_parent_node, dict):
                raise FileNotFoundError(
                    f"[FC STATE] _fc_move_entry destination parent not found: '{dest_parent_path}'"
                )
            dest_parent_path = candidate_parent
            basename = candidate_name

        dest_pointer = _child_pointer(running, dest_parent_path, basename)
        step_ops.append({"op": "remove", "path": source_pointer})
        step_ops.append({"op": "add", "path": dest_pointer, "value": _copy.deepcopy(source_value)})

    elif name == "_fc_replace_entry":
        entry_path = _resolve_entry_path(running, args.get("entry_path", ""))
        new_value_json = args.get("new_value_json", "{}")
        if isinstance(new_value_json, str):
            try:
                value = json.loads(new_value_json)
            except Exception as exc:
                raise ValueError(
                    f"[FC STATE] _fc_replace_entry invalid new_value_json: {new_value_json!r}"
                ) from exc
        else:
            value = new_value_json
        pointer, node = resolve_logical_path(running, entry_path)
        if node is None:
            # Upsert fallback: if replace target is missing, use add at same pointer.
            # apply_json_patch(add, create_missing=True) will materialize missing dict paths.
            step_ops.append({"op": "add", "path": pointer, "value": value})
        else:
            step_ops.append({"op": "replace", "path": pointer, "value": value})

    else:
        raise ValueError(f"[FC STATE] Unknown tool: {name}")


def _update_state_via_fc(
    previous_state: Dict[str, Any],
    tool_calls: List[Dict[str, Any]],
    task: Optional[str] = None,
    execution_results: Optional[List[Any]] = None,
    tool_descriptions: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    state_model: Optional[str] = None,
) -> Dict[str, Any]:
    """State update using structured function calling.

    The LLM receives semantic tools (add_entry, remove_entry, …) and returns
    tool-call requests.  A deterministic resolver then translates the logical
    paths into correct JSON Pointers, eliminating the path-construction burden
    from the LLM.
    """
    import copy as _copy

    if state_model is None:
        state_model = get_state_model()
    if state_model is None:
        raise RuntimeError("[FC STATE] No state model configured for function-calling mode")

    fc_tools = _build_fc_tools()
    state_agent = ChatAgent(
        _FC_SYSTEM_PROMPT,
        model=create_model(state_model, max_tokens=4096, temperature=0.001),
        external_tools=fc_tools,
        step_timeout=get_agent_timeout(),
    )

    # Build user query — identical structure to the freeform path.
    def _json_or_str(obj: Any) -> str:
        try:
            return json.dumps(obj, indent=2)
        except Exception:
            return str(obj)

    query_sections: List[str] = []
    required_state_effects = _extract_required_state_effects(tool_calls, tool_descriptions)
    query_sections.append(
        "Authoritative state policy: top-level toolkit state is canonical business truth; "
        "runtime_state is transient context only."
    )
    if task:
        query_sections.append(f"Task: {task}")
    if required_state_effects:
        query_sections.append(
            "Required state effects for successful calls:\n"
            f"{_json_or_str(required_state_effects)}"
        )
    query_sections.append(f"Previous state:\n{_json_or_str(previous_state)}")
    query_sections.append(f"Tool calls ({len(tool_calls)}, executed in order):\n{_json_or_str(tool_calls)}")
    if execution_results:
        query_sections.append(f"Execution results:\n{_json_or_str(execution_results)}")
    if tool_descriptions:
        query_sections.append(f"Tool descriptions:\n{_json_or_str(tool_descriptions)}")
    state_query = "\n\n".join(query_sections)

    state_response = None
    fc_requests: List[Any] = []
    try:
        _ct0 = datetime.now()
        logger.debug("[FC STATE] LLM START (model=%s)", state_model)
        state_response = state_agent.step(state_query)
        _ct1 = datetime.now()
        logger.debug("[FC STATE] LLM END (elapsed=%.3fs)", (_ct1 - _ct0).total_seconds())

        # Extract tool-call requests from response.
        info = getattr(state_response, "info", None) or {}
        fc_requests = info.get("external_tool_call_requests") or []

        if required_state_effects:
            req_names = _extract_fc_request_names(fc_requests)
            if req_names and all(name == "_fc_no_state_change" for name in req_names):
                correction = (
                    "CORRECTION: Required state effects were provided for successful calls. "
                    "You MUST realize those effects with concrete domain-state mutations "
                    "(_fc_add_entry/_fc_replace_entry/_fc_move_entry/_fc_remove_entry as appropriate). "
                    "Do NOT use _fc_no_state_change for these calls."
                )
                logger.warning(
                    "[FC STATE] Retrying because model returned only _fc_no_state_change "
                    "despite required successful state effects."
                )
                state_response = state_agent.step(f"{state_query}\n\n{correction}")
                info = getattr(state_response, "info", None) or {}
                fc_requests = info.get("external_tool_call_requests") or []
    except Exception as llm_exc:
        logger.warning("[FC STATE] LLM call failed: %s — degrading to auto-merge", llm_exc)
        fc_requests = []

    _fc_degraded = False
    if not fc_requests:
        logger.warning("[FC STATE] No function-call requests returned by state model — degrading to auto-merge")
        _fc_degraded = True
        state_result = _copy.deepcopy(previous_state)
    else:
        # Convert to JSON Patch ops (individual op errors are skipped inside).
        patch_ops = _fc_calls_to_patch(previous_state, fc_requests)

        try:
            logger.info("[FC STATE PATCH] ops: %s", json.dumps(patch_ops, ensure_ascii=False, indent=2, default=str))
        except Exception as _exc:
            logger.info("[FC STATE PATCH] ops: <unprintable> (%s)", _exc)

        if patch_ops:
            state_result = apply_json_patch(_copy.deepcopy(previous_state), patch_ops)
            logger.info("[FC STATE] Applied %d patch ops from %d tool calls", len(patch_ops), len(fc_requests))
        else:
            req_names = _extract_fc_request_names(fc_requests)
            if required_state_effects and req_names and all(n == "_fc_no_state_change" for n in req_names):
                logger.warning(
                    "[FC STATE] Required state effects present but model returned only "
                    "_fc_no_state_change — degrading to auto-merge"
                )
                _fc_degraded = True
            elif not req_names or not all(n == "_fc_no_state_change" for n in req_names):
                logger.warning(
                    "[FC STATE] Model returned function calls but produced no patch ops — degrading to auto-merge"
                )
                _fc_degraded = True
            state_result = _copy.deepcopy(previous_state)

    if required_state_effects:
        state_result = _apply_counter_effect_guards(
            previous_state=previous_state,
            state_result=state_result,
            required_state_effects=required_state_effects,
        )

    state_result = _auto_merge_response_fields(
        state_result=state_result,
        tool_calls=tool_calls,
        tool_descriptions=tool_descriptions,
    )

    # Record token usage.
    try:
        if session_id and state_response is not None:
            from ..utils.llm_usage import extract_token_usage
            usage = extract_token_usage(state_response)
            from ..handlers.session_handler import session_handler
            session_handler.record_llm_usage(
                session_id,
                category="state_update",
                model=str(state_model),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
    except Exception as usage_exc:
        logger.warning(f"Failed to record FC state update token usage: {usage_exc}")

    # Persist.
    if session_id:
        try:
            from ..handlers.session_handler import session_handler
            session_handler.add_to_state(session_id, state_result)
        except Exception as e:
            logger.warning(f"Failed to auto-update session {session_id}: {e}")

    return state_result


_BOOTSTRAP_SYSTEM_PROMPT = """You are a system state initializer. Infer initial runtime state from tool definitions and the initial data config.

Goal:
- Initialize ONLY runtime-related state (for example current working directory, selected resource, authenticated identity).
- Do NOT rewrite or copy the full config.
- Do NOT mutate domain data under toolkit roots unless absolutely required for runtime bootstrap.
- DO NOT repeat data that already exists in initial data config.

Reasoning steps:
1. Read toolkit operation descriptions and identify context variables that must exist for correct execution.
2. Infer initial values from the data config structure (use unambiguous values; prefer absolute paths).
3. Emit JSON Patch operations that add/initialize runtime state.

Output format (JSON only):
{
  "patch": [ ...RFC6902 operations... ],
  "reasoning": "brief explanation"
}

Patch requirements:
- patch can be [] if no runtime bootstrap is needed.
- Allowed ops: add, remove, replace, move, copy, test.
- Each op must include "op" and "path"; add/replace/test need "value"; move/copy need "from".
- runtime state must be top-level and toolkit-scoped:
  /runtime_state
  /runtime_state/toolkits/<ToolkitName>/<state_key>
- Preferred parent creation order when missing:
  1) add /runtime_state {"toolkits": {}}
  2) add /runtime_state/toolkits/<ToolkitName> {}
  3) add/replace leaf keys

Example:
{
  "patch": [
    {"op":"add","path":"/runtime_state","value":{"toolkits":{}}},
    {"op":"add","path":"/runtime_state/toolkits/GorillaFileSystem","value":{"current_working_directory":"/workspace"}}
  ],
  "reasoning":"Initialized cwd from the top-level directory."
}

Output only the JSON object.
"""


def extract_toolkit_summaries(schemas: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract compact toolkit summaries from OpenAPI schema dicts.

    Args:
        schemas: Mapping of toolkit_name -> full OpenAPI schema dict

    Returns:
        List of {
          name,
          description,
          operations: [{id, summary}],
          runtime_defaults: {...}  # from info.x-default-state.global.runtime_defaults
        }
    """
    summaries = []
    for name, schema in schemas.items():
        if not isinstance(schema, dict):
            continue
        info = schema.get("info", {})
        runtime_defaults: Dict[str, Any] = {}
        x_default_state = info.get("x-default-state")
        if isinstance(x_default_state, dict):
            global_block = x_default_state.get("global")
            if isinstance(global_block, dict):
                extracted_runtime_defaults = global_block.get("runtime_defaults")
                if isinstance(extracted_runtime_defaults, dict):
                    runtime_defaults = extracted_runtime_defaults
        ops = []
        for path, methods in (schema.get("paths") or {}).items():
            for method, op_def in methods.items():
                if not isinstance(op_def, dict) or "operationId" not in op_def:
                    continue
                ops.append({
                    "id": op_def["operationId"],
                    "summary": op_def.get("summary", ""),
                })
        summaries.append({
            "name": name,  # Use the config key, not info.title, for consistency
            "description": info.get("description", ""),
            "operations": ops,
            "runtime_defaults": runtime_defaults,
        })
    return summaries


def _canonical_runtime_state_key(field_name: str) -> str:
    """Normalize runtime-state key names while preserving generic behavior."""
    normalized = str(field_name or "").strip().lstrip("_")
    if not normalized:
        return normalized
    lower_name = normalized.lower()
    if lower_name in {"current_working_directory", "current_directory", "current_dir", "cwd"}:
        return "current_working_directory"
    if "current" in lower_name and (
        "dir" in lower_name or "directory" in lower_name or "path" in lower_name
    ):
        return "current_working_directory"
    return normalized


def _infer_root_path(initial_state: Dict[str, Any], toolkit_name: str) -> Optional[str]:
    """Infer toolkit root path (e.g. '/alex') from initial state structure."""
    toolkit_state = initial_state.get(toolkit_name)
    if not isinstance(toolkit_state, dict):
        return None

    root_container = toolkit_state.get("root")
    if isinstance(root_container, dict) and root_container:
        first_key = next(iter(root_container.keys()))
        if isinstance(first_key, str) and first_key.strip():
            return "/" + first_key.strip("/")

    dict_keys = [k for k, v in toolkit_state.items() if isinstance(v, dict)]
    if len(dict_keys) == 1:
        only_key = dict_keys[0]
        if isinstance(only_key, str) and only_key.strip():
            return "/" + only_key.strip("/")

    return None


def _extract_init_rules(runtime_defaults: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract normalized init rules from runtime_defaults."""
    rules: List[Dict[str, str]] = []
    seen: set[str] = set()
    if not isinstance(runtime_defaults, dict):
        return rules

    explicit_rules = runtime_defaults.get("init_rules")
    if isinstance(explicit_rules, list):
        for rule in explicit_rules:
            if not isinstance(rule, dict):
                continue
            field = rule.get("field")
            init_from = rule.get("init_from")
            if not isinstance(field, str) or not field.strip():
                continue
            if not isinstance(init_from, str) or not init_from.strip():
                continue
            normalized = {
                "field": _canonical_runtime_state_key(field),
                "init_from": init_from.strip(),
            }
            marker = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
            if marker not in seen:
                seen.add(marker)
                rules.append(normalized)

    for field, value in runtime_defaults.items():
        if field == "init_rules":
            continue
        if not isinstance(value, dict):
            continue
        init_from = value.get("init_from")
        if not isinstance(init_from, str) or not init_from.strip():
            continue
        normalized = {
            "field": _canonical_runtime_state_key(field),
            "init_from": init_from.strip(),
        }
        marker = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        if marker not in seen:
            seen.add(marker)
            rules.append(normalized)

    return rules


def _resolve_init_value(
    initial_state: Dict[str, Any],
    toolkit_name: str,
    init_from: str,
) -> Optional[Any]:
    """Resolve init value from a structured init rule."""
    source = str(init_from or "").strip()
    if not source:
        return None
    if source.startswith("/"):
        return source

    source_lower = source.lower()
    if source_lower == "root":
        root_path = _infer_root_path(initial_state, toolkit_name)
        return root_path if root_path is not None else "/"

    toolkit_state = initial_state.get(toolkit_name)
    if isinstance(toolkit_state, dict) and source in toolkit_state:
        value = toolkit_state[source]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

    return None


def _build_structured_bootstrap_patch(
    initial_state: Dict[str, Any],
    toolkit_summaries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build RFC6902 patch ops from structured x-default-state runtime rules."""
    patch_ops: List[Dict[str, Any]] = []
    tool_values: Dict[str, Dict[str, Any]] = {}

    for summary in toolkit_summaries:
        if not isinstance(summary, dict):
            continue
        toolkit_name = summary.get("name")
        if not isinstance(toolkit_name, str) or not toolkit_name:
            continue
        runtime_defaults = summary.get("runtime_defaults")
        rules = _extract_init_rules(runtime_defaults if isinstance(runtime_defaults, dict) else {})
        if not rules:
            continue

        for rule in rules:
            target_field = rule.get("field", "")
            init_from = rule.get("init_from", "")
            if not target_field or not init_from:
                continue
            value = _resolve_init_value(initial_state, toolkit_name, init_from)
            if value is None:
                continue
            tool_values.setdefault(toolkit_name, {})[target_field] = value

    if not tool_values:
        return patch_ops

    patch_ops.append({"op": "add", "path": "/runtime_state", "value": {"toolkits": {}}})
    for toolkit_name, values in tool_values.items():
        patch_ops.append({"op": "add", "path": f"/runtime_state/toolkits/{toolkit_name}", "value": {}})
        for key, value in values.items():
            patch_ops.append(
                {"op": "add", "path": f"/runtime_state/toolkits/{toolkit_name}/{key}", "value": value}
            )

    return patch_ops


def bootstrap_state(
    initial_state: Dict[str, Any],
    toolkit_summaries: List[Dict[str, Any]],
    state_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Bootstrap initial config by inferring runtime state from tool definitions.

    Analyzes toolkit descriptions and operations to determine what state variables
    the system needs to track (e.g. current working directory, authenticated user),
    then initializes them based on the data config structure.

    Args:
        initial_state: Raw initial config (may be empty {})
        toolkit_summaries: List of {name, description, operations: [{id, summary}]}
            as returned by extract_toolkit_summaries()
        state_model: LLM model to use (default: uses global config)

    Returns:
        Enriched config with runtime_state initialized via JSON Patch
    """
    structured_patch_ops = _build_structured_bootstrap_patch(initial_state, toolkit_summaries)
    if structured_patch_ops:
        try:
            enriched_config = apply_json_patch(initial_state, structured_patch_ops)
            logger.info(
                "[BOOTSTRAP] Applied %d structured patch ops from x-default-state runtime defaults",
                len(structured_patch_ops),
            )
            return enriched_config
        except Exception as exc:
            logger.warning(f"[BOOTSTRAP] Structured bootstrap patch failed: {exc}; falling back to LLM")

    if state_model is None:
        state_model = get_state_model()
    if state_model is None:
        logger.info("[BOOTSTRAP] No state model available, returning state as-is")
        return initial_state

    # Build compact user query
    query_parts: List[str] = []
    try:
        query_parts.append(f"Tool definitions:\n{json.dumps(toolkit_summaries, indent=2)}")
    except Exception:
        query_parts.append(f"Tool definitions:\n{str(toolkit_summaries)}")

    try:
        query_parts.append(f"Initial data config:\n{json.dumps(initial_state, indent=2)}")
    except Exception:
        query_parts.append(f"Initial data config:\n{str(initial_state)}")

    query = "\n\n".join(query_parts)

    bootstrap_agent = ChatAgent(
        _BOOTSTRAP_SYSTEM_PROMPT,
        model=create_model(state_model, max_tokens=16384, temperature=0.001),
        step_timeout=get_agent_timeout(),
    )

    _t0 = datetime.now()
    logger.info(f"[BOOTSTRAP] LLM START (model={state_model})")
    response = bootstrap_agent.step(query)
    _t1 = datetime.now()
    elapsed = (_t1 - _t0).total_seconds()
    logger.info(f"[BOOTSTRAP] LLM END (elapsed={elapsed:.3f}s)")

    response_str = response.msg.content if getattr(response, "msg", None) else "{}"
    response_str = sanitize_llm_json_text(response_str)

    payload = json_repair.loads(response_str)

    patch_ops: Optional[List[Dict[str, Any]]] = None
    if isinstance(payload, dict):
        payload_patch = payload.get("patch")
        if isinstance(payload_patch, list):
            patch_ops = payload_patch
        reasoning = payload.get("reasoning", "")
        if reasoning:
            logger.info(f"[BOOTSTRAP] Reasoning: {reasoning}")
    elif isinstance(payload, list):
        # Backward compatibility for direct patch-array outputs
        patch_ops = payload

    if patch_ops is None:
        logger.warning("[BOOTSTRAP] Missing valid 'patch' array, returning state as-is")
        return initial_state

    try:
        enriched_config = apply_json_patch(initial_state, patch_ops)
        logger.info(f"[BOOTSTRAP] Applied {len(patch_ops)} bootstrap patch ops")
        return enriched_config
    except Exception as exc:
        logger.warning(f"[BOOTSTRAP] Failed to apply bootstrap patch: {exc}")
        return initial_state


def update_state(
    previous_state: Dict[str, Any],
    tool_calls: List[Dict[str, Any]],
    task: Optional[str] = None,
    execution_results: Optional[List[Any]] = None,
    tool_descriptions: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    state_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Update config based on tool call results using function-calling state update.

    Updates the configuration to reflect persistent state changes caused by tool calls.
    Only write operations should update the config - read operations do not modify state.

    Args:
        previous_state: The configuration before the tool calls
        tool_calls: List of tool calls with name, arguments, and results
        task: Optional task description for context
        execution_results: Optional execution results (if not in tool_calls)
        tool_descriptions: Optional tool descriptions for context
        session_id: Optional session ID for automatic config persistence
        state_model: LLM model to use for state updates (default: None, uses global config)

    Returns:
        Updated configuration dictionary
    """
    if state_model is None:
        state_model = get_state_model()
    if state_model is None:
        logger.info("[STATE UPDATE] state_model is disabled; skipping state update")
        return previous_state.copy() if isinstance(previous_state, dict) else previous_state

    prepared_calls, skipped_error_calls = _prepare_tool_calls_for_state_update(
        tool_calls=tool_calls,
        tool_descriptions=tool_descriptions,
    )
    if skipped_error_calls > 0:
        logger.info(
            "[STATE UPDATE] Skipped %d error tool calls with no explicit error-state effects",
            skipped_error_calls,
        )

    if not prepared_calls:
        logger.info("[STATE UPDATE] No tool calls eligible for state update; state unchanged")
        return previous_state.copy() if isinstance(previous_state, dict) else previous_state

    return _update_state_via_fc(
        previous_state=previous_state,
        tool_calls=prepared_calls,
        task=task,
        execution_results=execution_results,
        tool_descriptions=tool_descriptions,
        session_id=session_id,
        state_model=state_model,
    )


def update_state_from_real_tool(
    previous_state: Dict[str, Any],
    tool_call: Any,
    session_id: Optional[str] = None,
    state_model: Optional[str] = None,
    task: Optional[str] = None,
    involved_classes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Update config based on Real Tool execution result(s).

    This is the unified entry point for all real tool state updates.
    Handles input normalization, format conversion, and delegates to update_state().

    Accepts various input formats:
    - Single tool_call dict: {'name': ..., 'arguments': ..., 'result': ...}
    - List of tool_call dicts
    - Alternative key names: 'function' instead of 'name', 'args' instead of 'arguments'

    Args:
        previous_state: The configuration before the tool call
        tool_call: Tool call dict or list of dicts with name/function, arguments/args, and result
        session_id: Optional session ID for automatic config persistence
        state_model: LLM model to use for state updates (default: None, uses global config)
        task: Optional task description for context (default: generic Hybrid Mode description)
        involved_classes: Optional list of involved API classes (for logging)

    Returns:
        Updated configuration dictionary

    Example:
        >>> update_state_from_real_tool(
        ...     previous_state={'users': []},
        ...     tool_call={'name': 'get_user', 'arguments': {'id': 1}, 'result': {'id': 1, 'name': 'Alice'}}
        ... )
    """
    # Normalize input to list format
    if isinstance(tool_call, list):
        raw_tool_calls = tool_call
    else:
        raw_tool_calls = [tool_call] if tool_call else []

    if not raw_tool_calls:
        logger.warning("No tool calls provided, returning previous state unchanged")
        return previous_state.copy() if isinstance(previous_state, dict) else previous_state

    def _normalize_real_arguments(raw_args: Any) -> Dict[str, Any]:
        """Normalize real-tool arguments to a flat mapping."""
        if not isinstance(raw_args, dict):
            return {}
        nested = raw_args.get("kwargs")
        if isinstance(nested, dict):
            return dict(nested)
        return dict(raw_args)

    # Normalize tool call format (handle 'function' vs 'name', 'args' vs 'arguments')
    formatted_calls = []
    for idx, tc in enumerate(raw_tool_calls):
        try:
            # Extract function name (support multiple key names)
            function_name = tc.get('name') or tc.get('function') or tc.get('function_name') or ''

            # Extract arguments (support multiple key names)
            raw_arguments = tc.get('arguments') or tc.get('args') or {}
            arguments = _normalize_real_arguments(raw_arguments)

            # Recover function name from envelope payload when needed.
            if not function_name and isinstance(raw_arguments, dict):
                nested_name = raw_arguments.get("_tool_name")
                if isinstance(nested_name, str) and nested_name.strip():
                    function_name = nested_name.strip()

            # Extract result
            result = tc.get('result')
            status, reason = classify_tool_call_status(result)

            if not function_name:
                logger.warning(f"Tool call {idx} missing function name, skipping")
                continue

            formatted_calls.append({
                'name': function_name,
                'arguments': arguments,
                'result': result,
                'execution_status': status,
                'error_reason': reason,
            })
        except Exception as e:
            logger.error(f"Error formatting tool call {idx}: {e}")
            continue

    if not formatted_calls:
        logger.warning("No valid tool calls after formatting, returning previous state unchanged")
        return previous_state.copy() if isinstance(previous_state, dict) else previous_state

    # Log batch processing info
    if involved_classes:
        logger.debug(f"Involved classes: {involved_classes}")
    logger.info(f"[STATE UPDATE] Processing {len(formatted_calls)} tool calls in batch mode")

    tool_descriptions: Optional[Dict[str, Any]] = None
    try:
        from ..schemas.global_loader import get_global_schema_loader
        schema_loader = get_global_schema_loader()
        if schema_loader is not None:
            _td: Dict[str, Any] = {}
            for fc in formatted_calls:
                fname = fc.get("name", "")
                # Split "ToolkitName_operationId" on first underscore
                parts = fname.split("_", 1)
                if len(parts) == 2:
                    toolkit_prefix, op_id = parts
                else:
                    toolkit_prefix, op_id = "", fname
                schema_file = schema_loader.find_schema_file(toolkit_prefix) if toolkit_prefix else None
                if not schema_file and fname:
                    schema_file = schema_loader.find_schema_file(fname)
                if schema_file:
                    schema = schema_loader.load_schema(schema_file)
                    x_ds = schema.get("info", {}).get("x-default-state", {})
                    tools_block = x_ds.get("tools", {})
                    tool_entry = tools_block.get(op_id, {})
                    if isinstance(tool_entry, dict) and tool_entry:
                        hints: Dict[str, Any] = {}
                        for key in ("state_effects", "state_effects_on_success",
                                    "state_effects_on_error", "state_effects_always"):
                            val = tool_entry.get(key)
                            if isinstance(val, list) and val:
                                hints[key] = val
                        if hints:
                            # Also include toolkit info for context
                            info = schema.get("info", {})
                            toolkit_info: Dict[str, Any] = {}
                            title = info.get("title")
                            if isinstance(title, str) and title:
                                toolkit_info["name"] = title
                            td_entry: Dict[str, Any] = {"state_hints": hints}
                            if toolkit_info:
                                td_entry["toolkit"] = toolkit_info
                            _td[fname] = td_entry
            if _td:
                tool_descriptions = _td
                logger.info(
                    "[STATE UPDATE] Built tool_descriptions for %d/%d calls from schemas",
                    len(_td), len(formatted_calls),
                )
    except Exception as e:
        logger.debug("[STATE UPDATE] Could not build tool_descriptions from schemas: %s", e)

    # Default task description
    if task is None:
        task = "Register entities discovered by Real Tool execution so Mock Tools can reference them (Hybrid Mode)"

    # Use unified update_state()
    try:
        result = update_state(
            previous_state=previous_state,
            tool_calls=formatted_calls,
            task=task,
            session_id=session_id,
            state_model=state_model,
            tool_descriptions=tool_descriptions,
        )
        logger.info(f"[STATE UPDATE] Batch update completed successfully for {len(formatted_calls)} tool calls")
        return result
    except Exception as e:
        logger.error(f"[STATE UPDATE] Batch update failed with {len(formatted_calls)} tool calls: {e}", exc_info=True)
        raise


# Backward compatibility alias
def calibrate_state_with_results(
    initial_state: Dict[str, Any],
    tool_calls_with_results: List[Dict[str, Any]],
    involved_classes: Optional[List[str]] = None,
    task: str = "",
    state_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Calibrate config based on real execution results.

    DEPRECATED: Use update_state_from_real_tool() directly.
    This function is kept for backward compatibility.

    Args:
        initial_state: Initial configuration state
        tool_calls_with_results: List of tool calls with execution results
        involved_classes: Optional list of involved API classes (for logging)
        task: Task description (optional)
        state_model: LLM model to use (default: None, uses global config)

    Returns:
        Updated configuration
    """
    logger.debug("calibrate_state_with_results() called - delegating to update_state_from_real_tool()")

    return update_state_from_real_tool(
        previous_state=initial_state,
        tool_call=tool_calls_with_results,
        state_model=state_model,
        task=task or "Calibrate config from real execution results",
        involved_classes=involved_classes
    )

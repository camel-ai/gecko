import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from camel.agents import ChatAgent
from inference.utils.log_context import bind_log_context
from utils.bfcl_multi_turn_tool_names import normalize_bfcl_multi_turn_tool_call
from utils.model_utils import create_model, sanitize_llm_json_text
from .contract_matching import (
    read_evidence_binding_for,
    tool_call_arguments_compatible,
)
from .global_config import get_state_model
from .state_update_hints import annotate_prompt_tool_calls_with_hints
from .state_update_hints import resolve_direct_logical_path
from .state_update_enriched import (
    enrich_annotated_calls,
    rewrite_persistent_state_paths_to_existing,
    state_context_label,
)
import json_repair
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

STATE_AGENT_TIMEOUT_SECONDS = 600.0


class DirectPatchFormatError(ValueError):
    """Raised when the state model repeatedly violates the direct patch schema."""


class _DirectPatchValueOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["set", "add", "replace"] = Field(
        description="State mutation operation that writes a value.",
    )
    path: str = Field(description="Logical state path separated by '/'.")
    value: Any = Field(description="JSON value to write at path.")


class _DirectPatchRemoveOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["remove"] = Field(description="Remove a state entry.")
    path: str = Field(description="Logical state path separated by '/'.")


class _DirectPatchTransferOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    op: Literal["move", "copy"] = Field(description="Move or copy an existing state entry.")
    from_path: str = Field(alias="from", description="Logical source path separated by '/'.")
    path: str = Field(description="Logical destination path separated by '/'.")


class _DirectPatchNoopOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["noop", "no_change"] = Field(description="Explicit no-op.")


DirectPatchOperation = Union[
    _DirectPatchValueOperation,
    _DirectPatchRemoveOperation,
    _DirectPatchTransferOperation,
    _DirectPatchNoopOperation,
]


class DirectPatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patch: List[DirectPatchOperation] = Field(
        description="Direct logical patch operations. Every item must be an object.",
    )
    reasoning: str = Field(description="One short sentence explaining the patch.")


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
    for index, part in enumerate(parent_parts):
        if isinstance(current, list):
            idx = _parse_list_index(part, len(current), allow_end=False)
            current = current[idx]
        elif isinstance(current, dict):
            if part not in current:
                if create_missing:
                    next_part = parent_parts[index + 1] if index + 1 < len(parent_parts) else last
                    current[part] = [] if next_part == "-" else {}
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



_FC_SYSTEM_PROMPT = """You are a state tracker. Given the previous state and a sequence of tool calls with their results, determine what state mutations occurred and call the provided tools to apply them.

Rules:
1. Read/query operations → no_state_change(). Write-like operations (post/create/update/delete/add/remove/send/set/move/...) that succeed MUST produce state mutations. no_state_change() is valid ONLY for truly read-only operations or explicit error responses.
2. Use tool results as source of truth. Process calls in listed order.
3. If a "Required state effects" section is provided, you MUST realize every listed effect. no_state_change() is invalid for those calls.
4. State location: top-level /<ToolkitName>/... is canonical for domain data (records, counters, collections, balances, inventories). runtime_state is ONLY for transient context (current_working_directory). Never put domain data in runtime_state.
5. When a successful write operation creates or returns an entity (with fields like id, content, tags, etc.), persist it in the appropriate canonical collection using add_entry, and update related counters/balances consistently.
6. When a successful write response returns fields matching existing canonical state keys, update those state values only if the fields are durable canonical state, not fields explicitly marked response_only_fields or transient read-context payload.
7. On auth/login success, persist authentication flags in canonical toolkit state (not runtime-only).
8. For filesystem toolkits: context-changing ops (cd/navigation) → set_runtime_field() with absolute paths; move/rename → move_entry() (atomic, never duplicate); scope from earlier calls affects subsequent ones.
9. Treat any task text as non-authoritative background. The previous state, executed tool calls, and tool results are the authority for state mutations.
10. Fields listed under response_only_fields are response payload only. Do not copy them into persistent toolkit state unless Required state effects or persistent_state_shapes explicitly say they are stored.
11. When a successful write contract requires a computed value that is not echoed in the write response, infer it from earlier read-context evidence or called-method/static contract data rather than reusing an unrelated existing state value.
12. Read-only calls embedded as matched_context_only_read_calls are evidence only. They matched the current write by method name and compatible shared arguments; use them only to fill values explicitly required by that write contract. Never copy their returned records wholesale and never mutate the read call's toolkit state just because it appeared in a read result.
13. Do not create new top-level toolkit fields from response payload names. Durable domain data must be written under existing public state containers or contract-declared persistent paths.
14. If a durable field is generated/random and the exact value is not returned, synthesize a schema-valid realistic value. Do not use placeholders such as 0, null, empty string, or "unknown" when they would violate later validation.
15. If a tool call entry includes resolved_effect_hints, treat those exact {op, path, value} mutations as authoritative concrete effects derived from the contract, previous state, and observed results. Prefer applying those path/value mutations over re-deriving shape or arithmetic from natural-language state_effects.
16. If a tool call entry includes dynamic_key_shape_hints, treat that hint as the canonical inserted/removed top-level collection item shape. Append/remove whole items with the dynamic-key object shape instead of merging into an existing same-key nested list/map, unless the value_template itself is explicitly a list or object.
17. If a tool call entry includes set_membership_hints, persist the scalar value as a collection member (list append / set add). Do not convert set-membership persistence into a boolean-valued mapping entry.
18. If a tool call entry includes latest_match_hints, and the contract text says "latest" or "most recent", select the last matching collection item when removing or updating — not the first.
19. If a tool call entry includes matched_read_scalar_candidates or helper_static_scalar_candidates, prefer those exact scalar values as the source for the named persisted field when the write contract requires a value not echoed by the write response.
20. If a tool call entry includes resolved_persistent_records, use those exact current-state record values as the canonical source for derived mutations, removals, or balance/counter adjustments tied to the same lookup key.

Path format for entry tools (add/remove/move/replace):
- Use absolute paths from config root: '<ToolkitName>/<key>/...' (e.g. '<ToolkitName>/records', '<ToolkitName>/items/12345', '<ToolkitName>/root/folder/document').
- Paths can also be relative to the current working scope when a filesystem toolkit is active.
- For filesystem toolkits, omit structural 'contents' wrappers — the resolver handles them.
"""


_DIRECT_PATCH_SYSTEM_PROMPT = """You are a direct state patcher.

Given the previous task state and executed tool calls with results, output a minimal logical patch that updates the task state.

Authority rules:
1. The previous state, executed tool calls, and tool results are authoritative.
2. Do not use outside assumptions or background task wording.
3. Read-only calls and failed calls should not mutate state.
4. Read-only/context calls may explain later write effects, but their returned fields are transient context unless a later write call explicitly persists them.
5. Never persist fields listed in response_only_fields. Use them only to understand branch outcome or response payload shape.
6. Persist only fields and records explicitly supported by write-call state_effects, persistent_state_shapes, or direct runtime context changes.
7. If a successful write-call contract requires a computed or derived persisted value that is absent from the write result itself, derive it from earlier context-only read results or called_method_static_data rather than substituting an unrelated state value.
8. Do not write placeholder values like 0, null, empty string, or "unknown" for required computed persisted fields when context-only read observations or called_method_static_data already provide a concrete value.
9. If a write-call entry includes matched_context_only_read_calls, treat those matched read results as the preferred evidence for derived values required by that write contract.
10. If enriched context includes resolved_persistent_records for the current write, use those exact current-state record values as the canonical source for derived fields, removals, or balance/counter adjustments tied to that record.
11. If helper_static_scalar_candidates are present for the current write, use those exact derived scalar values instead of recomputing from partial helper tables or defaulting to 0.
12. If a write-call entry includes dynamic_key_shape_hints or persistent_state_shapes, the shape declared there is the canonical persisted shape. The shape hint OVERRIDES whatever shape the tool response payload happens to use. Concretely:
    - If the hint says the container at path P is a list (e.g. `key_rule: "<key> maps to a list; initialize P[<key>] as [] if absent, then append"`), append a single object to that list. Do NOT write a dict-keyed-by-index like {"0": {...}}.
    - If the hint declares `fields: {a: ..., b: ..., c: ...}`, the appended/inserted item MUST contain ALL of those fields, including session-scoped values such as the authenticated `username`. Do not drop a field just because the tool response didn't echo it.
    - If a tool response payload uses a different container shape than the hint (e.g. response shows `{"0": {...}}` but the hint says list), the hint wins.
13. If set_membership_hints are present, append the candidate scalar value itself into the target collection. Do not encode set membership as a boolean-valued keyed map unless the contract explicitly says that.
14. If latest_match_hints are present, remove or update the last matching collection item in current state, not the first one.
15. Process tool calls in listed order.
16. If a write call includes resolved_effect_hints, treat them as authoritative concrete mutations derived from the contract, previous state, and observed results. Prefer copying those exact path/value mutations over re-deriving shape or arithmetic.
17. When a resolved_effect_hint already gives an exact scalar/list/object payload shape, do not wrap, regroup, or reinterpret it based on historical container examples.
18. If a durable field is generated/random and the exact value is not returned, synthesize a schema-valid realistic value. Do not use placeholders such as 0, null, empty string, or "unknown" when they would violate later validation.
19. When a state_effect or persistent_state_shape names a field with a bare path (no `runtime_state/...` prefix), use previous_state to decide WHERE that field already lives. If the field exists in `runtime_state.toolkits.<toolkit>.<field>` and NOT at `<toolkit>/<field>` in previous_state, write the new value only at the `runtime_state/...` path. Do not create a duplicate same-named field at the toolkit's top level. Persistent_state_shapes path strings are field-name hints, not commitments to a specific parent path.

Patch protocol:
Output JSON only:
{
  "patch": [
    {"op": "set", "path": "runtime_state/toolkits/GorillaFileSystem/current_working_directory", "value": "/workspace/docs"},
    {"op": "add", "path": "GorillaFileSystem/root/workspace/docs/notes.txt", "value": {"type": "file", "content": ""}},
    {"op": "remove", "path": "GorillaFileSystem/root/workspace/docs/old.txt"},
    {"op": "move", "from": "GorillaFileSystem/root/workspace/temp", "path": "GorillaFileSystem/root/workspace/archive"},
    {"op": "copy", "from": "GorillaFileSystem/root/workspace/a.txt", "path": "GorillaFileSystem/root/workspace/b.txt"}
  ],
  "reasoning": "one short sentence"
}

Patch array contract:
1. "patch" MUST be a JSON array.
2. Every item in "patch" MUST be a JSON object. Never output strings, nulls, booleans, numbers, or bare paths inside the patch array.
3. Each operation object MUST include an "op" string.
4. Allowed ops: set, add, replace, remove, move, copy, noop, no_change.
5. set/add/replace MUST include "path" and "value".
6. remove MUST include "path".
7. move/copy MUST include "from" and "path".

Path rules:
1. Use logical state paths, not JSON Pointers.
2. Path segments are ALWAYS separated by '/' only. Never use '.' as a separator. Keys or filenames that contain dots (e.g. 'notes.md', 'report_word_count.txt') are single segments — do not split them on the dot.
3. For filesystem entries, omit structural "contents" wrappers. The applier resolves them.
4. Relative filesystem paths are allowed and resolve against current_working_directory at that point in patch order.
5. For move/copy, "path" is the final destination entry path, not just the destination parent.
6. Runtime current_working_directory values must be absolute paths like "/workspace/docs".
7. Directory values use {"type": "directory", "contents": {}}.
8. File values use {"type": "file", "content": "..."}.
9. If nothing changed, return {"patch": [], "reasoning": "..."}.
"""



def _fc_set_runtime_field(toolkit: str, key: str, value: str) -> str:
    """Set a runtime state field for a toolkit.
    Use ONLY for transient context changes (e.g. current_working_directory, current_selection).
    Never use this for domain/business entities or counters (e.g. records, items, balances, inventories).
    value must be the definitive form (e.g. absolute path, not relative).
    """
    return "ok"


def _fc_add_entry(parent_path: str, name: str, value_json: str) -> str:
    """Add a new child entry under parent_path.
    parent_path: path with '/' separators (e.g. '<ToolkitName>/records', '<ToolkitName>/root/folder').
    name: key name for the new entry (e.g. 'item-1', 'document.txt').
    value_json: JSON string of the value to insert — any valid JSON (object, array, string, number, etc.).
    """
    return "ok"


def _fc_remove_entry(entry_path: str) -> str:
    """Remove the entry at entry_path.
    entry_path: logical path with '/' separators (e.g. '<ToolkitName>/records/0', '<ToolkitName>/root/folder/file.txt').
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
    entry_path: logical path with '/' separators (e.g. '<ToolkitName>/status/current', '<ToolkitName>/account/value').
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

        for nested_key in ("result", "message"):
            inner = result.get(nested_key)
            if isinstance(inner, str):
                stripped = inner.strip()
                lowered = stripped.lower()
                if lowered.startswith("error") or lowered.startswith("failed"):
                    return "error", stripped

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


def _get_tool_state_access(
    tool_name: str,
    tool_descriptions: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Return explicit schema state access classification for a tool."""
    if not isinstance(tool_descriptions, dict):
        return None
    desc_entry = tool_descriptions.get(tool_name)
    if not isinstance(desc_entry, dict):
        return None
    state_hints = desc_entry.get("state_hints")
    if not isinstance(state_hints, dict):
        return None

    raw_access = state_hints.get("state_access")
    if not isinstance(raw_access, str):
        return None
    access = raw_access.strip().lower()
    if access in {"read", "write"}:
        return access
    return None


def _get_tool_state_hints(
    tool_name: str,
    tool_descriptions: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return schema state hints for a tool when present."""
    if not isinstance(tool_descriptions, dict):
        return {}
    desc_entry = tool_descriptions.get(tool_name)
    if not isinstance(desc_entry, dict):
        return {}
    state_hints = desc_entry.get("state_hints")
    return state_hints if isinstance(state_hints, dict) else {}


def _has_declared_state_effects(state_hints: Dict[str, Any]) -> bool:
    """Whether schema hints declare any persistent state effect branch."""
    for key in (
        "state_effects",
        "state_effects_on_success",
        "state_effects_on_error",
        "state_effects_always",
    ):
        value = state_hints.get(key)
        if isinstance(value, list) and value:
            return True
    return False


def _is_read_only_tool_call(
    tool_name: str,
    tool_descriptions: Optional[Dict[str, Any]],
) -> bool:
    """Whether schema explicitly marks this operation as persistent-state read-only."""
    return _get_tool_state_access(tool_name, tool_descriptions) == "read"


def _is_observable_read_call(
    call: Dict[str, Any],
    tool_descriptions: Optional[Dict[str, Any]],
) -> bool:
    """Whether a real-tool call can be recorded as observed state without LLM mutation."""
    call_name = str(call.get("name") or call.get("function") or "").strip()
    if not call_name:
        return False
    status, _ = classify_tool_call_status(call.get("result"))
    if status == "error":
        return False
    state_hints = _get_tool_state_hints(call_name, tool_descriptions)
    return (
        str(state_hints.get("state_access") or "").strip().lower() == "read"
        and not _has_declared_state_effects(state_hints)
    )


def _real_observation_json_size(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return len(str(value))


def _compact_real_observation(value: Any, *, max_chars: int = 5000, depth: int = 0) -> Any:
    """Compact observed real-tool results enough for mock planning context."""
    import copy as _copy

    if max_chars <= 0 or _real_observation_json_size(value) <= max_chars:
        return _copy.deepcopy(value)
    if depth > 4:
        text = str(value)
        return text[:800] + f"... [truncated {max(0, len(text) - 800)} chars]"
    if isinstance(value, str):
        return value[:1000] + f"... [truncated {max(0, len(value) - 1000)} chars]"
    if isinstance(value, list):
        items = [
            _compact_real_observation(item, max_chars=max(500, max_chars // 5), depth=depth + 1)
            for item in value[:5]
        ]
        result: Dict[str, Any] = {"type": "list_projection", "count": len(value), "items": items}
        if len(value) > 5:
            result["omitted_count"] = len(value) - 5
        return result
    if isinstance(value, dict):
        projected: Dict[str, Any] = {}
        deferred: Dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            key_norm = key_str.lower()
            important = key_norm.endswith("_id") or key_norm in {
                "id",
                "user_id",
                "reservation_id",
                "order_id",
                "flight_number",
                "date",
                "origin",
                "destination",
                "status",
                "cabin",
                "price",
                "prices",
                "payment_id",
                "payment_methods",
                "payment_history",
                "flights",
                "reservations",
                "available_seats",
                "created_at",
                "insurance",
            }
            if important or isinstance(item, (dict, list)):
                projected[key_str] = _compact_real_observation(
                    item,
                    max_chars=max(500, max_chars // 3),
                    depth=depth + 1,
                )
            else:
                deferred[key_str] = item
        remaining = max_chars - _real_observation_json_size(projected)
        if remaining > 200:
            for key, item in deferred.items():
                candidate = _copy.deepcopy(projected)
                candidate[key] = _compact_real_observation(item, max_chars=min(remaining, 500), depth=depth + 1)
                if _real_observation_json_size(candidate) > max_chars:
                    break
                projected = candidate
        omitted = max(0, len(value) - len(projected))
        if omitted:
            projected["_projection_note"] = f"omitted {omitted} low-priority fields"
        return projected
    return _copy.deepcopy(value)


def _observation_key(name: str, arguments: Dict[str, Any]) -> str:
    try:
        rendered_args = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    except Exception:
        rendered_args = str(arguments)
    return f"{name}:{rendered_args}"


def _upsert_real_tool_observations(
    state: Dict[str, Any],
    calls: List[Dict[str, Any]],
    *,
    max_observations: int = 80,
) -> Dict[str, Any]:
    """Persist successful read observations as mutable task state.

    The entries are not auxiliary logs: response generation and later write-state
    updates may use and update them as the current observed world.  We still
    keep them in an observation namespace instead of asking the LLM to rewrite
    a domain-specific database snapshot for every read.
    """
    import copy as _copy

    next_state = _copy.deepcopy(state) if isinstance(state, dict) else {}
    existing = next_state.get("real_tool_calls")
    observations = list(existing) if isinstance(existing, list) else []
    key_to_index: Dict[str, int] = {}
    for idx, observation in enumerate(observations):
        if not isinstance(observation, dict):
            continue
        obs_name = str(observation.get("name") or "").strip()
        obs_args = observation.get("arguments")
        if obs_name and isinstance(obs_args, dict):
            key_to_index[_observation_key(obs_name, obs_args)] = idx

    for call in calls:
        name = str(call.get("name") or call.get("function") or "").strip()
        if not name:
            continue
        arguments = _copy.deepcopy(call.get("arguments") or call.get("args") or {})
        if not isinstance(arguments, dict):
            arguments = {}
        status, reason = classify_tool_call_status(call.get("result"))
        entry: Dict[str, Any] = {
            "name": name,
            "arguments": arguments,
            "result": _copy.deepcopy(call.get("result")),
            "execution_status": status,
            "state_role": "observed_state",
            "mutable": True,
            "seen_count": 1,
        }
        if reason:
            entry["error_reason"] = reason
        key = _observation_key(name, arguments)
        existing_idx = key_to_index.get(key)
        if existing_idx is not None and 0 <= existing_idx < len(observations):
            prior = observations[existing_idx]
            if isinstance(prior, dict):
                entry["seen_count"] = int(prior.get("seen_count") or 1) + 1
            observations[existing_idx] = entry
        else:
            key_to_index[key] = len(observations)
            observations.append(entry)
    next_state["real_tool_calls"] = observations[-max_observations:]
    return next_state


def _persist_state_snapshot(session_id: Optional[str], state: Dict[str, Any]) -> None:
    if not session_id:
        return
    try:
        from ..handlers.session_handler import session_handler
        session_handler.add_to_state(session_id, state)
    except Exception as exc:
        logger.warning("Failed to persist real-tool observation state for session %s: %s", session_id, exc)


def _build_state_hints_from_tool_entry(tool_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Extract state-update hints from an x-default-state tool metadata entry."""
    hints: Dict[str, Any] = {}
    state_access = tool_entry.get("state_access")
    if isinstance(state_access, str) and state_access.strip():
        hints["state_access"] = state_access.strip().lower()
    for key in (
        "state_effects",
        "state_effects_on_success",
        "state_effects_on_error",
        "state_effects_always",
        "persistent_state_shapes",
        "response_only_fields",
        "behavior_hints",
        "method_calls",
        "response_variants",
        "validation_rules",
    ):
        val = tool_entry.get(key)
        if isinstance(val, list) and val:
            hints[key] = val
    called_method_static_data = tool_entry.get("called_method_static_data")
    if isinstance(called_method_static_data, dict) and called_method_static_data:
        hints["called_method_static_data"] = called_method_static_data
    success_string_templates = tool_entry.get("success_string_templates")
    if isinstance(success_string_templates, dict) and success_string_templates:
        hints["success_string_templates"] = success_string_templates
    return hints


def _prepare_tool_calls_for_state_update(
    tool_calls: List[Dict[str, Any]],
    tool_descriptions: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Normalize tool-call status and split mutation calls from read evidence."""
    prepared: List[Dict[str, Any]] = []
    read_context_calls: List[Dict[str, Any]] = []
    skipped_error_calls = 0

    for source_index, call in enumerate(tool_calls or []):
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
        normalized["_source_index"] = source_index

        if _is_read_only_tool_call(call_name, tool_descriptions):
            if status != "error":
                read_context_calls.append(normalized)
            continue

        if status == "error" and not _has_error_state_effects(call_name, tool_descriptions):
            skipped_error_calls += 1
            continue

        prepared.append(normalized)

    return prepared, read_context_calls, skipped_error_calls


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

        success_effects = state_hints.get("state_effects_on_success")
        always_effects = state_hints.get("state_effects_always")
        persistent_state_shapes = state_hints.get("persistent_state_shapes")
        response_only_fields = state_hints.get("response_only_fields")
        if isinstance(success_effects, list) and success_effects:
            state_effects = list(success_effects)
        else:
            fallback_effects = state_hints.get("state_effects")
            state_effects = list(fallback_effects) if isinstance(fallback_effects, list) else []
        if isinstance(always_effects, list) and always_effects:
            state_effects.extend(always_effects)
        has_shapes = isinstance(persistent_state_shapes, list) and bool(persistent_state_shapes)
        if (not isinstance(state_effects, list) or not state_effects) and not has_shapes:
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
        if not normalized_effects and not has_shapes:
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
        if has_shapes:
            requirements[-1]["persistent_state_shapes"] = persistent_state_shapes
        if isinstance(response_only_fields, list) and response_only_fields:
            requirements[-1]["response_only_fields"] = response_only_fields

    return requirements


def _build_tool_call_contract_context(
    tool_calls: List[Dict[str, Any]],
    tool_descriptions: Optional[Dict[str, Any]],
    read_context_calls: Optional[List[Dict[str, Any]]] = None,
    previous_state: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Annotate tool calls with schema persistence hints for the FC updater."""
    annotated_calls: List[Dict[str, Any]] = []
    descriptions = tool_descriptions if isinstance(tool_descriptions, dict) else {}
    read_context = read_context_calls or []

    for index, call in enumerate(tool_calls or [], start=1):
        if not isinstance(call, dict):
            continue
        annotated = dict(call)
        annotated["call_index"] = index
        write_source_index = annotated.get("_source_index")
        annotated.pop("_source_index", None)
        call_name = annotated.get("name")
        desc_entry = descriptions.get(call_name) if isinstance(call_name, str) else None
        if "toolkit" not in annotated and isinstance(desc_entry, dict):
            toolkit_info = desc_entry.get("toolkit")
            if isinstance(toolkit_info, dict):
                toolkit_name = toolkit_info.get("name")
                if isinstance(toolkit_name, str) and toolkit_name.strip():
                    annotated["toolkit"] = toolkit_name.strip()
        state_hints = desc_entry.get("state_hints") if isinstance(desc_entry, dict) else None
        if isinstance(state_hints, dict):
            state_access = state_hints.get("state_access")
            if isinstance(state_access, str) and state_access.strip():
                annotated["state_access"] = state_access.strip().lower()
            for key in (
                "state_effects",
                "state_effects_on_success",
                "behavior_hints",
                "method_calls",
                "called_method_static_data",
                "success_string_templates",
                "validation_rules",
                "response_variants",
            ):
                value = state_hints.get(key)
                if value:
                    annotated[key] = value
            response_only_fields = state_hints.get("response_only_fields")
            if isinstance(response_only_fields, list) and response_only_fields:
                annotated["response_only_fields"] = response_only_fields
                annotated["persistence_policy"] = (
                    "Fields listed in response_only_fields are response payload only and must not "
                    "be copied into persistent toolkit state unless the contract explicitly stores them."
                )
            persistent_state_shapes = state_hints.get("persistent_state_shapes")
            if isinstance(persistent_state_shapes, list) and persistent_state_shapes:
                annotated["persistent_state_shapes"] = persistent_state_shapes
            method_calls = {
                str(method_name)
                for method_name in state_hints.get("method_calls", [])
                if isinstance(method_name, str) and method_name.strip()
            }
            if method_calls and read_context:
                matched_reads: List[Dict[str, Any]] = []
                for read_call in read_context:
                    if not isinstance(read_call, dict):
                        continue
                    read_name = str(read_call.get("name") or "").strip()
                    if read_name not in method_calls:
                        continue
                    read_source_index = read_call.get("_source_index")
                    if (
                        isinstance(write_source_index, int)
                        and isinstance(read_source_index, int)
                        and read_source_index >= write_source_index
                    ):
                        continue
                    binding = read_evidence_binding_for(state_hints, read_name)
                    if not tool_call_arguments_compatible(
                        annotated.get("arguments"),
                        read_call.get("arguments"),
                        binding=binding,
                    ):
                        continue
                    read_desc = descriptions.get(read_name) if isinstance(read_name, str) else None
                    read_hints = read_desc.get("state_hints") if isinstance(read_desc, dict) else None
                    matched_reads.append(
                        {
                            "name": read_name,
                            "arguments": read_call.get("arguments") or {},
                            "result": read_call.get("result"),
                            "response_only_fields": (
                                read_hints.get("response_only_fields", [])
                                if isinstance(read_hints, dict)
                                else []
                            ),
                            "usage_policy": (
                                "Context-only read evidence matched by this write's method_calls and compatible shared arguments, "
                                "and observed strictly before this write. Use only to derive values explicitly required by this write; "
                                "do not copy the returned record into persistent state or mutate the read toolkit."
                            ),
                        }
                    )
                if matched_reads:
                    annotated["matched_context_only_read_calls"] = matched_reads
        annotated_calls.append(annotated)

    return annotate_prompt_tool_calls_with_hints(
        annotated_calls,
        previous_state=previous_state,
    )


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



def resolve_logical_path(state: Dict[str, Any], logical_path: str) -> Tuple[str, Any]:
    """Resolve a logical path (without 'contents' wrappers) to a JSON Pointer.

    Walks the actual config tree, auto-detecting nodes that use a
    ``{"type": ..., "contents": {...}}`` directory pattern and inserting
    ``/contents`` in the pointer as needed.

    Args:
        state: The current config dict.
        logical_path: Slash-separated key names, e.g.
            ``<ToolkitName>/root/folder/document.txt``

    Returns:
        (json_pointer, node_value) — the resolved JSON Pointer string and the
        value at that location (or ``None`` if the path is new / not found).
    """
    segments = [s for s in logical_path.strip("/").split("/") if s]
    pointer_parts: List[str] = []
    current: Any = state

    for seg in segments:
        if not isinstance(current, dict):
            pointer_parts.append(seg)
            current = None
            continue

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
    ``<ToolkitName>/root/folder/root/folder`` ->
    ``<ToolkitName>/root/folder``
    """
    segments = [s for s in path_str.strip("/").split("/") if s]
    if len(segments) < 5:
        return path_str

    root_idx = 1
    if len(segments) <= root_idx:
        return path_str
    root_key = segments[root_idx]

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
    1. Full absolute: ``<ToolkitName>/root/folder/document``
    2. Partial absolute (from root key): ``folder/document``
    3. CWD-relative: ``document.txt``, ``temp``

    Uses tree-based disambiguation: builds candidate paths and picks the
    one that resolves to an existing node (or whose parent resolves).
    """
    segments = [s for s in raw_path.strip("/").split("/") if s]

    if segments:
        top_keys = set(running.keys()) - {"runtime_state"}
        if segments[0] in top_keys:
            return raw_path

    prefix = _find_cwd_prefix(running)
    if prefix is None:
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

    candidates: List[Tuple[str, List[str]]] = []
    if segments:
        if segments[0] == root_key:
            candidates.append(("toolkit_root_absolute", [toolkit_name] + segments))
            candidates.append(("cwd_relative", prefix + segments))
        else:
            candidates.append(("cwd_relative", prefix + segments))
        candidates.append(("root_relative", toolkit_root + segments))
    else:
        candidates.append(("cwd_self", list(prefix)))

    for label, parts in candidates:
        path_str = "/".join(parts)
        _, node = resolve_logical_path(running, path_str)
        if node is not None:
            logger.debug("[FC PATH] '%s' → '%s' (%s, leaf exists)", raw_path, path_str, label)
            return path_str

    for label, parts in candidates:
        if len(parts) <= 1:
            continue
        path_str = "/".join(parts)
        parent_str = "/".join(parts[:-1])
        _, parent_node = resolve_logical_path(running, parent_str)
        if parent_node is not None:
            logger.debug("[FC PATH] '%s' → '%s' (%s, parent exists)", raw_path, path_str, label)
            return path_str

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
    running = _copy.deepcopy(state)

    for req in fc_requests:
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
            pass
        else:
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
            step_ops.append({"op": "add", "path": pointer, "value": value})
        else:
            step_ops.append({"op": "replace", "path": pointer, "value": value})

    else:
        raise ValueError(f"[FC STATE] Unknown tool: {name}")


def _json_for_prompt(obj: Any, *, compact: bool = False) -> str:
    try:
        if compact:
            return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        return str(obj)


def _is_state_absolute_path(path: str, state: Dict[str, Any]) -> bool:
    first = path.strip("/").split("/", 1)[0]
    return bool(first) and isinstance(state, dict) and first in state


def _logical_path_from_runtime_path(path: str, state: Dict[str, Any]) -> Optional[str]:
    """Convert '/workspace/docs' to '<Toolkit>/root/workspace/docs' when possible."""
    if not isinstance(path, str) or not path.startswith("/"):
        return None
    segments = [segment for segment in path.strip("/").split("/") if segment]
    if not segments:
        return None
    for toolkit_name, toolkit_state in state.items():
        if toolkit_name == "runtime_state" or not isinstance(toolkit_state, dict):
            continue
        root_state = toolkit_state.get("root")
        if isinstance(root_state, dict) and segments[0] in root_state:
            return "/".join([toolkit_name, "root"] + segments)
    return None


def _normalize_dotted_path_suffix(path_suffix: str) -> str:
    parts: List[str] = []
    bracket_depth = 0
    for char in path_suffix:
        if char == "[":
            bracket_depth += 1
            parts.append(char)
            continue
        if char == "]":
            bracket_depth = max(0, bracket_depth - 1)
            parts.append(char)
            continue
        if char == "." and bracket_depth == 0:
            parts.append("/")
            continue
        parts.append(char)
    return "".join(parts)


def _normalize_direct_path_syntax(path: str, state: Dict[str, Any]) -> str:
    """Accept common dotted/bracket paths for top-level toolkit state."""
    normalized_path = path
    for top_key in state:
        if path == top_key:
            normalized_path = path
            break
        marker = f"{top_key}."
        if path.startswith(marker):
            rest = path[len(top_key):]
            normalized_path = f"{top_key}{_normalize_dotted_path_suffix(rest)}"
            break

    segments: List[str] = []
    for segment in normalized_path.split("/"):
        expanded_segments: List[str] = []
        cursor = 0
        for match in re.finditer(r"\[([^\[\]]+)\]", segment):
            prefix = segment[cursor:match.start()]
            if prefix:
                expanded_segments.append(prefix)
            expanded_segments.append(match.group(1))
            cursor = match.end()
        suffix = segment[cursor:]
        if suffix:
            if cursor > 0 and suffix.startswith("."):
                expanded_segments.extend(part for part in suffix.lstrip(".").split(".") if part)
            else:
                expanded_segments.append(suffix)
        if not expanded_segments:
            expanded_segments.append(segment)

        for expanded in expanded_segments:
            if expanded.endswith(".contents") and len(expanded) > len(".contents"):
                segments.append(expanded[: -len(".contents")])
            else:
                segments.append(expanded)
    return "/".join(segments)


def _resolve_direct_patch_path(
    running: Dict[str, Any],
    raw_path: Any,
    *,
    op_name: str = "set",
) -> Tuple[str, str, Any]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"Patch path must be a non-empty string, got {raw_path!r}")

    path = _normalize_direct_path_syntax(raw_path.strip(), running)
    if path.startswith("/") and _is_state_absolute_path(path, running):
        path = path.strip("/")
    elif path.startswith("/"):
        runtime_logical = _logical_path_from_runtime_path(path, running)
        path = runtime_logical if runtime_logical else path.strip("/")

    top = path.split("/", 1)[0]
    if top in running:
        logical_path = path
    else:
        logical_path = _resolve_entry_path(running, path)

    pointer, node = resolve_direct_logical_path(
        running,
        logical_path,
        prefer_append_for_missing_leaf=op_name in {"add", "set"},
    )
    return logical_path, pointer, node


def _normalize_direct_set_value(running: Dict[str, Any], logical_path: str, value: Any) -> Any:
    if (
        logical_path.endswith("/current_working_directory")
        and isinstance(value, str)
        and not value.startswith("/")
    ):
        parts = [part for part in logical_path.split("/") if part]
        toolkit = parts[-2] if len(parts) >= 2 else ""
        return _normalize_cwd_value(running, toolkit, value) if toolkit else value
    return value


def _direct_patch_op_to_json_patch(running: Dict[str, Any], op: Dict[str, Any]) -> List[Dict[str, Any]]:
    op_name = op.get("op")
    if op_name in {"noop", "no_change"}:
        return []
    if op_name == "replace":
        op_name = "set"
    if op_name not in {"set", "add", "remove", "move", "copy"}:
        raise ValueError(f"Unsupported direct patch op: {op_name!r}")

    if op_name in {"set", "add", "remove"}:
        logical_path, pointer, node = _resolve_direct_patch_path(
            running,
            op.get("path"),
            op_name=op_name,
        )
        if op_name == "remove":
            if node is None:
                return []
            return [{"op": "remove", "path": pointer}]

        value = op.get("value")
        if op_name == "set":
            value = _normalize_direct_set_value(running, logical_path, value)
            return [{"op": "replace" if node is not None else "add", "path": pointer, "value": value}]
        return [{"op": "add", "path": pointer, "value": value}]

    from_logical, from_pointer, from_node = _resolve_direct_patch_path(
        running,
        op.get("from"),
        op_name="remove",
    )
    if from_node is None:
        return []
    to_logical, to_pointer, to_node = _resolve_direct_patch_path(
        running,
        op.get("path"),
        op_name="add",
    )
    if isinstance(to_node, dict) and to_node.get("type") == "directory":
        basename = from_logical.rstrip("/").rsplit("/", 1)[-1]
        to_logical = f"{to_logical.rstrip('/')}/{basename}"
        to_pointer = _child_pointer(running, to_logical.rsplit("/", 1)[0], basename)

    import copy as _copy
    if op_name == "move":
        return [
            {"op": "remove", "path": from_pointer},
            {"op": "add", "path": to_pointer, "value": _copy.deepcopy(from_node)},
        ]
    return [{"op": "add", "path": to_pointer, "value": _copy.deepcopy(from_node)}]


def apply_direct_logical_patch(
    previous_state: Dict[str, Any],
    patch_ops: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Apply direct logical patch ops and return (updated_state, json_patch_ops)."""
    import copy as _copy

    if not isinstance(patch_ops, list):
        raise ValueError("Direct patch must be a list")

    running = _copy.deepcopy(previous_state)
    json_patch_ops: List[Dict[str, Any]] = []
    for idx, op in enumerate(patch_ops):
        if not isinstance(op, dict):
            raise ValueError(f"Patch op {idx} must be an object")
        try:
            step_ops = _direct_patch_op_to_json_patch(running, op)
        except Exception as exc:
            logger.warning("[DIRECT STATE] Skipping failed direct op %d: %s", idx, exc)
            continue
        if not step_ops:
            continue
        running = apply_json_patch(running, step_ops)
        json_patch_ops.extend(step_ops)
    return running, json_patch_ops


def _parse_direct_patch_response(raw_text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cleaned = sanitize_llm_json_text(raw_text)
    payload = json_repair.loads(cleaned)
    if isinstance(payload, list):
        _validate_direct_patch_ops(payload)
        return payload, {"patch": payload}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object or patch array, got {type(payload).__name__}")
    patch = payload.get("patch")
    if not isinstance(patch, list):
        raise ValueError("Patch response must contain a patch array")
    _validate_direct_patch_ops(patch)
    return patch, payload


def _parse_structured_direct_patch_response(response: Any, raw_text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    msg = getattr(response, "msg", None)
    parsed = getattr(msg, "parsed", None)

    if isinstance(parsed, DirectPatchResponse):
        payload = parsed.model_dump(by_alias=True)
        patch = payload.get("patch", [])
        _validate_direct_patch_ops(patch)
        return patch, payload

    if isinstance(parsed, BaseModel):
        parsed = parsed.model_dump(by_alias=True)

    if isinstance(parsed, dict):
        payload = DirectPatchResponse.model_validate(parsed).model_dump(by_alias=True)
        patch = payload.get("patch", [])
        _validate_direct_patch_ops(patch)
        return patch, payload

    return _parse_direct_patch_response(raw_text)


def _direct_patch_response_text(response: Any) -> str:
    msg = getattr(response, "msg", None)
    content = getattr(msg, "content", None)
    if isinstance(content, str) and content.strip():
        return content

    parsed = getattr(msg, "parsed", None)
    if isinstance(parsed, BaseModel):
        return parsed.model_dump_json(by_alias=True)
    if parsed is not None:
        try:
            return json.dumps(parsed, ensure_ascii=False, default=str)
        except Exception:
            return str(parsed)
    return ""


def _validate_direct_patch_ops(patch_ops: List[Any]) -> None:
    if not isinstance(patch_ops, list):
        raise ValueError("Direct patch must be a list")

    allowed_ops = {"set", "add", "replace", "remove", "move", "copy", "noop", "no_change"}
    for idx, op in enumerate(patch_ops):
        if not isinstance(op, dict):
            raise ValueError(f"Patch op {idx} must be an object, got {type(op).__name__}")

        op_name = op.get("op")
        if not isinstance(op_name, str) or not op_name:
            raise ValueError(f"Patch op {idx} must include string 'op'")
        if op_name not in allowed_ops:
            raise ValueError(f"Patch op {idx} has unsupported op {op_name!r}")

        if op_name in {"noop", "no_change"}:
            continue

        path = op.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"Patch op {idx} must include non-empty string 'path'")

        if op_name in {"set", "add", "replace"} and "value" not in op:
            raise ValueError(f"Patch op {idx} with op {op_name!r} must include 'value'")

        if op_name in {"move", "copy"}:
            from_path = op.get("from")
            if not isinstance(from_path, str) or not from_path.strip():
                raise ValueError(f"Patch op {idx} with op {op_name!r} must include non-empty string 'from'")


def _build_direct_patch_repair_query(state_query: str, raw_response: str, error: Exception) -> str:
    return "\n\n".join(
        [
            state_query,
            "The previous direct logical patch response was rejected before applying it.",
            f"Validation error: {type(error).__name__}: {str(error)[:1000]}",
            "Rejected response:\n" + (raw_response or "<empty>")[:4000],
            (
                "Return a corrected JSON object only. It must have the shape "
                '{"patch":[{"op":"set|add|replace|remove|move|copy|noop|no_change",...}],'
                '"reasoning":"..."} and every patch array item must be an object.'
            ),
        ]
    )


def _build_context_only_read_observations(read_context_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for call in read_context_calls or []:
        if not isinstance(call, dict):
            continue
        result = call.get("result")
        if not isinstance(result, dict):
            continue
        for field_name, value in result.items():
            observation: Dict[str, Any] = {
                "source_tool": call.get("name"),
                "arguments": call.get("arguments") or {},
                "field": field_name,
                "value": value,
                "usage_policy": (
                    "Transient read observation: may be consumed to derive later write-state mutations, "
                    "but this field itself must not be persisted unless a later write call explicitly stores it."
                ),
            }
            if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (str, int, float, bool)):
                observation["singleton_scalar_candidate"] = value[0]
            observations.append(observation)
    return observations


def _update_state_via_direct_patch(
    previous_state: Dict[str, Any],
    tool_calls: List[Dict[str, Any]],
    task: Optional[str] = None,
    execution_results: Optional[List[Any]] = None,
    tool_descriptions: Optional[Dict[str, Any]] = None,
    read_context_calls: Optional[List[Dict[str, Any]]] = None,
    session_id: Optional[str] = None,
    state_model: Optional[str] = None,
) -> Dict[str, Any]:
    """State update using direct logical patch JSON instead of FC tools."""
    import copy as _copy

    if state_model is None:
        state_model = get_state_model()
    if state_model is None:
        raise RuntimeError("[DIRECT STATE] No state model configured for direct patch mode")

    tool_descriptions = rewrite_persistent_state_paths_to_existing(tool_descriptions, previous_state)
    required_state_effects = _extract_required_state_effects(tool_calls, tool_descriptions)
    annotated_tool_calls = _build_tool_call_contract_context(
        tool_calls,
        tool_descriptions,
        read_context_calls=read_context_calls,
        previous_state=previous_state,
    )
    try:
        from ..schemas.global_loader import get_global_schema_loader
        _schema_loader_for_enrichment = get_global_schema_loader()
    except Exception:
        _schema_loader_for_enrichment = None
    annotated_tool_calls = enrich_annotated_calls(
        annotated_tool_calls,
        tool_calls,
        previous_state=previous_state,
        schema_loader=_schema_loader_for_enrichment,
    )

    query_sections: List[str] = []
    query_sections.append("State format: raw JSON. Patch output paths must be logical paths, not JSON Pointers.")
    context_label = state_context_label(previous_state)
    if context_label:
        query_sections.append(f"State context: {context_label}")
    query_sections.append(
        "Authoritative state policy: each field has exactly one canonical home in previous_state. "
        "Always write to that exact path. Never duplicate the same field across "
        "`runtime_state.toolkits.<X>.<field>` and `<X>/<field>`."
    )
    query_sections.append(
        "Observed-state policy: `real_tool_calls` entries with `state_role=observed_state` "
        "are mutable task state, not auxiliary logs. Use matching entries as known entity snapshots "
        "or candidate results for later writes, update the matching entry when a later write "
        "changes that observed entity, and treat search/list observations as returned "
        "candidates rather than exhaustive databases."
    )
    if task:
        query_sections.append(
            "Background task (non-authoritative; do not override previous state "
            f"or executed tool results): {task}"
        )
    query_sections.append(
        "Persistence policy:\n"
        "- Calls marked state_access=read are evidence only and must not be copied into persistent state.\n"
        "- `real_tool_calls` entries with `state_role=observed_state` are mutable observed task state, not auxiliary logs. Use matching observations as known entity snapshots or candidate results for later writes, and update the matching entry when a later write changes that observed entity.\n"
        "- Search/list observations describe returned candidates only; they are not proof that omitted records do not exist.\n"
        "- Fields listed in response_only_fields are transient response payload only unless a later write explicitly stores them.\n"
        "- resolved_effect_hints are exact concrete mutations; use them directly when present."
    )
    query_sections.append(
        "Derived value policy:\n"
        "- Later write calls may consume earlier context_only_read results to fill computed values required by their state_effects or persistent_state_shapes.\n"
        "- If a write call contract requires a computed field that is not present in the write result, prefer matching earlier read-call results or called_method_static_data. Do not substitute unrelated existing state values.\n"
        "- If a context_only_read observation exposes a singleton_scalar_candidate, treat it as the preferred scalar candidate for a later write that needs one derived value.\n"
        "- If a required durable generated/random value is not echoed by the tool result, use a schema-valid realistic value rather than a validation-breaking placeholder."
    )
    if required_state_effects:
        query_sections.append(
            "Required state effects for successful calls:\n"
            f"{_json_for_prompt(required_state_effects)}"
        )
    observations = _build_context_only_read_observations(read_context_calls or [])
    if observations:
        query_sections.append(f"Context-only read observations:\n{_json_for_prompt(observations)}")
    query_sections.append(f"Previous task state:\n{_json_for_prompt(previous_state)}")
    query_sections.append(
        f"Executed mutation-candidate tool calls ({len(annotated_tool_calls)}, in order):\n"
        f"{_json_for_prompt(annotated_tool_calls)}"
    )
    if execution_results:
        query_sections.append(f"Execution results:\n{_json_for_prompt(execution_results)}")
    if tool_descriptions:
        query_sections.append(f"Tool descriptions:\n{_json_for_prompt(tool_descriptions)}")
    query_sections.append(
        "Return the direct logical patch JSON object only. Do not include markdown fences or prose."
    )
    state_query = "\n\n".join(query_sections)

    state_agent = ChatAgent(
        _DIRECT_PATCH_SYSTEM_PROMPT,
        model=create_model(
            state_model,
            max_tokens=8192,
            temperature=0.001,
            timeout=STATE_AGENT_TIMEOUT_SECONDS,
        ),
        step_timeout=STATE_AGENT_TIMEOUT_SECONDS,
        tool_execution_timeout=STATE_AGENT_TIMEOUT_SECONDS,
    )

    patch_ops: List[Dict[str, Any]] = []
    raw_response = ""
    try:
        payload: Dict[str, Any] = {}
        patch_query = state_query
        last_format_error: Optional[Exception] = None
        for attempt_idx in range(2):
            _ct0 = datetime.now()
            logger.debug("[DIRECT STATE] LLM START (model=%s, attempt=%d)", state_model, attempt_idx + 1)
            with bind_log_context(agent_role="gecko_state_updater"):
                state_response = state_agent.step(patch_query)
            _ct1 = datetime.now()
            logger.debug("[DIRECT STATE] LLM END (elapsed=%.3fs, attempt=%d)", (_ct1 - _ct0).total_seconds(), attempt_idx + 1)
            raw_response = _direct_patch_response_text(state_response)
            try:
                patch_ops, payload = _parse_structured_direct_patch_response(state_response, raw_response)
                break
            except Exception as format_error:
                last_format_error = format_error
                if attempt_idx == 1:
                    raise DirectPatchFormatError(
                        f"State model returned invalid direct patch after retry: {format_error}"
                    ) from format_error
                logger.warning(
                    "[DIRECT STATE] Invalid patch response, retrying once: %s",
                    format_error,
                )
                patch_query = _build_direct_patch_repair_query(state_query, raw_response, format_error)
        else:
            if last_format_error is not None:
                raise last_format_error

        reasoning = payload.get("reasoning") if isinstance(payload, dict) else None
        if reasoning:
            logger.info("[DIRECT STATE] Reasoning: %s", reasoning)
    except DirectPatchFormatError:
        raise
    except Exception as llm_exc:
        logger.warning("[DIRECT STATE] LLM patch failed: %s - degrading to auto-merge", llm_exc)
        state_result = _copy.deepcopy(previous_state)
    else:
        state_result, json_patch_ops = apply_direct_logical_patch(previous_state, patch_ops)
        try:
            logger.info(
                "[DIRECT STATE PATCH] logical=%s json=%s",
                json.dumps(patch_ops, ensure_ascii=False, indent=2, default=str),
                json.dumps(json_patch_ops, ensure_ascii=False, indent=2, default=str),
            )
        except Exception as exc:
            logger.info("[DIRECT STATE PATCH] ops: <unprintable> (%s)", exc)
        logger.info(
            "[DIRECT STATE] Applied %d JSON patch ops from %d logical ops",
            len(json_patch_ops),
            len(patch_ops),
        )

    if session_id:
        try:
            from ..handlers.session_handler import session_handler
            session_handler.add_to_state(session_id, state_result)
        except Exception as e:
            logger.warning(f"Failed to auto-update session {session_id}: {e}")

    return state_result


def _update_state_via_fc(
    previous_state: Dict[str, Any],
    tool_calls: List[Dict[str, Any]],
    task: Optional[str] = None,
    execution_results: Optional[List[Any]] = None,
    tool_descriptions: Optional[Dict[str, Any]] = None,
    read_context_calls: Optional[List[Dict[str, Any]]] = None,
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
        model=create_model(
            state_model,
            max_tokens=8192,
            temperature=0.001,
            timeout=STATE_AGENT_TIMEOUT_SECONDS,
        ),
        external_tools=fc_tools,
        step_timeout=STATE_AGENT_TIMEOUT_SECONDS,
        tool_execution_timeout=STATE_AGENT_TIMEOUT_SECONDS,
    )

    def _json_or_str(obj: Any) -> str:
        try:
            return json.dumps(obj, indent=2)
        except Exception:
            return str(obj)

    query_sections: List[str] = []
    required_state_effects = _extract_required_state_effects(tool_calls, tool_descriptions)
    query_sections.append(
        "Authoritative state policy: each field has exactly one canonical home in previous_state. "
        "Always write to that exact path. Never duplicate the same field across "
        "`runtime_state.toolkits.<X>.<field>` and `<X>/<field>`."
    )
    query_sections.append(
        "Observed-state policy: `real_tool_calls` entries with `state_role=observed_state` "
        "are mutable task state, not auxiliary logs. Use matching entries as known entity snapshots "
        "or candidate results for later writes, update the matching entry when a later write "
        "changes that observed entity, and treat search/list observations as returned "
        "candidates rather than exhaustive databases."
    )
    if task:
        query_sections.append(
            "Background task (non-authoritative; do not override previous state "
            f"or executed tool results): {task}"
        )
    if required_state_effects:
        query_sections.append(
            "Required state effects for successful calls:\n"
            f"{_json_or_str(required_state_effects)}"
        )
    query_sections.append(f"Previous state:\n{_json_or_str(previous_state)}")
    annotated_tool_calls = _build_tool_call_contract_context(
        tool_calls,
        tool_descriptions,
        read_context_calls=read_context_calls,
        previous_state=previous_state,
    )
    query_sections.append(
        f"Tool calls ({len(annotated_tool_calls)}, executed in order):\n"
        f"{_json_or_str(annotated_tool_calls)}"
    )
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
        with bind_log_context(agent_role="gecko_state_updater"):
            state_response = state_agent.step(state_query)
        _ct1 = datetime.now()
        logger.debug("[FC STATE] LLM END (elapsed=%.3fs)", (_ct1 - _ct0).total_seconds())

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
2. If toolkit summaries include runtime_defaults or init_rules from x-default-state, treat them as hints about what runtime context matters and how it is usually initialized.
3. Resolve symbolic sources such as "root" against concrete values present in the initial data config. Do not invent generic placeholders or copy example values that are not present in the config.
4. Emit JSON Patch operations that add/initialize runtime state.

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
        {"op":"add","path":"/runtime_state/toolkits/<ToolkitName>","value":{"current_working_directory":"/actual-root-from-config"}}
  ],
    "reasoning":"Initialized runtime context from the initial state layout and schema hints."
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
          runtime_defaults: {...}  # bootstrap hints filtered from info.x-default-state.global.runtime_defaults
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
                    init_rules = extracted_runtime_defaults.get("init_rules")
                    if isinstance(init_rules, list) and init_rules:
                        runtime_defaults["init_rules"] = init_rules
                    for field_name, field_value in extracted_runtime_defaults.items():
                        if field_name == "init_rules":
                            continue
                        if not isinstance(field_value, dict):
                            continue
                        init_from = field_value.get("init_from")
                        if isinstance(init_from, str) and init_from.strip():
                            runtime_defaults[field_name] = field_value
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


def _resolve_bootstrap_hint_value(
    initial_state: Dict[str, Any],
    toolkit_name: str,
    init_from: str,
) -> Optional[Any]:
    """Resolve a schema-declared bootstrap hint to a concrete initial value."""
    source = str(init_from or "").strip()
    if not source:
        return None
    if source.startswith("/"):
        return source

    toolkit_state = initial_state.get(toolkit_name)
    if not isinstance(toolkit_state, dict):
        return None

    if source.lower() == "root":
        root_container = toolkit_state.get("root")
        if isinstance(root_container, dict):
            root_children = [
                key for key, value in root_container.items()
                if isinstance(key, str) and key.strip() and isinstance(value, dict)
            ]
            if len(root_children) == 1:
                return "/" + root_children[0].strip("/")
        return None

    direct_value = toolkit_state.get(source)
    if isinstance(direct_value, (str, int, float, bool)) or direct_value is None:
        return direct_value
    return None


def _apply_bootstrap_hint_guards(
    initial_state: Dict[str, Any],
    enriched_config: Dict[str, Any],
    toolkit_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply explicit init_from hints as a narrow post-LLM bootstrap guard."""
    import copy as _copy

    guarded = _copy.deepcopy(enriched_config)
    runtime_state = guarded.setdefault("runtime_state", {})
    if not isinstance(runtime_state, dict):
        runtime_state = {}
        guarded["runtime_state"] = runtime_state
    toolkits_state = runtime_state.setdefault("toolkits", {})
    if not isinstance(toolkits_state, dict):
        toolkits_state = {}
        runtime_state["toolkits"] = toolkits_state

    applied_guards = 0
    seen_fields: set[tuple[str, str]] = set()
    for summary in toolkit_summaries:
        if not isinstance(summary, dict):
            continue
        toolkit_name = summary.get("name")
        runtime_defaults = summary.get("runtime_defaults")
        if not isinstance(toolkit_name, str) or not toolkit_name:
            continue
        if not isinstance(runtime_defaults, dict):
            continue

        init_rules = runtime_defaults.get("init_rules")
        if not isinstance(init_rules, list):
            continue

        for rule in init_rules:
            if not isinstance(rule, dict):
                continue
            field_name = rule.get("field")
            init_from = rule.get("init_from")
            if not isinstance(field_name, str) or not field_name.strip():
                continue
            if not isinstance(init_from, str) or not init_from.strip():
                continue
            marker = (toolkit_name, field_name)
            if marker in seen_fields:
                continue
            seen_fields.add(marker)

            expected_value = _resolve_bootstrap_hint_value(initial_state, toolkit_name, init_from)
            if expected_value is None:
                continue

            toolkit_runtime = toolkits_state.setdefault(toolkit_name, {})
            if not isinstance(toolkit_runtime, dict):
                toolkit_runtime = {}
                toolkits_state[toolkit_name] = toolkit_runtime
            if toolkit_runtime.get(field_name) != expected_value:
                toolkit_runtime[field_name] = expected_value
                applied_guards += 1

    if applied_guards:
        logger.info(
            "[BOOTSTRAP] Applied %d explicit init_from guard updates after LLM bootstrap",
            applied_guards,
        )

    return guarded


def bootstrap_state(
    initial_state: Dict[str, Any],
    toolkit_summaries: List[Dict[str, Any]],
    state_model: Optional[str] = None,
    allow_llm_fallback: bool = False,
) -> Dict[str, Any]:
    """Bootstrap initial config by inferring runtime state with the state model.

    Uses toolkit descriptions, x-default-state hints, and the initial data config
    to infer runtime context (e.g. current working directory, authenticated user)
    and emit JSON Patch operations.

    Args:
        initial_state: Raw initial config (may be empty {})
        toolkit_summaries: List of {name, description, operations: [{id, summary}]}
            as returned by extract_toolkit_summaries()
        state_model: LLM model to use (default: uses global config)
        allow_llm_fallback: Retained for backward compatibility. Bootstrap is
            now LLM-first whenever a state model is available.

    Returns:
        Enriched config with runtime_state initialized via JSON Patch
    """
    if state_model is None:
        state_model = get_state_model()
    if state_model is None:
        logger.info("[BOOTSTRAP] No state model available, returning state as-is")
        return initial_state
    if not isinstance(toolkit_summaries, list) or not toolkit_summaries:
        logger.info("[BOOTSTRAP] No toolkit summaries available, returning state as-is")
        return initial_state

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
        model=create_model(
            state_model,
            max_tokens=8192,
            temperature=0.001,
            timeout=STATE_AGENT_TIMEOUT_SECONDS,
        ),
        step_timeout=STATE_AGENT_TIMEOUT_SECONDS,
        tool_execution_timeout=STATE_AGENT_TIMEOUT_SECONDS,
    )

    _t0 = datetime.now()
    logger.info(f"[BOOTSTRAP] LLM START (model={state_model})")
    with bind_log_context(agent_role="gecko_bootstrap"):
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
        patch_ops = payload

    if patch_ops is None:
        logger.warning("[BOOTSTRAP] Missing valid 'patch' array, returning state as-is")
        return initial_state

    try:
        enriched_config = apply_json_patch(initial_state, patch_ops)
        enriched_config = _apply_bootstrap_hint_guards(
            initial_state=initial_state,
            enriched_config=enriched_config,
            toolkit_summaries=toolkit_summaries,
        )
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

    prepared_calls, read_context_calls, skipped_error_calls = _prepare_tool_calls_for_state_update(
        tool_calls=tool_calls,
        tool_descriptions=tool_descriptions,
    )
    if read_context_calls:
        logger.info(
            "[STATE UPDATE] Preserved %d read-only tool calls as context-only evidence",
            len(read_context_calls),
        )
    if skipped_error_calls > 0:
        logger.info(
            "[STATE UPDATE] Skipped %d error tool calls with no explicit error-state effects",
            skipped_error_calls,
        )

    if not prepared_calls:
        logger.info("[STATE UPDATE] No tool calls eligible for state update; state unchanged")
        return previous_state.copy() if isinstance(previous_state, dict) else previous_state

    updater_mode = os.environ.get("GECKO_STATE_UPDATER_MODE", "direct").strip().lower()
    if updater_mode in {"fc", "function_call", "function-calling"}:
        logger.info("[STATE UPDATE] Using legacy function-calling updater")
        result = _update_state_via_fc(
            previous_state=previous_state,
            tool_calls=prepared_calls,
            task=task,
            execution_results=execution_results,
            tool_descriptions=tool_descriptions,
            read_context_calls=read_context_calls,
            session_id=session_id,
            state_model=state_model,
        )
    else:
        if updater_mode not in {"direct", "direct_patch", "logical_patch", ""}:
            logger.warning(
                "[STATE UPDATE] Unknown GECKO_STATE_UPDATER_MODE=%r; using direct logical-patch updater",
                updater_mode,
            )
        logger.info("[STATE UPDATE] Using direct logical-patch updater")
        result = _update_state_via_direct_patch(
            previous_state=previous_state,
            tool_calls=prepared_calls,
            task=task,
            execution_results=execution_results,
            tool_descriptions=tool_descriptions,
            read_context_calls=read_context_calls,
            session_id=session_id,
            state_model=state_model,
        )
    return result


def update_state_from_real_tool(
    previous_state: Dict[str, Any],
    tool_call: Any,
    session_id: Optional[str] = None,
    state_model: Optional[str] = None,
    task: Optional[str] = None,
    involved_classes: Optional[List[str]] = None,
    read_only_sync_policy: Optional[str] = None,
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
        read_only_sync_policy: Optional sync behavior. The legacy "history_only"
            value is accepted; schema-declared read-only calls are now persisted
            as observed state without invoking the LLM mutation updater.

    Returns:
        Updated configuration dictionary

    Example:
        >>> update_state_from_real_tool(
        ...     previous_state={'users': []},
        ...     tool_call={'name': 'get_user', 'arguments': {'id': 1}, 'result': {'id': 1, 'name': 'Alice'}}
        ... )
    """
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

    formatted_calls = []
    for idx, tc in enumerate(raw_tool_calls):
        try:
            normalized_tc = normalize_bfcl_multi_turn_tool_call(
                tc,
                rename=True,
                preserve_original=True,
            )
            function_name = (
                normalized_tc.get('name')
                or normalized_tc.get('function')
                or normalized_tc.get('function_name')
                or ''
            )

            raw_arguments = normalized_tc.get('arguments') or normalized_tc.get('args') or {}
            arguments = _normalize_real_arguments(raw_arguments)

            if not function_name and isinstance(raw_arguments, dict):
                nested_name = raw_arguments.get("_tool_name")
                if isinstance(nested_name, str) and nested_name.strip():
                    function_name = nested_name.strip()

            result = normalized_tc.get('result')
            status, reason = classify_tool_call_status(result)

            if not function_name:
                logger.warning(f"Tool call {idx} missing function name, skipping")
                continue

            formatted_call = {
                'name': function_name,
                'arguments': arguments,
                'result': result,
                'execution_status': status,
                'error_reason': reason,
            }
            if normalized_tc.get("toolkit"):
                formatted_call["toolkit"] = normalized_tc["toolkit"]
            formatted_calls.append(formatted_call)
        except Exception as e:
            logger.error(f"Error formatting tool call {idx}: {e}")
            continue

    if not formatted_calls:
        logger.warning("No valid tool calls after formatting, returning previous state unchanged")
        return previous_state.copy() if isinstance(previous_state, dict) else previous_state

    if involved_classes:
        logger.debug(f"Involved classes: {involved_classes}")
    logger.info(f"[STATE UPDATE] Processing {len(formatted_calls)} tool calls in batch mode")

    normalized_sync_policy = (read_only_sync_policy or "").strip().lower()
    observed_read_sync = normalized_sync_policy in {
        "history_only",
        "context_only",
        "read_only_history_only",
        "observed_state",
        "upsert_observed_state",
    }
    tool_descriptions: Optional[Dict[str, Any]] = None
    try:
        from ..schemas.global_loader import get_global_schema_loader
        schema_loader = get_global_schema_loader()
        if schema_loader is not None:
            _td: Dict[str, Any] = {}
            for fc in formatted_calls:
                fname = fc.get("name", "")
                toolkit_name = fc.get("toolkit", "")
                op_id = fname
                schema_file = schema_loader.find_schema_file(toolkit_name) if toolkit_name else None
                if not schema_file:
                    parts = fname.split("_", 1)
                    if len(parts) == 2:
                        toolkit_prefix, candidate_op_id = parts
                        candidate_schema_file = schema_loader.find_schema_file(toolkit_prefix) if toolkit_prefix else None
                        if candidate_schema_file:
                            schema_file = candidate_schema_file
                            op_id = candidate_op_id
                    else:
                        toolkit_prefix = ""
                if not schema_file and fname:
                    schema_file = schema_loader.find_schema_file(fname)
                if schema_file:
                    schema = schema_loader.load_schema(schema_file)
                    x_ds = schema.get("info", {}).get("x-default-state", {})
                    tools_block = x_ds.get("tools", {})
                    tool_entry = tools_block.get(op_id, {})
                    if isinstance(tool_entry, dict) and tool_entry:
                        hints = _build_state_hints_from_tool_entry(tool_entry)
                        if hints:
                            info = schema.get("info", {})
                            toolkit_info: Dict[str, Any] = {}
                            title = info.get("title")
                            if isinstance(title, str) and title:
                                toolkit_info["name"] = title
                            td_entry: Dict[str, Any] = {"state_hints": hints}
                            if toolkit_info:
                                td_entry["toolkit"] = toolkit_info
                            _td[fname] = td_entry
                if fname not in _td and hasattr(schema_loader, "find_tool_entry"):
                    match = schema_loader.find_tool_entry(fname, toolkit_name or None)
                    if match:
                        _, schema, tool_entry = match
                        hints = _build_state_hints_from_tool_entry(tool_entry)
                        if hints:
                            info = schema.get("info", {})
                            toolkit_info: Dict[str, Any] = {}
                            title = info.get("title")
                            if isinstance(title, str) and title:
                                toolkit_info["name"] = title
                            td_entry = {"state_hints": hints}
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

    if task is None:
        task = "Register entities discovered by Real Tool execution so Mock Tools can reference them (Hybrid Mode)"

    read_observation_calls: List[Dict[str, Any]] = []
    state_update_calls: List[Dict[str, Any]] = []
    for fc in formatted_calls:
        if observed_read_sync and _is_observable_read_call(fc, tool_descriptions):
            read_observation_calls.append(fc)
        else:
            state_update_calls.append(fc)

    if read_observation_calls:
        logger.info(
            "[STATE UPDATE] Persisting %d successful real read-only calls as observed state",
            len(read_observation_calls),
        )
    if normalized_sync_policy and not observed_read_sync:
        logger.warning("[STATE UPDATE] Unknown read_only_sync_policy=%r; using default state sync", read_only_sync_policy)

    running_state = previous_state
    succeeded = 0
    observed = 0
    try:
        for idx, fc in enumerate(formatted_calls):
            if observed_read_sync and _is_observable_read_call(fc, tool_descriptions):
                running_state = _upsert_real_tool_observations(running_state, [fc])
                _persist_state_snapshot(session_id, running_state)
                observed += 1
                continue

            fname = fc.get("name", "")
            per_call_descriptions: Optional[Dict[str, Any]] = None
            if tool_descriptions and fname in tool_descriptions:
                per_call_descriptions = {fname: tool_descriptions[fname]}
            running_state = update_state(
                previous_state=running_state,
                tool_calls=[fc],
                task=task,
                session_id=session_id,
                state_model=state_model,
                tool_descriptions=per_call_descriptions,
            )
            succeeded += 1
        if state_update_calls:
            logger.info(
                "[STATE UPDATE] Sequential update completed successfully for %d/%d state-updating tool calls; observed %d read calls",
                succeeded,
                len(state_update_calls),
                observed,
            )
        else:
            logger.info("[STATE UPDATE] No real tool calls require LLM mutation update; observed %d read calls", observed)
        return running_state.copy() if isinstance(running_state, dict) else running_state
    except Exception as e:
        logger.error(
            f"[STATE UPDATE] Sequential update failed at call {succeeded}/{len(formatted_calls)}: {e}",
            exc_info=True,
        )
        raise


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

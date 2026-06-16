"""Per-call prompt enrichment helpers for the state updater.

These supplement the contract-grounded hints in :mod:`state_update_hints` with
information lifted directly from the OpenAPI schema and the previous state
snapshot:

- ``operation_schema``: method/path/summary/description/request_properties for
  the operation, read from the toolkit's OpenAPI ``paths`` block.
- ``result_effect``: classifies the executed call's result text as
  ``rename`` / ``move_into_directory`` / ``move_or_rename`` so the LLM does not
  have to reverse-engineer the verb from free-form English.
- ``resolved_operation_facts``: when a toolkit exposes a current-working-
  directory and the call is mv/cp, resolves source/destination logical paths,
  predicts the branch (rename vs. move-into-directory) and the expected final
  entry path.
- ``state_context_label``: a short human-readable label
  (e.g. ``"GorillaFileSystem cwd: /workspace/docs"``) injected into the prompt
  header.

All helpers are toolkit-agnostic: cwd-dependent logic activates only when the
state actually has ``runtime_state.toolkits.<X>.current_working_directory``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .state_update_hints import resolve_direct_logical_path


def _toolkit_with_cwd(state: Dict[str, Any], toolkit_hint: Optional[str] = None) -> Optional[str]:
    runtime_toolkits = (
        state.get("runtime_state", {}).get("toolkits", {})
        if isinstance(state, dict)
        else {}
    )
    if not isinstance(runtime_toolkits, dict):
        return None
    if toolkit_hint and isinstance(runtime_toolkits.get(toolkit_hint), dict):
        cwd = runtime_toolkits[toolkit_hint].get("current_working_directory")
        if isinstance(cwd, str) and cwd.strip():
            return toolkit_hint
    for name, entry in runtime_toolkits.items():
        if not isinstance(entry, dict):
            continue
        cwd = entry.get("current_working_directory")
        if isinstance(cwd, str) and cwd.strip():
            return name
    return None


def _logical_path_from_runtime_path(path: str, state: Dict[str, Any]) -> Optional[str]:
    """Convert ``/workspace/docs`` to ``GorillaFileSystem/root/workspace/docs``."""
    if not isinstance(path, str) or not path.startswith("/"):
        return None
    segments = [seg for seg in path.strip("/").split("/") if seg]
    if not segments:
        return None
    if not isinstance(state, dict):
        return None
    for toolkit_name, toolkit_state in state.items():
        if toolkit_name == "runtime_state" or not isinstance(toolkit_state, dict):
            continue
        root_state = toolkit_state.get("root")
        if isinstance(root_state, dict) and segments[0] in root_state:
            return "/".join([toolkit_name, "root"] + segments)
    return None


def _current_logical_cwd(state: Dict[str, Any], toolkit: Optional[str]) -> Optional[str]:
    if not toolkit:
        return None
    cwd = (
        state.get("runtime_state", {})
        .get("toolkits", {})
        .get(toolkit, {})
        .get("current_working_directory")
    )
    if not isinstance(cwd, str) or not cwd.startswith("/"):
        return None
    return _logical_path_from_runtime_path(cwd, state)


def _node_kind(node: Any) -> Optional[str]:
    if isinstance(node, dict):
        node_type = node.get("type")
        if isinstance(node_type, str):
            return node_type
        return "object"
    if isinstance(node, list):
        return "list"
    if node is None:
        return None
    return type(node).__name__


def _find_named_paths(state: Dict[str, Any], toolkit: str, name: str) -> List[str]:
    toolkit_state = state.get(toolkit) if isinstance(state, dict) else None
    if not isinstance(toolkit_state, dict):
        return []
    root = toolkit_state.get("root")
    if not isinstance(root, dict):
        return []

    paths: List[str] = []

    def walk(node: Any, logical_path: str) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "directory":
            contents = node.get("contents")
            if not isinstance(contents, dict):
                return
            for child_name, child in contents.items():
                child_path = f"{logical_path}/{child_name}"
                if child_name == name:
                    paths.append(child_path)
                walk(child, child_path)
            return
        for child_name, child in node.items():
            child_path = f"{logical_path}/{child_name}"
            if child_name == name:
                paths.append(child_path)
            walk(child, child_path)

    walk(root, f"{toolkit}/root")
    return paths


def state_context_label(state: Dict[str, Any]) -> Optional[str]:
    """Return a short human-readable cwd label, or ``None`` when no toolkit
    in the current state exposes a current_working_directory."""
    toolkit = _toolkit_with_cwd(state)
    if not toolkit:
        return None
    cwd = (
        state.get("runtime_state", {})
        .get("toolkits", {})
        .get(toolkit, {})
        .get("current_working_directory")
    )
    if not isinstance(cwd, str) or not cwd.strip():
        return None
    return f"{toolkit} cwd: {cwd}"


def operation_schema_context(
    schema_loader: Any,
    toolkit: Optional[str],
    operation_name: Optional[str],
) -> Dict[str, Any]:
    """Read OpenAPI metadata for ``operation_name`` from the toolkit schema.

    Returns ``operation`` (id/method/path/summary/description) and
    ``request_properties`` (parameter name -> description). Empty dict on miss.
    """
    if not toolkit or not operation_name or schema_loader is None:
        return {}
    try:
        schema_file = schema_loader.find_schema_file(toolkit)
        if not schema_file:
            return {}
        schema = schema_loader.load_schema(schema_file)
    except Exception:
        return {}

    context: Dict[str, Any] = {}
    paths_block = schema.get("paths") if isinstance(schema, dict) else None
    if not isinstance(paths_block, dict):
        return {}
    for path, methods in paths_block.items():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if not isinstance(operation, dict):
                continue
            if operation.get("operationId") != operation_name:
                continue
            context["operation"] = {
                "operation_id": operation_name,
                "method": str(method).upper(),
                "path": path,
                "summary": operation.get("summary", ""),
                "description": operation.get("description", ""),
            }
            request_schema = (
                operation.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema", {})
            )
            if isinstance(request_schema, dict):
                properties = request_schema.get("properties")
                if isinstance(properties, dict):
                    context["request_properties"] = {
                        key: value.get("description", "")
                        for key, value in properties.items()
                        if isinstance(value, dict)
                    }
            return context
    return context


def normalize_result_effect(call: Dict[str, Any]) -> Dict[str, Any]:
    """Classify the call result so the LLM does not have to reverse-engineer
    rename vs. move-into-directory from free-form English."""
    result = call.get("result") if isinstance(call, dict) else None
    if not isinstance(result, dict):
        return {"status": "success", "raw_result": result}
    if result.get("error"):
        return {"status": "error", "raw_result": result}

    text = result.get("result")
    if isinstance(text, str):
        lowered = text.lower()
        if " renamed to " in lowered:
            return {
                "status": "success",
                "effect_kind": "rename",
                "raw_result": result,
                "interpretation": (
                    "The source entry was renamed to the destination name, not moved into a directory."
                ),
            }
        if " moved to " in lowered and "/" in text.rsplit(" moved to ", 1)[-1]:
            return {
                "status": "success",
                "effect_kind": "move_into_directory",
                "raw_result": result,
                "interpretation": "The source entry was moved into a destination directory.",
            }
        if " moved to " in lowered:
            return {
                "status": "success",
                "effect_kind": "move_or_rename",
                "raw_result": result,
                "interpretation": (
                    "Use operation semantics and cwd-local destination existence to decide whether "
                    "this is a move-into-directory or rename."
                ),
            }
    return {"status": "success", "raw_result": result}


_CWD_LOCAL_OPERATIONS = {"mv", "cp"}


def resolved_operation_facts(
    previous_state: Dict[str, Any],
    call: Dict[str, Any],
) -> Dict[str, Any]:
    """For mv/cp on a cwd-bearing toolkit, resolve source/destination logical
    paths and predict the post-call branch."""
    if not isinstance(previous_state, dict) or not isinstance(call, dict):
        return {}
    name = call.get("name")
    args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    toolkit_hint = call.get("toolkit") if isinstance(call.get("toolkit"), str) else None
    toolkit = _toolkit_with_cwd(previous_state, toolkit_hint)
    if not toolkit:
        return {}

    facts: Dict[str, Any] = {}
    current_dir = _current_logical_cwd(previous_state, toolkit)
    if current_dir:
        facts["current_working_directory_logical_path"] = current_dir

    if name in _CWD_LOCAL_OPERATIONS and current_dir:
        source = args.get("source")
        destination = args.get("destination")
        if isinstance(source, str) and isinstance(destination, str):
            source_path = f"{current_dir}/{source}"
            destination_path = f"{current_dir}/{destination}"
            _, source_node = resolve_direct_logical_path(previous_state, source_path)
            _, destination_node = resolve_direct_logical_path(previous_state, destination_path)
            destination_kind = _node_kind(destination_node)
            same_named_paths = _find_named_paths(previous_state, toolkit, destination)
            if destination_kind == "directory":
                branch = "move_or_copy_into_existing_cwd_directory"
                final_entry_path = f"{destination_path}/{source}"
            else:
                branch = "rename_to_cwd_local_destination_name"
                final_entry_path = destination_path
            facts.update(
                {
                    "source_local_path": source_path,
                    "source_exists_in_cwd": source_node is not None,
                    "destination_local_path": destination_path,
                    "destination_exists_in_cwd": destination_node is not None,
                    "destination_kind_in_cwd": destination_kind,
                    "same_named_destination_paths_anywhere": same_named_paths,
                    "expected_branch_from_cwd_local_semantics": branch,
                    "expected_final_entry_path": final_entry_path,
                }
            )
    return facts


def rewrite_persistent_state_paths_to_existing(
    tool_descriptions: Optional[Dict[str, Any]],
    previous_state: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """For each ``persistent_state_shapes[*].path`` that is a bare field name,
    if the field already exists at ``runtime_state.toolkits.<toolkit>/<field>``
    in ``previous_state``, rewrite the path to that fully-qualified location.
    This eliminates the contract ambiguity that otherwise lets the LLM mirror
    the field at toolkit top level (dual-write)."""
    if not isinstance(tool_descriptions, dict) or not isinstance(previous_state, dict):
        return tool_descriptions
    runtime_toolkits = (
        previous_state.get("runtime_state", {}).get("toolkits", {})
        if isinstance(previous_state.get("runtime_state"), dict)
        else {}
    )
    if not isinstance(runtime_toolkits, dict) or not runtime_toolkits:
        return tool_descriptions

    out: Dict[str, Any] = {}
    for op_name, td in tool_descriptions.items():
        if not isinstance(td, dict):
            out[op_name] = td
            continue
        toolkit_name = ""
        toolkit_info = td.get("toolkit")
        if isinstance(toolkit_info, dict):
            tk = toolkit_info.get("name")
            if isinstance(tk, str):
                toolkit_name = tk.strip()
        state_hints = td.get("state_hints")
        if not (toolkit_name and isinstance(state_hints, dict)):
            out[op_name] = td
            continue
        shapes = state_hints.get("persistent_state_shapes")
        if not isinstance(shapes, list) or not shapes:
            out[op_name] = td
            continue
        toolkit_runtime = runtime_toolkits.get(toolkit_name)
        if not isinstance(toolkit_runtime, dict):
            out[op_name] = td
            continue

        new_shapes: List[Dict[str, Any]] = []
        rewrote = False
        for sh in shapes:
            if isinstance(sh, dict):
                path = sh.get("path")
                if (
                    isinstance(path, str)
                    and path
                    and "/" not in path
                    and path in toolkit_runtime
                ):
                    new_sh = dict(sh)
                    new_sh["path"] = f"runtime_state/toolkits/{toolkit_name}/{path}"
                    new_shapes.append(new_sh)
                    rewrote = True
                    continue
            new_shapes.append(sh)
        if not rewrote:
            out[op_name] = td
            continue
        new_state_hints = dict(state_hints)
        new_state_hints["persistent_state_shapes"] = new_shapes
        new_td = dict(td)
        new_td["state_hints"] = new_state_hints
        out[op_name] = new_td
    return out


def enrich_annotated_calls(
    annotated_calls: Sequence[Dict[str, Any]],
    raw_calls: Sequence[Dict[str, Any]],
    previous_state: Optional[Dict[str, Any]],
    schema_loader: Any,
) -> List[Dict[str, Any]]:
    """Attach ``operation_schema`` / ``result_effect`` / ``resolved_operation_facts``
    to each entry. Skips read-only entries for ``resolved_operation_facts`` because
    those facts only matter for state-mutating operations."""
    raw_by_index: Dict[int, Dict[str, Any]] = {}
    for idx, raw in enumerate(raw_calls or [], start=1):
        if isinstance(raw, dict):
            raw_by_index[idx] = raw

    enriched: List[Dict[str, Any]] = []
    for entry in annotated_calls:
        if not isinstance(entry, dict):
            enriched.append(entry)
            continue
        new_entry = dict(entry)
        toolkit = entry.get("toolkit") if isinstance(entry.get("toolkit"), str) else ""
        op_name = entry.get("name") if isinstance(entry.get("name"), str) else ""
        op_schema = operation_schema_context(schema_loader, toolkit, op_name)
        if op_schema:
            new_entry["operation_schema"] = op_schema
        call_index = entry.get("call_index")
        raw_call = raw_by_index.get(call_index) if isinstance(call_index, int) else None
        if isinstance(raw_call, dict):
            effect = normalize_result_effect(raw_call)
            if effect:
                new_entry["result_effect"] = effect
            if entry.get("state_access") != "read" and isinstance(previous_state, dict):
                facts = resolved_operation_facts(previous_state, raw_call)
                if facts:
                    new_entry["resolved_operation_facts"] = facts
        enriched.append(new_entry)
    return enriched

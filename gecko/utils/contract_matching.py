"""Shared helpers for linking read-call evidence to later write calls.

These utilities keep read/write evidence matching consistent across state
update paths.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def tool_call_arguments_compatible(
    write_args: Any,
    read_args: Any,
    *,
    binding: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether a read's arguments make it usable evidence for a write.

    Matching policy:

    1. An explicit ``binding`` mapping ``{write_arg_name: read_arg_name}``
       takes precedence. Every bound pair must be present on both sides and
       must agree; any missing or mismatched binding rejects the read.
    2. Without a binding, the write and read must share at least one
       non-private scalar argument name, and every shared argument must
       agree. A read with zero shared argument names is rejected as
       potentially stale cross-entity evidence (e.g. ``get_flight_cost``
       attached to a later ``book_flight`` after an argument rename).
    """
    if not isinstance(write_args, dict) or not isinstance(read_args, dict):
        return False

    if binding:
        for write_key, read_key in binding.items():
            if not isinstance(write_key, str) or not isinstance(read_key, str):
                return False
            if write_key not in write_args or read_key not in read_args:
                return False
            if write_args.get(write_key) != read_args.get(read_key):
                return False
        return True

    common_keys = {
        key
        for key in write_args.keys() & read_args.keys()
        if isinstance(key, str) and not key.startswith("_")
    }
    if not common_keys:
        return False
    for key in common_keys:
        if write_args.get(key) != read_args.get(key):
            return False
    return True


def read_evidence_binding_for(
    state_hints: Any,
    read_method_name: str,
) -> Optional[Dict[str, str]]:
    """Extract a per-read-method argument binding from a write's state_hints.

    Contract authors can declare an explicit mapping so that renamed or
    aliased arguments can still be linked to a matching read result:

        state_hints["read_evidence_binding"] = {
            "get_flight_cost": {"origin": "from", "destination": "to"},
        }

    The returned mapping uses ``{write_arg_name: read_arg_name}`` form.
    """
    if not isinstance(state_hints, dict) or not read_method_name:
        return None
    bindings = state_hints.get("read_evidence_binding")
    if not isinstance(bindings, dict):
        return None
    entry = bindings.get(read_method_name)
    if not isinstance(entry, dict):
        return None
    clean: Dict[str, str] = {}
    for write_key, read_key in entry.items():
        if isinstance(write_key, str) and isinstance(read_key, str):
            clean[write_key] = read_key
    return clean or None

"""ToolRegistry setup for tau2-bench airline/retail domains."""

from __future__ import annotations

import logging
import json
from typing import Any, Dict, Iterable, List, Optional

from benchmarks.taubench.config import DOMAIN_TOOLS, taubench_mock_openapi_dir, validate_domain
from inference.real_tools import ToolRegistry

logger = logging.getLogger(__name__)


def extract_toolkit_instance(tools: Iterable[Any]) -> Any:
    """Extract the tau2 toolkit instance from tau2 Tool objects."""
    for tool in tools or []:
        func = getattr(tool, "_func", None)
        if func is not None and hasattr(func, "__self__"):
            return func.__self__
        if hasattr(tool, "__self__"):
            return tool.__self__
    raise ValueError("Could not extract a tau2 toolkit instance from tools")


def clone_toolkit_instance(original_toolkit: Any) -> Any:
    """Deep clone a tau2 toolkit so GATS simulation attempts cannot mutate tau2 state."""
    if not hasattr(original_toolkit, "db"):
        raise ValueError(f"Toolkit {type(original_toolkit).__name__} has no db attribute")

    original_db = original_toolkit.db
    if hasattr(original_db, "model_copy"):
        cloned_db = original_db.model_copy(deep=True)
    elif hasattr(original_db, "copy"):
        cloned_db = original_db.copy(deep=True)
    else:
        import copy

        cloned_db = copy.deepcopy(original_db)

    return type(original_toolkit)(cloned_db)


def _tool_name(tool: Any) -> Optional[str]:
    try:
        return tool.get_function_name()
    except Exception:
        pass
    schema = getattr(tool, "openai_tool_schema", None)
    if isinstance(schema, dict):
        fn = schema.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            return fn["name"]
        if isinstance(schema.get("name"), str):
            return schema["name"]
    return None


def _rename_tool(tool: Any, canonical_name: str) -> None:
    schema = getattr(tool, "openai_tool_schema", None)
    if isinstance(schema, dict):
        fn = schema.setdefault("function", {})
        if isinstance(fn, dict):
            fn["name"] = canonical_name


def _load_tool_metadata(domain: str) -> Dict[str, Dict[str, Any]]:
    cfg = DOMAIN_TOOLS[domain]
    path = taubench_mock_openapi_dir() / cfg.mock_openapi_file
    with open(path, "r", encoding="utf-8") as handle:
        spec = json.load(handle)
    tools = (
        (spec.get("info") or {})
        .get("x-default-state", {})
        .get("tools", {})
    )
    return tools if isinstance(tools, dict) else {}


def _attach_metadata_to_tool(tool: Any, metadata: Dict[str, Any]) -> None:
    state_access = metadata.get("state_access")
    if isinstance(state_access, str) and state_access:
        setattr(tool, "bfcl_state_access", state_access)

    effects = metadata.get("state_effects_on_success") or metadata.get("state_effects")
    if isinstance(effects, list):
        clean_effects = [str(effect) for effect in effects if isinstance(effect, str) and effect]
        if clean_effects:
            setattr(tool, "bfcl_state_effects", clean_effects)

    preconditions = metadata.get("preconditions")
    if isinstance(preconditions, list) and preconditions:
        setattr(tool, "bfcl_preconditions", preconditions)


def _attach_tau2_tool_metadata(registry: ToolRegistry, domain: str) -> None:
    """Attach schema-declared read/write semantics to real and mock wrappers."""
    metadata_by_tool = _load_tool_metadata(domain)
    for tool_name, tool in {**registry.real_tools, **registry.mock_tools}.items():
        metadata = metadata_by_tool.get(tool_name)
        if isinstance(metadata, dict):
            _attach_metadata_to_tool(tool, metadata)


def _select_mock_write_tools(registry: ToolRegistry, write_tools: List[str]) -> None:
    """Keep only canonical mock write tools from an OpenAPI-derived registry."""
    selected: Dict[str, Any] = {}
    mock_tools = getattr(registry, "mock_tools", {})

    for canonical_name in write_tools:
        match_name = None
        match_tool = None
        for name, tool in mock_tools.items():
            lower = str(name).lower()
            suffix = f"_{canonical_name.lower()}"
            if lower == canonical_name.lower() or lower.endswith(suffix):
                match_name = name
                match_tool = tool
                break
        if match_tool is None:
            raise ValueError(f"OpenAPI schema did not provide required write tool {canonical_name!r}")
        _rename_tool(match_tool, canonical_name)
        selected[canonical_name] = match_tool
        logger.debug("Mapped tau2 mock tool %s -> %s", match_name, canonical_name)

    mock_tools.clear()
    mock_tools.update(selected)


def create_tau2_tool_registry(
    *,
    original_tools: List[Any],
    domain: str,
    mode: str = "hybrid",
    gecko_url: str = "http://localhost:8000",
) -> ToolRegistry:
    """Create a ToolRegistry for tau-bench GATS simulation planning.

    Modes:
    - ``hybrid``: real read/generic tools plus mock write tools.
    - ``real``: all configured tools use cloned real tau2 toolkit methods.
    - ``mock``: all tools come from the domain mock OpenAPI schema.
    """
    domain = validate_domain(domain)
    mode = (mode or "hybrid").strip().lower()
    if mode not in {"hybrid", "real", "mock"}:
        raise ValueError(f"Unsupported tau-bench GATS mode {mode!r}")

    cfg = DOMAIN_TOOLS[domain]
    original_toolkit = extract_toolkit_instance(original_tools)
    toolkit = clone_toolkit_instance(original_toolkit)
    registry = ToolRegistry()
    real_tool_names: List[str] = []

    def register_real(tool_name: str) -> None:
        if not hasattr(toolkit, tool_name):
            raise AttributeError(
                f"{type(toolkit).__name__} does not expose tau2 tool {tool_name!r}"
            )
        # GATS binds a Gecko SessionContext for each attempt. Read/generic
        # tau-bench calls are flushed as observed state so later mock writes
        # in the same attempt can see facts discovered by real reads.
        registry.register_real_tool(toolkit, tool_name, sync_config=True)
        real_tool_names.append(tool_name)

    if mode == "real":
        for tool_name in cfg.read_tools + cfg.write_tools + cfg.generic_tools:
            register_real(tool_name)

    elif mode == "mock":
        openapi_path = taubench_mock_openapi_dir() / cfg.mock_openapi_file
        if not openapi_path.exists():
            raise FileNotFoundError(f"Missing tau2 OpenAPI schema: {openapi_path}")
        registry.register_mock_tools_from_openapi(str(openapi_path), base_url=gecko_url)

    else:
        for tool_name in cfg.read_tools + cfg.generic_tools:
            register_real(tool_name)

        openapi_path = taubench_mock_openapi_dir() / cfg.mock_openapi_file
        if not openapi_path.exists():
            raise FileNotFoundError(f"Missing tau2 OpenAPI schema: {openapi_path}")
        registry.register_mock_tools_from_openapi(str(openapi_path), base_url=gecko_url)
        _select_mock_write_tools(registry, cfg.write_tools)

    def reset_with_fresh_clone() -> bool:
        nonlocal toolkit
        fresh_toolkit = clone_toolkit_instance(original_toolkit)
        toolkit = fresh_toolkit
        registry.real_tools.clear()
        for tool_name in real_tool_names:
            registry.register_real_tool(fresh_toolkit, tool_name, sync_config=True)
        _attach_tau2_tool_metadata(registry, domain)
        return True

    registry.reset_with_fresh_clone = reset_with_fresh_clone  # type: ignore[attr-defined]
    registry._tau2_domain = domain  # type: ignore[attr-defined]
    registry._tau2_mode = mode  # type: ignore[attr-defined]
    registry._tau2_real_tool_names = list(real_tool_names)  # type: ignore[attr-defined]
    _attach_tau2_tool_metadata(registry, domain)
    return registry

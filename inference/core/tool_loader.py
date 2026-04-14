import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ToolLoader:
    """Pure utility class for tool loading and definition management."""

    @staticmethod
    def load_from_openapi(
        openapi_paths: List[str],
        override_server_url: Optional[str] = None,
        tool_registry: Optional[Any] = None,
    ) -> Tuple[List[Any], Any]:
        """Load tools from OpenAPI specification files.

        Args:
            openapi_paths: Paths to OpenAPI spec files.
            override_server_url: If set, override servers in specs with this URL.
            tool_registry: Optional ToolRegistry for flush-wrapper binding.

        Returns:
            (list_of_FunctionTool, OpenAPIToolkit_instance)
        """
        from camel.toolkits import OpenAPIToolkit, FunctionTool

        all_tools: List[Any] = []
        openapi_toolkit = OpenAPIToolkit()

        if override_server_url:
            openapi_toolkit.set_override_server_url(override_server_url)

        for path in openapi_paths:
            try:
                logger.info(f"Loading OpenAPI spec from {path}")
                with open(path, "r") as f:
                    openapi_json = json.load(f)

                api_name = openapi_json.get("info", {}).get("title", "Unknown API")

                toolkit = openapi_toolkit.generate_openapi_funcs(api_name, openapi_json)
                schemas = openapi_toolkit.openapi_spec_to_openai_schemas(api_name, openapi_json)

                tools = [
                    FunctionTool(func=func, openai_tool_schema=schema)
                    for func, schema in zip(toolkit, schemas)
                ]

                from utils.openapi_toolkit_fix import fix_openapi_tools
                tools = fix_openapi_tools(tools)

                # Wrap mock tools to flush real-tool buffer before execution
                if tool_registry is not None:
                    def create_flush_wrapper(func):
                        def wrapper(*args, **kwargs):
                            tool_registry.flush_session_buffer()
                            return func(*args, **kwargs)
                        return wrapper

                    for tool in tools:
                        tool.func = create_flush_wrapper(tool.func)

                all_tools.extend(tools)
                logger.info(f"Loaded {len(tools)} tools from {api_name}")

            except Exception as e:
                logger.error(f"Failed to load OpenAPI spec from {path}: {e}")
                continue

        logger.info(f"Total {len(all_tools)} tools loaded from {len(openapi_paths)} OpenAPI specs")
        return all_tools, openapi_toolkit

    @staticmethod
    def build_simple_definitions(tools: Optional[List[Any]]) -> List[Dict[str, Any]]:
        """Build tool definitions (name + description + input schema) for judge/checklist.

        Supports CAMEL FunctionTool/RealToolWrapper and tau2 Tool-like objects.
        """
        definitions: List[Dict[str, Any]] = []
        if not tools:
            return definitions

        def _extract_parameters(schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            if not isinstance(schema, dict):
                return None
            params = schema.get("parameters") or schema.get("function", {}).get("parameters")
            if isinstance(params, dict):
                return params
            request_body = (
                schema.get("requestBody")
                or schema.get("function", {}).get("requestBody")
            )
            if isinstance(request_body, dict):
                content = request_body.get("content", {})
                if isinstance(content, dict):
                    app_json = (
                        content.get("application/json", {})
                        or content.get("application/json; charset=utf-8", {})
                    )
                    if isinstance(app_json, dict):
                        body_schema = app_json.get("schema")
                        if isinstance(body_schema, dict):
                            return {
                                "type": "object",
                                "properties": body_schema.get("properties", {}),
                                "required": body_schema.get("required", []),
                                "description": body_schema.get("description", "request body"),
                            }
            return None

        for tool in tools:
            try:
                name: Optional[str] = None
                description: Optional[str] = None
                parameters: Optional[Dict[str, Any]] = None

                schema = getattr(tool, "openai_tool_schema", None)
                if isinstance(schema, dict):
                    name = schema.get("name") or schema.get("function", {}).get("name")
                    description = schema.get("description") or schema.get("function", {}).get("description")
                    parameters = _extract_parameters(schema)

                if not name and hasattr(tool, "name"):
                    name = getattr(tool, "name", None)
                if not description and hasattr(tool, "description"):
                    description = getattr(tool, "description", None)

                if not name:
                    name = (
                        getattr(getattr(tool, "func", None), "__name__", None)
                        or getattr(tool, "__name__", None)
                        or repr(tool)
                    )

                definitions.append(
                    {
                        "name": str(name),
                        "description": str(description) if description is not None else "",
                        **({"parameters": parameters} if parameters else {}),
                    }
                )
            except Exception as e:
                logger.debug(f"[TOOL_LOADER] Skipped tool when building definitions: {e}")

        return definitions

    @staticmethod
    def build_tool_catalog(tool_definitions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Build lightweight tool catalog (name + description + params summary)."""
        catalog: List[Dict[str, str]] = []
        for td in tool_definitions or []:
            if not isinstance(td, dict):
                continue
            name = td.get("name")
            if not isinstance(name, str) or not name:
                continue
            entry: Dict[str, str] = {
                "name": name,
                "description": str(td.get("description", "") or ""),
            }
            # Include parameter names so the judge knows which params are valid.
            params = td.get("parameters")
            if isinstance(params, dict):
                properties = params.get("properties", {})
                required = set(params.get("required", []))
                if isinstance(properties, dict) and properties:
                    parts = []
                    for pname, pschema in properties.items():
                        req = "*" if pname in required else ""
                        desc = ""
                        if isinstance(pschema, dict):
                            desc = pschema.get("description", "")
                        parts.append(f"{pname}{req}: {desc}" if desc else f"{pname}{req}")
                    entry["parameters"] = "; ".join(parts)
            catalog.append(entry)
        return catalog

    @staticmethod
    def tool_name_variants(raw_name: str) -> List[str]:
        """Create normalized name variants for matching tool calls to definitions."""
        name = str(raw_name or "").strip()
        if not name:
            return []
        base = name.split("/")[-1]
        variants = {name, base, name.lower(), base.lower(), base.replace(".", "_"), base.replace("_", ".")}
        return [v for v in variants if v]

    @staticmethod
    def select_involved_definitions(
        tool_definitions: List[Dict[str, Any]],
        tool_calls: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Return detailed schemas only for tools involved in given tool calls."""
        if not tool_definitions:
            return []

        used_variants: set[str] = set()
        for tc in tool_calls or []:
            if not isinstance(tc, dict):
                continue
            call_name = tc.get("function") or tc.get("name") or ""
            used_variants.update(ToolLoader.tool_name_variants(str(call_name)))

        if not used_variants:
            return []

        selected: List[Dict[str, Any]] = []
        for td in tool_definitions:
            if not isinstance(td, dict):
                continue
            def_name = str(td.get("name") or "").strip()
            if not def_name:
                continue
            def_variants = set(ToolLoader.tool_name_variants(def_name))
            if def_variants.intersection(used_variants):
                selected.append(td)
                continue
            def_lower = def_name.lower()
            if any(def_lower.endswith(v.lower()) or v.lower().endswith(def_lower) for v in used_variants):
                selected.append(td)

        return selected

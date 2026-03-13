from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MockServerToolManager:
    """Loads OpenAPI schemas and provides basic tool metadata/execution hooks."""

    def __init__(self, schemas_dir: Optional[str] = None):
        self.schemas_dir = schemas_dir or os.getenv("OPENAPI_SCHEMAS_DIR", "data/openapi")
        self._tools_cache: Dict[str, Dict[str, Any]] = {}
        self._openapi_to_function_mapping: Dict[str, str] = {}
        self._function_to_class_mapping: Dict[str, str] = {}
        self._load_tools()

    def _load_tools(self) -> None:
        schemas_path = Path(self.schemas_dir)
        if not schemas_path.exists():
            logger.warning("Schemas directory not found: %s", self.schemas_dir)
            return

        for schema_file in schemas_path.glob("*.json"):
            try:
                schema_data = json.loads(schema_file.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.error("Failed to load schema %s: %s", schema_file, exc)
                continue

            tool_name = schema_file.stem
            self._tools_cache[tool_name] = schema_data

            for path, path_item in (schema_data.get("paths") or {}).items():
                if not isinstance(path_item, dict):
                    continue
                for method, operation in path_item.items():
                    if method.lower() not in {"get", "post", "put", "delete", "patch", "head", "options"}:
                        continue
                    operation_id = str((operation or {}).get("operationId", "")).strip()
                    if not operation_id:
                        operation_id = path.rstrip("/").split("/")[-1]
                    function_name = operation_id
                    openapi_key = f"{method.upper()} {path}"
                    self._openapi_to_function_mapping[openapi_key] = function_name
                    self._function_to_class_mapping[function_name] = tool_name

    def list_available_tools(self) -> List[str]:
        return list(self._tools_cache.keys())

    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        return self._tools_cache.get(tool_name)

    def list_tool_functions(self, tool_name: str) -> List[str]:
        schema = self.get_tool_schema(tool_name)
        if not schema:
            return []
        functions: List[str] = []
        for path, path_item in (schema.get("paths") or {}).items():
            if not isinstance(path_item, dict):
                continue
            for method, operation in path_item.items():
                if method.lower() not in {"get", "post", "put", "delete", "patch", "head", "options"}:
                    continue
                operation_id = str((operation or {}).get("operationId", "")).strip()
                functions.append(operation_id or path.rstrip("/").split("/")[-1])
        return functions

    def get_function_schema(self, tool_name: str, function_name: str) -> Optional[Dict[str, Any]]:
        schema = self.get_tool_schema(tool_name)
        if not schema:
            return None
        for path_item in (schema.get("paths") or {}).values():
            if not isinstance(path_item, dict):
                continue
            for method, operation in path_item.items():
                if method.lower() not in {"get", "post", "put", "delete", "patch", "head", "options"}:
                    continue
                operation_id = str((operation or {}).get("operationId", "")).strip()
                op_name = operation_id or ""
                if op_name == function_name:
                    return operation
        return None

    def execute_function(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute a function in mock mode.

        Supports both call styles:
        1) execute_function(tool_name, function_name, params)
        2) execute_function(function_name=..., state=..., **arguments)
        """
        if len(args) == 3 and isinstance(args[0], str):
            tool_name, function_name, params = args
            params = params if isinstance(params, dict) else {"value": params}
        else:
            function_name = kwargs.pop("function_name", "")
            kwargs.pop("state", None)
            params = kwargs
            tool_name = kwargs.pop("tool_name", None) or self._function_to_class_mapping.get(function_name, "")

        if not function_name:
            raise ValueError("Missing function_name")

        return {
            "success": True,
            "result": f"Mock execution of {tool_name}.{function_name}" if tool_name else f"Mock execution of {function_name}",
            "params": params,
        }

    def validate_params(self, tool_name: str, function_name: str, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        schema = self.get_function_schema(tool_name, function_name)
        if not schema:
            return False, f"Function {function_name} not found in tool {tool_name}"
        # Placeholder validation hook. Full OpenAPI validation is handled in request validator.
        return True, None

    def reset_tool_instance(self, tool_name: str) -> None:
        """Compatibility no-op for API reset endpoint."""
        if tool_name not in self._tools_cache:
            raise ValueError(f"Tool {tool_name} not found")

    def reset_all_instances(self) -> None:
        """Compatibility no-op for API reset endpoint."""
        return


_tool_manager_instance: Optional[MockServerToolManager] = None


def get_tool_manager(schemas_dir: Optional[str] = None) -> MockServerToolManager:
    """Return singleton tool manager instance."""
    global _tool_manager_instance
    if _tool_manager_instance is None or schemas_dir is not None:
        _tool_manager_instance = MockServerToolManager(schemas_dir)
    return _tool_manager_instance


__all__ = ["MockServerToolManager", "get_tool_manager"]

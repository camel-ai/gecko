import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from openapi_spec_validator import validate_spec


class SchemaLoader:
    def __init__(self, schemas_dir: Union[str, List[str]]):
        # Support single directory or a list of directories
        if isinstance(schemas_dir, list):
            self.schema_dirs: List[str] = [d for d in schemas_dir if isinstance(d, str) and d]
        else:
            self.schema_dirs = [schemas_dir]
        self.schemas_cache: Dict[str, Dict[str, Any]] = {}

    def find_schema_file(self, api_name: str) -> Optional[str]:
        """Find the schema file for the given API name.
        Supports .json/.yaml/.yml, case-insensitive filenames, nested '<api_name>/openapi.json',
        and recursive search in subdirectories.
        """
        for base in self.schema_dirs:
            # Direct candidates in base directory
            candidates = [
                os.path.join(base, f"{api_name}.json"),
                os.path.join(base, f"{api_name}.yaml"),
                os.path.join(base, f"{api_name}.yml"),
                os.path.join(base, api_name, "openapi.json"),
                os.path.join(base, api_name, "openapi.yaml"),
                os.path.join(base, api_name, "openapi.yml"),
            ]
            for path in candidates:
                if os.path.exists(path):
                    return path
            
            # Case-insensitive scan in base (flat files)
            try:
                for entry in os.listdir(base):
                    lower = entry.lower()
                    if lower in {f"{api_name}.json", f"{api_name}.yaml", f"{api_name}.yml"}:
                        p = os.path.join(base, entry)
                        if os.path.exists(p):
                            return p
            except Exception:
                pass
            
            # Recursive search in subdirectories
            result = self._recursive_search(base, api_name)
            if result:
                return result
                
        return None

    def iter_schema_files(self) -> List[str]:
        """Return all OpenAPI schema files under the configured schema directories."""
        schema_files: List[str] = []
        seen = set()
        for base in self.schema_dirs:
            try:
                for root, _, files in os.walk(base):
                    for filename in files:
                        if not filename.lower().endswith((".json", ".yaml", ".yml")):
                            continue
                        path = os.path.join(root, filename)
                        if path in seen:
                            continue
                        seen.add(path)
                        schema_files.append(path)
            except Exception:
                continue
        return schema_files

    def find_tool_entry(
        self,
        operation_id: str,
        toolkit_name: Optional[str] = None,
    ) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
        """Find x-default-state tool metadata by operation id.

        Returns ``(schema_path, schema, tool_entry)`` when the active schemas
        declare metadata for the operation.
        """
        op_id = (operation_id or "").strip()
        if not op_id:
            return None

        candidate_paths: List[str] = []
        if toolkit_name:
            schema_file = self.find_schema_file(toolkit_name)
            if schema_file:
                candidate_paths.append(schema_file)
        for schema_file in self.iter_schema_files():
            if schema_file not in candidate_paths:
                candidate_paths.append(schema_file)

        for schema_file in candidate_paths:
            try:
                schema = self.load_schema(schema_file)
            except Exception:
                continue
            tools_block = schema.get("info", {}).get("x-default-state", {}).get("tools", {})
            if not isinstance(tools_block, dict):
                continue
            tool_entry = tools_block.get(op_id)
            if isinstance(tool_entry, dict) and tool_entry:
                return schema_file, schema, tool_entry
        return None

    def _recursive_search(self, directory: str, api_name: str) -> Optional[str]:
        """Recursively search for schema files in subdirectories."""
        try:
            for entry in os.listdir(directory):
                entry_path = os.path.join(directory, entry)
                if os.path.isdir(entry_path):
                    # Check if this subdirectory contains the schema
                    candidates = [
                        os.path.join(entry_path, f"{api_name}.json"),
                        os.path.join(entry_path, f"{api_name}.yaml"),
                        os.path.join(entry_path, f"{api_name}.yml"),
                    ]
                    for candidate in candidates:
                        if os.path.exists(candidate):
                            return candidate
                    
                    # Case-insensitive search in this subdirectory
                    try:
                        for sub_entry in os.listdir(entry_path):
                            lower = sub_entry.lower()
                            if lower in {f"{api_name}.json", f"{api_name}.yaml", f"{api_name}.yml"}:
                                p = os.path.join(entry_path, sub_entry)
                                if os.path.exists(p):
                                    return p
                    except Exception:
                        pass
                    
                    # Continue recursive search
                    result = self._recursive_search(entry_path, api_name)
                    if result:
                        return result
        except Exception:
            pass
        return None

    def load_schema(self, schema_path: str) -> Dict[str, Any]:
        """Load and validate OpenAPI schema."""
        if schema_path in self.schemas_cache:
            return self.schemas_cache[schema_path]

        with open(schema_path, 'r', encoding='utf-8') as f:
            if schema_path.endswith('.yaml') or schema_path.endswith('.yml'):
                schema = yaml.safe_load(f)
            else:
                schema = json.load(f)
        try:
            from benchmarks.bfcl.source_schema import (
                align_bfcl_single_turn_openapi_schema,
            )

            align_bfcl_single_turn_openapi_schema(schema_path, schema)
        except Exception:
            pass

        # Validate the schema
        validate_spec(schema)
        _augment_error_branches(schema)
        self.schemas_cache[schema_path] = schema
        return schema


_ERROR_BRANCH: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "error": {
            "type": "string",
            "description": "Reason the operation failed (use when state validation forbids the requested operation).",
        }
    },
    "required": ["error"],
    "additionalProperties": False,
}


def _has_error_branch(schema: Any) -> bool:
    """True iff any (possibly nested) branch declares an `error` property at top level."""
    if not isinstance(schema, dict):
        return False
    for combinator in ("oneOf", "anyOf", "allOf"):
        nested = schema.get(combinator)
        if isinstance(nested, list):
            return any(_has_error_branch(b) for b in nested)
    props = schema.get("properties")
    return isinstance(props, dict) and "error" in props


def _inject_error_branch(response_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Append an error variant to a response schema that lacks one.

    Many BFCL OpenAPI schemas only enumerate success branches in their 200
    response. When state validation mandates failure, the response model has
    no schema-legal way to express it and tends to either fabricate a success
    or loop until max_tokens. Adding a uniform `{error: <string>}` variant
    gives the model a clean exit.
    """
    if not isinstance(response_schema, dict) or _has_error_branch(response_schema):
        return response_schema
    if isinstance(response_schema.get("oneOf"), list):
        return {**response_schema, "oneOf": list(response_schema["oneOf"]) + [_ERROR_BRANCH]}
    if isinstance(response_schema.get("anyOf"), list):
        return {**response_schema, "anyOf": list(response_schema["anyOf"]) + [_ERROR_BRANCH]}
    return {"oneOf": [response_schema, _ERROR_BRANCH]}


def _augment_error_branches(schema: Dict[str, Any]) -> None:
    """In-place: ensure every 200/201 JSON response schema has an error branch."""
    if not isinstance(schema, dict):
        return
    paths = schema.get("paths")
    if not isinstance(paths, dict):
        return
    for methods in paths.values():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if not isinstance(operation, dict):
                continue
            if method.startswith("x-") or method in {"parameters", "servers", "summary", "description"}:
                continue
            responses = operation.get("responses")
            if not isinstance(responses, dict):
                continue
            for status_code in ("200", "201"):
                response = responses.get(status_code)
                if not isinstance(response, dict):
                    continue
                content = response.get("content")
                if not isinstance(content, dict):
                    continue
                json_content = content.get("application/json")
                if not isinstance(json_content, dict):
                    continue
                resp_schema = json_content.get("schema")
                if isinstance(resp_schema, dict):
                    json_content["schema"] = _inject_error_branch(resp_schema)

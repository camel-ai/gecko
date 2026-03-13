from __future__ import annotations

import ast
import json
import logging
import re
import tempfile
import textwrap
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FunctionInfo:
    name: str
    docstring: str
    parameters: List[Dict[str, Any]]
    source_code: str
    return_type: Optional[str] = None
    decorators: List[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    name: str
    methods: List[FunctionInfo]
    docstring: str
    source_code: str
    init_method: Optional[str] = None
    class_variables: List[str] = field(default_factory=list)
    instance_variables: Dict[str, str] = field(default_factory=dict)
    imports: List[str] = field(default_factory=list)
    constants: Dict[str, Any] = field(default_factory=dict)
    helper_functions: List[FunctionInfo] = field(default_factory=list)


@dataclass
class ExtractedData:
    """Extracted data from Python file."""

    default_state: Dict[str, Any] = field(default_factory=dict)
    preset_databases: Dict[str, Any] = field(default_factory=dict)
    constants: Dict[str, Any] = field(default_factory=dict)
    imports: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    data_models_content: str = ""


@dataclass
class BatchEndpointResult:
    """Result from a batch endpoint generation call."""

    method_name: str
    endpoint: Dict[str, Any]
    source: str
    state_updates: Dict[str, Any] = field(default_factory=dict)


def clean_json_response(content: str) -> str:
    """Remove JSON markdown fences from an LLM response."""
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def unwrap_http_method(endpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Unwrap accidental HTTP method wrapper from LLM output."""
    if isinstance(endpoint, dict):
        for method in ("get", "post", "put", "delete", "patch"):
            if method in endpoint and len(endpoint) == 1:
                return endpoint[method]
    return endpoint


def sanitize_openapi_schema(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize generated OpenAPI schemas before they are written to disk."""

    def _sanitize(node: Any) -> Any:
        if isinstance(node, list):
            return [_sanitize(item) for item in node]
        if not isinstance(node, dict):
            return node

        fixed = {key: _sanitize(value) for key, value in node.items()}

        if fixed.get("type") == "array" and "items" not in fixed:
            fixed["items"] = {
                "anyOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                    {"type": "null"},
                    {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                ]
            }

        if fixed.get("type") == "object":
            props = fixed.get("properties")
            if isinstance(props, dict) and "additionalProperties" not in fixed:
                fixed["additionalProperties"] = False

        if isinstance(fixed.get("additionalProperties"), dict):
            fixed["additionalProperties"] = _sanitize(fixed["additionalProperties"])

        return fixed

    return _sanitize(spec)


def build_data_models_info(state_data: Dict[str, Any]) -> str:
    """Build compact model summary used in endpoint prompts."""
    if "data_models" not in state_data or not state_data["data_models"]:
        return ""
    models_detail = []
    for model_name, model_info in list(state_data.get("data_models", {}).items())[:8]:
        if not isinstance(model_info, dict):
            continue
        if "fields" in model_info:
            fields_str = []
            for field_name, field_data in list(model_info.get("fields", {}).items())[:5]:
                if isinstance(field_data, dict):
                    field_type = field_data.get("type", "Any")
                    field_desc = field_data.get("description", "")
                    fields_str.append(f"    {field_name}: {field_type} - {field_desc}")
            if fields_str:
                models_detail.append(f"  {model_name}:\n" + "\n".join(fields_str))
        elif "type" in model_info and model_info["type"] == "alias":
            models_detail.append(f"  {model_name}: {model_info.get('definition', 'Type alias')}")
    if models_detail:
        return (
            "\n\nAvailable Data Models (use these for PRECISE response schemas):\n"
            + "\n".join(models_detail)
        )
    return ""


def _source_segment(lines: List[str], node: ast.AST) -> str:
    """Extract full source segment for an AST node."""
    start = max(0, getattr(node, "lineno", 1) - 1)
    end_lineno = getattr(node, "end_lineno", getattr(node, "lineno", 1))
    end = end_lineno
    return "\n".join(lines[start:end])


class EnhancedPythonParser:
    """Enhanced parser for Python files with better extraction capabilities."""

    def parse_multiple_files(
        self,
        main_file: str,
        additional_files: Optional[List[str]] = None,
        preferred_class_name: Optional[str] = None,
    ) -> Tuple[Optional[ClassInfo], ExtractedData]:
        """Parse main file and additional files for comprehensive extraction."""
        main_class, main_data = self.parse_file(main_file, preferred_class_name=preferred_class_name)

        if additional_files:
            data_model_contents = []
            for file_path in additional_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as source_file:
                        content = source_file.read()
                        data_model_contents.append(f"# File: {file_path}\n{content}")
                        logger.info(f"Read {len(content)} characters from {file_path}")
                except Exception as exc:
                    raise RuntimeError(f"Failed to read additional file {file_path}: {exc}") from exc

            if data_model_contents:
                main_data.data_models_content = "\n\n".join(data_model_contents)

        return main_class, main_data

    def parse_file(
        self, file_path: str, preferred_class_name: Optional[str] = None
    ) -> Tuple[Optional[ClassInfo], ExtractedData]:
        """Parse Python file and extract comprehensive information."""
        try:
            with open(file_path, "r", encoding="utf-8") as source_file:
                content = source_file.read()
        except Exception as exc:
            logger.error(f"Error reading file {file_path}: {exc}")
            return None, ExtractedData()

        try:
            tree = ast.parse(content)
        except SyntaxError as exc:
            logger.error(f"Syntax error in {file_path}: {exc}")
            return None, ExtractedData()

        extracted_data = self._extract_all_data(tree, content)

        main_class = None
        class_nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
        ]
        if preferred_class_name:
            target = next((node for node in class_nodes if node.name == preferred_class_name), None)
            if target is None:
                logger.error(
                    "Preferred class '%s' not found in %s. Available classes: %s",
                    preferred_class_name,
                    file_path,
                    [node.name for node in class_nodes],
                )
                return None, extracted_data
            main_class = self._parse_class(target, content, extracted_data)
        elif class_nodes:
            main_class = self._parse_class(class_nodes[0], content, extracted_data)

        return main_class, extracted_data

    def _extract_all_data(self, tree: ast.AST, content: str) -> ExtractedData:
        """Extract preset data, constants, and imports."""
        extracted = ExtractedData()
        lines = content.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    extracted.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    import_name = alias.name
                    extracted.imports.append(f"{module}.{import_name}")
                    if "EXTENSION" in import_name or "extension" in import_name:
                        extracted.preset_databases[import_name] = f"Imported from {module}"

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        func_name = item.name
                        for child in item.body:
                            if isinstance(child, ast.Assign):
                                for target in child.targets:
                                    if isinstance(target, ast.Name) and isinstance(
                                        child.value, (ast.Dict, ast.List, ast.Set)
                                    ):
                                        try:
                                            assignment_source = _source_segment(lines, child)
                                            key = f"{func_name}_{target.id}"
                                            extracted.preset_databases[key] = assignment_source
                                        except Exception:
                                            pass
                            elif isinstance(child, ast.AnnAssign):
                                if isinstance(child.target, ast.Name) and isinstance(
                                    child.value, (ast.Dict, ast.List, ast.Set, ast.Tuple)
                                ):
                                    try:
                                        assignment_source = _source_segment(lines, child)
                                        key = f"{func_name}_{child.target.id}"
                                        extracted.preset_databases[key] = assignment_source
                                    except Exception:
                                        pass
                            elif isinstance(child, ast.Return) and isinstance(
                                child.value, (ast.List, ast.Dict, ast.Set)
                            ):
                                try:
                                    return_source = _source_segment(lines, child)
                                    key = f"{func_name}_returns"
                                    if key not in extracted.preset_databases:
                                        extracted.preset_databases[key] = return_source
                                except Exception:
                                    pass

        for node in tree.body:
            target_names: List[str] = []
            value_node: Optional[ast.AST] = None
            if isinstance(node, ast.Assign):
                target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                value_node = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target_names = [node.target.id]
                value_node = node.value

            if not target_names:
                continue

            try:
                assignment_source = _source_segment(lines, node)
                for var_name in target_names:
                    if "DEFAULT_STATE" in var_name:
                        extracted.default_state[var_name] = assignment_source
                    elif any(
                        pattern in var_name.lower()
                        for pattern in [
                            "cost",
                            "rate",
                            "price",
                            "extension",
                            "data",
                            "config",
                            "database",
                            "record",
                            "lookup",
                            "mapping",
                            "airport",
                            "flight",
                        ]
                    ):
                        extracted.preset_databases[var_name] = assignment_source
                    elif var_name.isupper():
                        extracted.constants[var_name] = assignment_source
            except Exception:
                pass
        return extracted

    def _parse_class(
        self, node: ast.ClassDef, content: str, extracted_data: ExtractedData
    ) -> ClassInfo:
        methods: List[FunctionInfo] = []
        helper_functions: List[FunctionInfo] = []
        class_variables: List[str] = []
        instance_variables: Dict[str, str] = {}
        init_method = None

        lines = content.splitlines()
        class_source = "\n".join(lines[node.lineno - 1 : (node.end_lineno or node.lineno)])

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                func_info = self._parse_function(item, content)
                if func_info:
                    if item.name == "__init__":
                        init_method = func_info.source_code
                        for child in ast.walk(item):
                            if isinstance(child, ast.Assign):
                                for target in child.targets:
                                    if (
                                        isinstance(target, ast.Attribute)
                                        and isinstance(target.value, ast.Name)
                                        and target.value.id == "self"
                                    ):
                                        var_type = "Any"
                                        if hasattr(target, "annotation"):
                                            var_type = (
                                                ast.unparse(target.annotation)
                                                if hasattr(ast, "unparse")
                                                else "Any"
                                            )
                                        instance_variables[target.attr] = var_type
                            elif isinstance(child, ast.AnnAssign):
                                target = child.target
                                if (
                                    isinstance(target, ast.Attribute)
                                    and isinstance(target.value, ast.Name)
                                    and target.value.id == "self"
                                ):
                                    var_type = "Any"
                                    if child.annotation:
                                        var_type = (
                                            ast.unparse(child.annotation)
                                            if hasattr(ast, "unparse")
                                            else "Any"
                                        )
                                    instance_variables[target.attr] = var_type
                    elif item.name.startswith("_"):
                        helper_functions.append(func_info)
                    else:
                        methods.append(func_info)
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                class_variables.append(item.target.id)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_variables.append(target.id)

        return ClassInfo(
            name=node.name,
            methods=methods,
            docstring=ast.get_docstring(node) or "",
            source_code=class_source,
            init_method=init_method,
            class_variables=class_variables,
            instance_variables=instance_variables,
            imports=extracted_data.imports,
            constants=extracted_data.constants,
            helper_functions=helper_functions,
        )

    def _parse_function(self, node: ast.FunctionDef, content: str) -> Optional[FunctionInfo]:
        docstring = ast.get_docstring(node) or ""
        parameters = []
        decorators = []

        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                decorators.append(decorator.func.id)

        lines = content.splitlines()
        func_source = "\n".join(lines[node.lineno - 1 : (node.end_lineno or node.lineno)])

        for arg in node.args.args:
            if arg.arg != "self":
                param_info: Dict[str, Any] = {"name": arg.arg, "required": True}
                if arg.annotation:
                    if hasattr(ast, "unparse"):
                        param_info["type_hint"] = ast.unparse(arg.annotation)
                    elif isinstance(arg.annotation, ast.Name):
                        param_info["type_hint"] = arg.annotation.id
                    else:
                        param_info["type_hint"] = "Any"
                else:
                    param_info["type_hint"] = "Any"
                parameters.append(param_info)

        defaults = node.args.defaults
        if defaults:
            num_no_default = len(node.args.args) - len(defaults)
            self_offset = 1 if node.args.args and node.args.args[0].arg == "self" else 0
            for index, param in enumerate(parameters):
                arg_index = index + self_offset
                if arg_index >= num_no_default:
                    param["required"] = False

        return_type = None
        if node.returns:
            if hasattr(ast, "unparse"):
                return_type = ast.unparse(node.returns)
            elif isinstance(node.returns, ast.Name):
                return_type = node.returns.id

        return FunctionInfo(
            name=node.name,
            docstring=docstring,
            parameters=parameters,
            source_code=func_source,
            return_type=return_type,
            decorators=decorators,
        )


def _safe_unparse(node: ast.AST) -> str:
    if hasattr(ast, "unparse"):
        try:
            return ast.unparse(node)
        except Exception:
            return str(node)
    return str(node)


def _json_key(key: Any) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, tuple) and all(isinstance(x, (str, int, float, bool)) for x in key):
        return "|".join(str(x) for x in key)
    try:
        return json.dumps(_to_jsonable(key), ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(key)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {_json_key(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, set):
        normalized = [_to_jsonable(v) for v in value]
        try:
            return sorted(normalized, key=lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))
        except Exception:
            return normalized
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _literal_eval_node(node: Optional[ast.AST]) -> Any:
    if node is None:
        return None
    try:
        return _to_jsonable(ast.literal_eval(node))
    except Exception:
        return None


def _extract_assigned_value(source: str, var_name: str) -> Any:
    try:
        tree = ast.parse(source)
    except Exception:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == var_name for t in node.targets):
                return _literal_eval_node(node.value)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == var_name:
                return _literal_eval_node(node.value)
    return None


def _parse_method_node(source_code: str) -> Optional[ast.FunctionDef]:
    dedented = textwrap.dedent(source_code or "")
    try:
        tree = ast.parse(dedented)
    except Exception:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def _collect_self_root(expr: ast.AST, roots: Set[str]) -> None:
    if isinstance(expr, ast.Attribute):
        if isinstance(expr.value, ast.Name) and expr.value.id == "self":
            roots.add(expr.attr)
            return
        _collect_self_root(expr.value, roots)
        return
    if isinstance(expr, ast.Subscript):
        _collect_self_root(expr.value, roots)
        return
    if isinstance(expr, (ast.Tuple, ast.List)):
        for elt in expr.elts:
            _collect_self_root(elt, roots)
        return
    if isinstance(expr, ast.Starred):
        _collect_self_root(expr.value, roots)


def _extract_static_data_from_method(method: FunctionInfo) -> Dict[str, Any]:
    func_node = _parse_method_node(method.source_code)
    if not func_node:
        return {}

    def _should_keep(name: str, literal: Any) -> bool:
        if name == "__return__":
            return False
        if isinstance(literal, dict):
            return True
        if isinstance(literal, list):
            return len(literal) > 0
        if isinstance(literal, tuple):
            return len(literal) > 0
        return False

    static_data: Dict[str, Any] = {}
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign):
            literal = _literal_eval_node(node.value)
            if literal is None:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if _should_keep(target.id, literal):
                        static_data[target.id] = literal
        elif isinstance(node, ast.AnnAssign):
            literal = _literal_eval_node(node.value)
            if literal is None:
                continue
            if isinstance(node.target, ast.Name):
                if _should_keep(node.target.id, literal):
                    static_data[node.target.id] = literal
    # Capture direct constant return payloads (e.g., list_all_* methods returning
    # literal lists) so callers can inherit deterministic constraints.
    direct_returns: List[Any] = []
    for stmt in func_node.body:
        if not isinstance(stmt, ast.Return):
            continue
        literal = _literal_eval_node(stmt.value)
        if isinstance(literal, (dict, list, tuple)) and literal:
            direct_returns.append(literal)
    if len(direct_returns) == 1:
        static_data["return_value"] = direct_returns[0]
    elif len(direct_returns) > 1:
        static_data["return_values"] = direct_returns[:3]
    return static_data


def _infer_called_static_aliases(called_method_name: str) -> List[str]:
    """Infer generic alias names for called-method return literals."""
    if not isinstance(called_method_name, str):
        return []
    name = called_method_name.strip()
    if not name:
        return []
    aliases: List[str] = []
    if name.startswith("list_all_") and len(name) > len("list_all_"):
        aliases.append(f"all_{name[len('list_all_'):]}")
    if name.startswith("list_") and len(name) > len("list_"):
        aliases.append(name[len("list_"):])
    if name.startswith("get_all_") and len(name) > len("get_all_"):
        aliases.append(f"all_{name[len('get_all_'):]}")
    return aliases


def _extract_return_signature(value: Optional[ast.AST]) -> Optional[Dict[str, Any]]:
    if not isinstance(value, ast.Dict):
        return None
    result: Dict[str, Any] = {}
    for key_node, val_node in zip(value.keys, value.values):
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            continue
        key = key_node.value
        if key == "error" or key.endswith("_status") or key in {"status", "message"}:
            literal = _literal_eval_node(val_node)
            result[key] = literal if literal is not None else _safe_unparse(val_node)
    return result or None


def _extract_validation_rules_from_method(method: FunctionInfo) -> List[Dict[str, Any]]:
    func_node = _parse_method_node(method.source_code)
    if not func_node:
        return []

    rules: List[Dict[str, Any]] = []
    seen = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.If):
            condition = _safe_unparse(node.test)
            for body_node in node.body:
                if isinstance(body_node, ast.Return):
                    signature = _extract_return_signature(body_node.value)
                    if signature:
                        rule = {"condition": condition, "result": signature}
                        marker = json.dumps(rule, ensure_ascii=False, sort_keys=True)
                        if marker not in seen:
                            seen.add(marker)
                            rules.append(rule)
        elif isinstance(node, ast.Raise):
            if isinstance(node.exc, ast.Call):
                exc_name = _safe_unparse(node.exc.func)
                if exc_name.endswith("ValueError"):
                    message = ""
                    if node.exc.args:
                        literal = _literal_eval_node(node.exc.args[0])
                        message = literal if isinstance(literal, str) else _safe_unparse(node.exc.args[0])
                    rule = {"raises": "ValueError", "message": message}
                    marker = json.dumps(rule, ensure_ascii=False, sort_keys=True)
                    if marker not in seen:
                        seen.add(marker)
                        rules.append(rule)
    return rules


def _extract_state_io_and_calls(method: FunctionInfo) -> Tuple[List[str], List[str], List[str]]:
    func_node = _parse_method_node(method.source_code)
    if not func_node:
        return [], [], []

    reads: Set[str] = set()
    writes: Set[str] = set()
    calls: Set[str] = set()

    for node in ast.walk(func_node):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                if isinstance(node.ctx, ast.Load):
                    reads.add(node.attr)
                elif isinstance(node.ctx, ast.Store):
                    writes.add(node.attr)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "self":
                    calls.add(node.func.attr)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                _collect_self_root(target, writes)
        elif isinstance(node, ast.AnnAssign):
            _collect_self_root(node.target, writes)
        elif isinstance(node, ast.AugAssign):
            _collect_self_root(node.target, writes)

    return sorted(reads), sorted(writes), sorted(calls)


def _self_root_name(expr: ast.AST) -> Optional[str]:
    """Return the root self attribute name (e.g., self.tweets[...] -> tweets)."""
    if isinstance(expr, ast.Attribute):
        if isinstance(expr.value, ast.Name) and expr.value.id == "self":
            return expr.attr
        return _self_root_name(expr.value)
    if isinstance(expr, ast.Subscript):
        return _self_root_name(expr.value)
    return None


def _self_access_path(expr: ast.AST) -> Optional[str]:
    """Return self access path with subscripts (e.g., self.comments[tweet_id])."""
    if isinstance(expr, ast.Attribute):
        if isinstance(expr.value, ast.Name) and expr.value.id == "self":
            return expr.attr
        parent = _self_access_path(expr.value)
        if parent:
            return f"{parent}.{expr.attr}"
        return None
    if isinstance(expr, ast.Subscript):
        parent = _self_access_path(expr.value)
        if not parent:
            return None
        key_expr = _safe_unparse(expr.slice).strip()
        if key_expr:
            return f"{parent}[{key_expr}]"
        return parent
    return None


def _is_same_self_attr(expr: ast.AST, attr_name: str) -> bool:
    return (
        isinstance(expr, ast.Attribute)
        and isinstance(expr.value, ast.Name)
        and expr.value.id == "self"
        and expr.attr == attr_name
    )


def _extract_state_effects_from_method(method: FunctionInfo) -> List[str]:
    """Extract concise state transition semantics from a method body."""
    func_node = _parse_method_node(method.source_code)
    if not func_node:
        return []

    effects: List[str] = []
    seen: Set[str] = set()
    local_dict_literals: Dict[str, ast.Dict] = {}

    def _add_effect(text: str) -> None:
        normalized = " ".join((text or "").split()).strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        effects.append(normalized)

    def _dict_literal_summary(dict_node: ast.Dict) -> Optional[str]:
        parts: List[str] = []
        for key_node, val_node in zip(dict_node.keys, dict_node.values):
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                continue
            key = key_node.value
            value_expr = _safe_unparse(val_node).strip()
            if not value_expr:
                value_expr = "value"
            parts.append(f"{key}: {value_expr}")
        if not parts:
            return None
        if len(parts) > 8:
            parts = parts[:8] + ["..."]
        return "{ " + ", ".join(parts) + " }"

    def _value_semantic_hint(value_node: Optional[ast.AST]) -> Optional[str]:
        if value_node is None:
            return None
        if isinstance(value_node, ast.Dict):
            summary = _dict_literal_summary(value_node)
            if summary:
                return f"using object fields {summary}"
            return "using a constructed object payload"
        if isinstance(value_node, ast.Name):
            dict_node = local_dict_literals.get(value_node.id)
            if dict_node is not None:
                summary = _dict_literal_summary(dict_node)
                if summary:
                    return f"using `{value_node.id}` fields {summary}"
                return f"using constructed object `{value_node.id}`"
            return f"using `{value_node.id}`"
        if isinstance(value_node, ast.Call):
            return f"using {_safe_unparse(value_node).strip()}"
        return None

    def _assignment_effect(target: ast.AST, value: Optional[ast.AST]) -> Optional[str]:
        root = _self_root_name(target)
        if not root:
            return None

        if isinstance(target, ast.Subscript):
            key_expr = _safe_unparse(target.slice).strip()
            semantic_hint = _value_semantic_hint(value)
            if key_expr:
                if semantic_hint:
                    return f"on success, set `{root}[{key_expr}]` {semantic_hint}."
                return f"on success, set `{root}[{key_expr}]` to the new value."
            return f"on success, update an entry in `{root}`."

        if isinstance(target, ast.Attribute):
            if isinstance(value, ast.BinOp) and _is_same_self_attr(value.left, root):
                delta_text = _safe_unparse(value.right).strip()
                if isinstance(value.op, ast.Add):
                    if delta_text == "1":
                        return f"on success, increment `{root}` by 1."
                    return f"on success, increase `{root}` by {delta_text}."
                if isinstance(value.op, ast.Sub):
                    if delta_text == "1":
                        return f"on success, decrement `{root}` by 1."
                    return f"on success, decrease `{root}` by {delta_text}."
            return f"on success, set `{root}`."

        return None

    def _augassign_effect(target: ast.AST, op: ast.operator, value: ast.AST) -> Optional[str]:
        root = _self_root_name(target)
        if not root:
            return None

        amount = _safe_unparse(value).strip()
        if isinstance(op, ast.Add):
            if amount == "1":
                return f"on success, increment `{root}` by 1."
            return f"on success, increase `{root}` by {amount}."
        if isinstance(op, ast.Sub):
            if amount == "1":
                return f"on success, decrement `{root}` by 1."
            return f"on success, decrease `{root}` by {amount}."
        if isinstance(op, ast.Mult):
            return f"on success, multiply `{root}` by {amount}."
        if isinstance(op, ast.Div):
            return f"on success, divide `{root}` by {amount}."
        return f"on success, update `{root}`."

    def _call_value_hint(call_node: ast.Call) -> Optional[str]:
        """Build a compact payload hint from first positional argument."""
        if not call_node.args:
            return None
        return _value_semantic_hint(call_node.args[0])

    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    local_dict_literals[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and isinstance(node.value, ast.Dict):
            local_dict_literals[node.target.id] = node.value

    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                effect = _assignment_effect(target, node.value)
                if effect:
                    _add_effect(effect)
        elif isinstance(node, ast.AnnAssign):
            effect = _assignment_effect(node.target, node.value)
            if effect:
                _add_effect(effect)
        elif isinstance(node, ast.AugAssign):
            effect = _augassign_effect(node.target, node.op, node.value)
            if effect:
                _add_effect(effect)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            root = _self_root_name(node.func.value)
            access_path = _self_access_path(node.func.value)
            target_label = access_path or root
            if not target_label:
                continue
            mutator = node.func.attr
            payload_hint = _call_value_hint(node)
            action_map = {
                "append": (
                    f"on success, append an item to `{target_label}` {payload_hint}."
                    if payload_hint
                    else f"on success, append an item to `{target_label}`."
                ),
                "extend": (
                    f"on success, extend `{target_label}` with new items {payload_hint}."
                    if payload_hint
                    else f"on success, extend `{target_label}` with new items."
                ),
                "update": (
                    f"on success, update mapping `{target_label}` with provided keys {payload_hint}."
                    if payload_hint
                    else f"on success, update mapping `{target_label}` with provided keys."
                ),
                "remove": f"on success, remove an item from `{target_label}`.",
                "pop": f"on success, pop an item from `{target_label}`.",
                "clear": f"on success, clear `{target_label}`.",
                "setdefault": f"on success, ensure a default entry in `{target_label}`.",
                "add": (
                    f"on success, add an item into `{target_label}` {payload_hint}."
                    if payload_hint
                    else f"on success, add an item into `{target_label}`."
                ),
            }
            action = action_map.get(mutator)
            if action:
                _add_effect(action)

    return effects


def _format_joined_str_template(node: ast.JoinedStr) -> str:
    """Convert an f-string AST node into a readable template string."""
    parts: List[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant):
            parts.append(str(value.value))
        elif isinstance(value, ast.FormattedValue):
            expr = _safe_unparse(value.value)
            parts.append("{" + expr + "}")
    return "".join(parts).strip()


def _extract_success_string_templates_from_method(method: FunctionInfo) -> Dict[str, List[str]]:
    """Extract string templates from successful returned dict fields."""
    func_node = _parse_method_node(method.source_code)
    if not func_node:
        return {}

    templates: Dict[str, List[str]] = {}
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Dict):
            continue
        for key_node, value_node in zip(node.value.keys, node.value.values):
            if (
                not isinstance(key_node, ast.Constant)
                or not isinstance(key_node.value, str)
                or not isinstance(value_node, ast.JoinedStr)
            ):
                continue
            key_name = key_node.value
            if key_name == "error":
                continue
            template = _format_joined_str_template(value_node)
            if not template:
                continue
            existing = templates.setdefault(key_name, [])
            if template not in existing:
                existing.append(template)
    return templates


def _extract_behavior_hints_from_method(method: FunctionInfo) -> List[str]:
    """Extract simulation-relevant behavior hints from method AST."""
    func_node = _parse_method_node(method.source_code)
    if not func_node:
        return []

    hints: List[str] = []
    seen: Set[str] = set()

    def _add_hint(text: str) -> None:
        normalized = " ".join((text or "").split()).strip()
        if not normalized:
            return
        if normalized in seen:
            return
        seen.add(normalized)
        hints.append(normalized)

    # Pattern A: non-recursive listing with hidden-dot filtering.
    has_list_contents_call = False
    has_dot_hidden_filter = False
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "_list_contents":
                has_list_contents_call = True

            if node.func.attr == "startswith":
                if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == ".":
                    has_dot_hidden_filter = True
        elif isinstance(node, ast.ListComp):
            for generator in node.generators:
                for condition in generator.ifs:
                    if isinstance(condition, ast.UnaryOp) and isinstance(condition.op, ast.Not):
                        operand = condition.operand
                        if isinstance(operand, ast.Call) and isinstance(operand.func, ast.Attribute):
                            if (
                                operand.func.attr == "startswith"
                                and operand.args
                                and isinstance(operand.args[0], ast.Constant)
                                and operand.args[0].value == "."
                            ):
                                has_dot_hidden_filter = True
    if has_list_contents_call:
        _add_hint(
            "Listing returns direct children of the current scope only and does not recurse into nested directories."
        )
    if has_list_contents_call and has_dot_hidden_filter:
        _add_hint(
            "Dot-prefixed hidden names are excluded only when the hide/filter flag is false; when enabled, hidden names in the current scope are included."
        )

    # Pattern B: line-based grep semantics (return full lines, not substrings).
    for node in ast.walk(func_node):
        if not isinstance(node, ast.ListComp):
            continue
        if not isinstance(node.elt, ast.Name):
            continue
        line_var = node.elt.id
        if not node.generators:
            continue
        generator = node.generators[0]
        iter_expr = generator.iter
        is_splitlines_iter = (
            isinstance(iter_expr, ast.Call)
            and isinstance(iter_expr.func, ast.Attribute)
            and iter_expr.func.attr == "splitlines"
        )
        if not is_splitlines_iter:
            continue
        has_pattern_in_line = False
        for condition in generator.ifs:
            if (
                isinstance(condition, ast.Compare)
                and len(condition.ops) == 1
                and isinstance(condition.ops[0], ast.In)
                and isinstance(condition.comparators[0], ast.Name)
                and condition.comparators[0].id == line_var
            ):
                has_pattern_in_line = True
                break
        if has_pattern_in_line:
            _add_hint(
                "Pattern matching is performed against splitlines() output and each returned entry must be the full original line verbatim (no substring extraction, no trimming, no normalization)."
            )

    # Pattern C: f-string response templates from returned dict payloads.
    templates = _extract_success_string_templates_from_method(method)
    for key_name, values in templates.items():
        for template in values:
            _add_hint(
                f"Success field '{key_name}' uses formatted template: {template}"
            )

    return hints


def _derive_runtime_field_name(attr_name: str) -> str:
    """Normalize attribute names for runtime-state export."""
    normalized = (attr_name or "").strip()
    if not normalized:
        return normalized
    if normalized.startswith("_"):
        normalized = normalized[1:]
    return normalized


def _is_runtime_field_candidate(field_name: str) -> bool:
    """Return whether a field is likely runtime state (not documentation text)."""
    lowered = (field_name or "").lower()
    if not lowered:
        return False
    if "description" in lowered or "docstring" in lowered:
        return False
    return True


def _extract_runtime_defaults_from_init_helpers(class_info: Optional[ClassInfo]) -> Dict[str, Any]:
    """Extract runtime defaults/rules from __init__ and init-like helper methods."""
    if not class_info:
        return {}

    method_sources: List[Tuple[str, str]] = []
    if class_info.init_method:
        method_sources.append(("__init__", class_info.init_method))

    for helper in class_info.helper_functions:
        helper_name = helper.name
        helper_lower = helper_name.lower()
        if (
            helper_name == "_load_scenario"
            or helper_lower.startswith("_load")
            or helper_lower.startswith("_init")
            or helper_lower.startswith("_setup")
            or helper_lower.startswith("_bootstrap")
        ):
            method_sources.append((helper_name, helper.source_code))

    if not method_sources:
        return {}

    runtime_defaults: Dict[str, Any] = {}
    init_rules: List[Dict[str, Any]] = []
    seen_rules: Set[str] = set()

    def _iter_self_assignment_targets(node: ast.AST) -> List[str]:
        targets: List[str] = []
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    targets.append(target.attr)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                targets.append(target.attr)
        return targets

    for method_name, source in method_sources:
        method_node = _parse_method_node(source)
        if not method_node:
            continue

        for node in ast.walk(method_node):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue

            targets = _iter_self_assignment_targets(node)
            if not targets:
                continue

            value_node = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
            literal_value = _literal_eval_node(value_node)
            from_self_attr = (
                isinstance(value_node, ast.Attribute)
                and isinstance(value_node.value, ast.Name)
                and value_node.value.id == "self"
            )

            for raw_target in targets:
                target_field = _derive_runtime_field_name(raw_target)
                if not target_field:
                    continue
                if not _is_runtime_field_candidate(target_field):
                    continue

                if literal_value is not None:
                    if target_field not in runtime_defaults:
                        runtime_defaults[target_field] = literal_value
                    continue

                if from_self_attr:
                    source_field = _derive_runtime_field_name(value_node.attr)
                    if not source_field:
                        continue
                    rule = {
                        "field": target_field,
                        "init_from": source_field,
                        "source_method": method_name,
                    }
                    marker = json.dumps(rule, ensure_ascii=False, sort_keys=True)
                    if marker not in seen_rules:
                        seen_rules.add(marker)
                        init_rules.append(rule)

                    if target_field not in runtime_defaults:
                        runtime_defaults[target_field] = {"init_from": source_field}

    if init_rules:
        runtime_defaults["init_rules"] = init_rules

    return runtime_defaults


def extract_rule_based_state_data(
    extracted_data: ExtractedData, class_info: Optional[ClassInfo]
) -> Dict[str, Any]:
    """Build state data in a deterministic rule-based way."""
    runtime_defaults: Dict[str, Any] = {}
    for var_name, source in extracted_data.default_state.items():
        parsed = _extract_assigned_value(str(source), var_name)
        if isinstance(parsed, dict):
            runtime_defaults.update(parsed)
        elif parsed is not None:
            runtime_defaults[var_name] = parsed

    helper_runtime_defaults = _extract_runtime_defaults_from_init_helpers(class_info)
    for key, value in helper_runtime_defaults.items():
        if key not in runtime_defaults:
            runtime_defaults[key] = value
        elif isinstance(runtime_defaults.get(key), dict) and isinstance(value, dict):
            merged = dict(runtime_defaults[key])
            merged.update(value)
            runtime_defaults[key] = merged

    global_state: Dict[str, Any] = {"runtime_defaults": runtime_defaults}

    tools_state: Dict[str, Any] = {}
    method_static_map: Dict[str, Dict[str, Any]] = {}
    method_calls_map: Dict[str, List[str]] = {}
    if class_info:
        method_names = {m.name for m in class_info.methods}
        for method in class_info.methods:
            static_data = _extract_static_data_from_method(method)
            method_static_map[method.name] = static_data
            validation_rules = _extract_validation_rules_from_method(method)
            state_effects = _extract_state_effects_from_method(method)
            behavior_hints = _extract_behavior_hints_from_method(method)
            success_string_templates = _extract_success_string_templates_from_method(method)
            _, _, calls = _extract_state_io_and_calls(method)
            method_calls = [name for name in calls if name in method_names and name != method.name]
            method_calls_map[method.name] = method_calls

            entry: Dict[str, Any] = {}
            if static_data:
                entry["static_data"] = static_data
            if validation_rules:
                entry["validation_rules"] = validation_rules
            if state_effects:
                entry["state_effects"] = state_effects
            if behavior_hints:
                entry["behavior_hints"] = behavior_hints
            if success_string_templates:
                entry["success_string_templates"] = success_string_templates
            if method_calls:
                entry["method_calls"] = method_calls
            if entry:
                tools_state[method.name] = entry

        # Enrich each method with static data inherited from called methods.
        for method_name, method_calls in method_calls_map.items():
            if not method_calls:
                continue
            existing_entry = tools_state.get(method_name)
            entry: Dict[str, Any] = existing_entry if isinstance(existing_entry, dict) else {}
            static_data_existing = entry.get("static_data")
            static_data: Dict[str, Any] = (
                dict(static_data_existing) if isinstance(static_data_existing, dict) else {}
            )

            called_static_bundle: Dict[str, Any] = {}
            for called_name in method_calls:
                called_static = method_static_map.get(called_name, {})
                if not isinstance(called_static, dict) or not called_static:
                    continue
                called_static_bundle[called_name] = called_static

                # Propagate useful return literals through generic alias inference.
                return_literal = None
                if "return_value" in called_static:
                    return_literal = called_static.get("return_value")
                elif "return_values" in called_static:
                    values = called_static.get("return_values")
                    if isinstance(values, list) and values:
                        return_literal = values[0]
                if return_literal is not None:
                    for alias in _infer_called_static_aliases(called_name):
                        if alias not in static_data:
                            static_data[alias] = return_literal

            if static_data and (
                not isinstance(static_data_existing, dict) or static_data_existing != static_data
            ):
                entry["static_data"] = static_data
            if called_static_bundle:
                entry["called_method_static_data"] = called_static_bundle
            if entry:
                tools_state[method_name] = entry

    return {"global": global_state, "tools": tools_state}


def _fallback_ref_schema(ref_name: str, reason: Optional[str] = None) -> Dict[str, Any]:
    """Create a safe inline schema when a $ref cannot be expanded."""
    description = re.sub(r"(?<!^)(?=[A-Z])", " ", ref_name).strip() or "Object"
    if reason:
        description = f"{description} details ({reason})"
    else:
        description = f"{description} details"
    return {"type": "object", "description": description}


def resolve_refs_inplace(
    obj: Any,
    components_schemas: Optional[Dict[str, Any]] = None,
    resolution_stack: Tuple[str, ...] = (),
    seen_objects: Optional[set[int]] = None,
    depth: int = 0,
    max_depth: int = 30,
) -> None:
    """Safely expand $ref recursively with cycle/depth protection."""
    if components_schemas is None:
        components_schemas = {}
    if seen_objects is None:
        seen_objects = set()
    if depth > max_depth:
        return
    if not isinstance(obj, (dict, list)):
        return

    obj_id = id(obj)
    if obj_id in seen_objects:
        return
    seen_objects.add(obj_id)
    try:
        child_stack = resolution_stack

        if isinstance(obj, dict) and "$ref" in obj:
            ref_path = str(obj.get("$ref"))
            ref_name = ref_path.rsplit("/", 1)[-1] if ref_path else "Object"
            extras = {k: v for k, v in obj.items() if k != "$ref"}

            if ref_name in resolution_stack:
                obj.clear()
                obj.update(_fallback_ref_schema(ref_name, "recursive reference"))
                obj.update(extras)
            elif depth >= max_depth:
                obj.clear()
                obj.update(_fallback_ref_schema(ref_name, "max depth reached"))
                obj.update(extras)
            elif ref_name in components_schemas and isinstance(components_schemas[ref_name], dict):
                resolved = deepcopy(components_schemas[ref_name])
                obj.clear()
                obj.update(resolved)
                obj.update(extras)
                child_stack = resolution_stack + (ref_name,)
            else:
                obj.clear()
                obj.update(_fallback_ref_schema(ref_name, "unresolved reference"))
                obj.update(extras)

        if isinstance(obj, dict):
            for key in list(obj.keys()):
                resolve_refs_inplace(
                    obj[key],
                    components_schemas=components_schemas,
                    resolution_stack=child_stack,
                    seen_objects=seen_objects,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
        elif isinstance(obj, list):
            for item in obj:
                resolve_refs_inplace(
                    item,
                    components_schemas=components_schemas,
                    resolution_stack=child_stack,
                    seen_objects=seen_objects,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
    finally:
        seen_objects.remove(obj_id)


def remove_refs_generic(obj: Any, components_schemas: Optional[Dict[str, Any]] = None) -> None:
    """Resolve and inline $ref values recursively."""
    resolve_refs_inplace(obj, components_schemas)


def remove_examples_inplace(obj: Any) -> None:
    """Remove `example`/`examples` recursively."""
    if isinstance(obj, dict):
        obj.pop("example", None)
        obj.pop("examples", None)
        for value in obj.values():
            remove_examples_inplace(value)
    elif isinstance(obj, list):
        for item in obj:
            remove_examples_inplace(item)


def ensure_descriptions_inplace(obj: Any) -> None:
    """Ensure each property schema has a description."""
    if isinstance(obj, dict):
        if "properties" in obj:
            for prop_name, prop_schema in obj["properties"].items():
                if isinstance(prop_schema, dict) and "description" not in prop_schema:
                    prop_type = prop_schema.get("type", "any")
                    prop_schema["description"] = f"{prop_name} ({prop_type})"
        for value in obj.values():
            ensure_descriptions_inplace(value)
    elif isinstance(obj, list):
        for item in obj:
            ensure_descriptions_inplace(item)


def create_default_200_response() -> Dict[str, Any]:
    return {
        "description": "Operation result",
        "content": {
            "application/json": {
                "schema": {
                    "oneOf": [
                        {"type": "object", "description": "Success response"},
                        {
                            "type": "object",
                            "properties": {
                                "error": {"type": "string", "description": "Error message"}
                            },
                            "required": ["error"],
                            "description": "Error response",
                        },
                    ]
                }
            }
        },
    }


def fix_responses_generic(responses: Dict[str, Any]) -> Dict[str, Any]:
    """Convert response map to a single 200 response using `oneOf`."""
    if not responses:
        return {"200": create_default_200_response()}

    if "200" in responses:
        resp_200 = responses["200"]
        if "content" in resp_200 and "application/json" in resp_200["content"]:
            schema = resp_200["content"]["application/json"].get("schema", {})
            if "oneOf" in schema:
                return {"200": resp_200}

    success_schema = None
    error_descriptions = []

    for status, response in responses.items():
        if status in ["200", 200]:
            if "content" in response and "application/json" in response["content"]:
                schema = response["content"]["application/json"].get("schema", {})
                if "oneOf" not in schema:
                    success_schema = schema
        else:
            desc = response.get("description", f"Error {status}")
            error_descriptions.append(desc)

    if not success_schema:
        success_schema = {"type": "object", "description": "Operation result"}

    one_of_schema = {
        "oneOf": [
            {**success_schema, "description": "Success response"},
            {
                "type": "object",
                "properties": {
                    "error": {
                        "type": "string",
                        "description": (
                            f"Error message. Possible errors: {'; '.join(error_descriptions)}"
                            if error_descriptions
                            else "Error message"
                        ),
                    }
                },
                "required": ["error"],
                "description": "Error response",
            },
        ]
    }
    return {
        "200": {
            "description": "Operation result",
            "content": {"application/json": {"schema": one_of_schema}},
        }
    }


def fix_endpoint_structure(endpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Fill required endpoint fields and normalize responses."""
    if not endpoint:
        return {}

    if "operationId" not in endpoint:
        endpoint["operationId"] = "operation"

    if "requestBody" not in endpoint:
        endpoint["requestBody"] = {
            "content": {
                "application/json": {
                    "schema": {"type": "object", "properties": {}, "description": "Request body"}
                }
            }
        }

    endpoint["responses"] = fix_responses_generic(endpoint.get("responses", {}))
    return endpoint


def validate_and_fix_endpoint_structure(endpoint: Dict[str, Any]) -> int:
    """Apply minimal structural fixes for one endpoint and return fix count."""
    issues_fixed = 0

    if "requestBody" in endpoint:
        rb = endpoint["requestBody"]
        if "content" not in rb:
            endpoint["requestBody"] = {
                "content": {"application/json": {"schema": {"type": "object", "properties": {}}}}
            }
            issues_fixed += 1

    if "responses" in endpoint:
        for response in endpoint["responses"].values():
            if "content" in response and "application/json" in response["content"]:
                if "schema" not in response["content"]["application/json"]:
                    response["content"]["application/json"]["schema"] = {"type": "object"}
                    issues_fixed += 1

    return issues_fixed


def validate_and_fix_paths(paths: Dict[str, Any]) -> int:
    """Apply minimal structural fixes for all endpoints in all paths."""
    issues_fixed = 0
    for methods in paths.values():
        for endpoint in methods.values():
            if isinstance(endpoint, dict):
                issues_fixed += validate_and_fix_endpoint_structure(endpoint)
    return issues_fixed


def validate_spec_with_camel(spec: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate an OpenAPI spec with CAMEL OpenAPIToolkit."""
    try:
        started = time.perf_counter()
        from camel.toolkits import OpenAPIToolkit

        toolkit = OpenAPIToolkit()
        logger.debug("validate_spec_with_camel: toolkit initialized")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
            json.dump(spec, temp_file, indent=2)
            temp_path = temp_file.name
        logger.debug(f"validate_spec_with_camel: temp file created at {temp_path}")
        try:
            parse_start = time.perf_counter()
            spec_dict = toolkit.parse_openapi_file(temp_path)
            logger.debug(
                "validate_spec_with_camel: parse_openapi_file elapsed=%.2fs",
                time.perf_counter() - parse_start,
            )
            if not spec_dict:
                return False, "OpenAPIToolkit.parse_openapi_file returned empty result"
            func_start = time.perf_counter()
            toolkit.generate_openapi_funcs("test", spec_dict)
            logger.debug(
                "validate_spec_with_camel: generate_openapi_funcs elapsed=%.2fs",
                time.perf_counter() - func_start,
            )
            logger.debug(
                "validate_spec_with_camel: total elapsed=%.2fs",
                time.perf_counter() - started,
            )
            return True, ""
        finally:
            Path(temp_path).unlink(missing_ok=True)
    except Exception as exc:
        return False, str(exc)

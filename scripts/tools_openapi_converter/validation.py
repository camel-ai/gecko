"""OpenAPI spec validation and post-processing."""

from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Dict

from openapi_spec_validator import validate_spec

logger = logging.getLogger(__name__)


class SpecValidator:
    """Stateless validation utilities for OpenAPI specs."""

    _UNICODE_PROPERTY_PATTERN = re.compile(r"\\p\{([A-Za-z]+)\}")

    @staticmethod
    def _normalize_regex_pattern(pattern: Any) -> Any:
        """Convert unsupported Unicode-property regex into conservative ASCII-safe regex."""
        if not isinstance(pattern, str) or "\\p{" not in pattern:
            return pattern

        replacements = {
            "L": "A-Za-z",
            "N": "0-9",
        }

        unsupported = False

        def repl(match: re.Match[str]) -> str:
            nonlocal unsupported
            token = match.group(1)
            replacement = replacements.get(token)
            if replacement is None:
                unsupported = True
                return match.group(0)
            return replacement

        normalized = SpecValidator._UNICODE_PROPERTY_PATTERN.sub(repl, pattern)
        if unsupported or "\\p{" in normalized:
            return None
        return normalized

    @staticmethod
    def remove_examples_inplace(obj: Any) -> None:
        if isinstance(obj, dict):
            obj.pop("example", None)
            obj.pop("examples", None)
            for value in obj.values():
                SpecValidator.remove_examples_inplace(value)
        elif isinstance(obj, list):
            for item in obj:
                SpecValidator.remove_examples_inplace(item)

    @staticmethod
    def assert_no_refs(obj: Any, path: str = "root") -> None:
        if isinstance(obj, dict):
            if "$ref" in obj:
                raise ValueError(f"$ref is not allowed in strict mode at {path}")
            for key, value in obj.items():
                SpecValidator.assert_no_refs(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                SpecValidator.assert_no_refs(item, f"{path}[{index}]")

    @staticmethod
    def assert_descriptions(obj: Any, path: str = "root") -> None:
        if isinstance(obj, dict):
            if "type" in obj:
                description = obj.get("description")
                if not isinstance(description, str) or not description.strip():
                    raise ValueError(f"Missing description for schema node at {path}")
            for key, value in obj.items():
                SpecValidator.assert_descriptions(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                SpecValidator.assert_descriptions(item, f"{path}[{index}]")

    @staticmethod
    def normalize_schema_compat(obj: Any) -> None:
        """Normalize generated schemas to valid OpenAPI 3.1 / JSON Schema forms."""
        if isinstance(obj, dict):
            if obj.get("type") == "any":
                obj.pop("type", None)
            if (
                obj.get("exclusiveMinimum") is True
                and isinstance(obj.get("minimum"), (int, float))
            ):
                obj["exclusiveMinimum"] = obj["minimum"]
                obj.pop("minimum", None)
            if (
                obj.get("exclusiveMaximum") is True
                and isinstance(obj.get("maximum"), (int, float))
            ):
                obj["exclusiveMaximum"] = obj["maximum"]
                obj.pop("maximum", None)
            if "pattern" in obj:
                normalized_pattern = SpecValidator._normalize_regex_pattern(obj.get("pattern"))
                if normalized_pattern is None:
                    obj.pop("pattern", None)
                else:
                    obj["pattern"] = normalized_pattern
            for value in obj.values():
                SpecValidator.normalize_schema_compat(value)
        elif isinstance(obj, list):
            for item in obj:
                SpecValidator.normalize_schema_compat(item)

    @staticmethod
    def post_process(spec: Dict[str, Any]) -> None:
        """Clean up and structurally validate the full spec (single pass)."""
        SpecValidator.remove_examples_inplace(spec)
        SpecValidator.normalize_schema_compat(spec)
        SpecValidator.assert_no_refs(spec)

        for path, methods in spec.get("paths", {}).items():
            if not isinstance(methods, dict):
                raise ValueError(f"Invalid methods object at path {path!r}")
            for method, endpoint in methods.items():
                if method.lower() not in {
                    "get", "post", "put", "patch", "delete", "head", "options",
                }:
                    raise ValueError(f"Unsupported HTTP method {method!r} under path {path!r}")
                if not isinstance(endpoint, dict):
                    raise ValueError(f"Endpoint definition must be object at {path} {method}")

                operation_id = endpoint.get("operationId")
                if not isinstance(operation_id, str) or not operation_id.strip():
                    raise ValueError(f"Missing operationId at {path} {method}")
                summary = endpoint.get("summary")
                if not isinstance(summary, str) or not summary.strip():
                    raise ValueError(f"Missing summary at {path} {method}")
                description = endpoint.get("description")
                if not isinstance(description, str) or not description.strip():
                    raise ValueError(f"Missing description at {path} {method}")

                request_body = endpoint.get("requestBody")
                if not isinstance(request_body, dict):
                    raise ValueError(f"Missing requestBody at {path} {method}")
                req_schema = (
                    request_body.get("content", {})
                    .get("application/json", {})
                    .get("schema")
                )
                if not isinstance(req_schema, dict):
                    raise ValueError(
                        f"Missing requestBody.content.application/json.schema at {path} {method}"
                    )
                if req_schema.get("type") != "object":
                    raise ValueError(f"request schema type must be object at {path} {method}")

                responses = endpoint.get("responses")
                if not isinstance(responses, dict) or "200" not in responses:
                    raise ValueError(f"Missing 200 response at {path} {method}")
                response_schema = (
                    responses.get("200", {})
                    .get("content", {})
                    .get("application/json", {})
                    .get("schema")
                )
                if not isinstance(response_schema, dict):
                    raise ValueError(f"Missing response schema at {path} {method}")

    @staticmethod
    def validate_openapi(spec: Dict[str, Any]) -> None:
        validate_spec(spec)

    @staticmethod
    def validate_with_camel(spec: Dict[str, Any]) -> None:
        try:
            from camel.toolkits import OpenAPIToolkit
        except Exception as exc:
            raise RuntimeError(f"CAMEL import failed: {exc}") from exc

        toolkit = OpenAPIToolkit()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp:
            json.dump(spec, temp, indent=2, ensure_ascii=False)
            temp_path = Path(temp.name)

        try:
            spec_dict = toolkit.parse_openapi_file(str(temp_path))
            if not spec_dict:
                raise RuntimeError("CAMEL failed to parse generated OpenAPI spec")
            funcs = toolkit.generate_openapi_funcs("toolkit", spec_dict)
            count = len(funcs) if isinstance(funcs, list) else len(funcs or {})
            logger.info("CAMEL validation succeeded: %d callable functions", count)
        except Exception as exc:
            raise RuntimeError(f"CAMEL validation failed: {exc}") from exc
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def mandatory_validate(spec: Dict[str, Any]) -> None:
        SpecValidator.validate_openapi(spec)
        SpecValidator.validate_with_camel(spec)

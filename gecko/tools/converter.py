"""Convert Python callables/classes to basic OpenAPI schemas."""

from __future__ import annotations

import inspect
import logging
from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Optional, Type, get_type_hints

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SchemaConverter:
    """Converts Python signatures into simple OpenAPI documents."""

    def __init__(self):
        self.type_mapping = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array"},
            dict: {"type": "object"},
            List[str]: {"type": "array", "items": {"type": "string"}},
            List[int]: {"type": "array", "items": {"type": "integer"}},
            List[float]: {"type": "array", "items": {"type": "number"}},
            List[Dict]: {"type": "array", "items": {"type": "object"}},
        }

    def convert_function_to_openapi(
        self,
        func: callable,
        api_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert one callable into a single-path OpenAPI schema."""
        api_name = api_name or func.__name__
        description = description or func.__doc__ or f"API for {func.__name__}"

        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        parameters = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            param_schema = self._convert_parameter(param_name, param, type_hints)
            if param_schema:
                parameters.append(param_schema)

        return {
            "openapi": "3.0.0",
            "info": {
                "title": f"{api_name} API",
                "version": "1.0.0",
                "description": description,
            },
            "servers": [{"url": "http://localhost:8000", "description": "Gecko server"}],
            "paths": {
                f"/{api_name.lower()}/{func.__name__}": {
                    "post": {
                        "summary": func.__name__.replace("_", " ").title(),
                        "description": func.__doc__ or f"Execute {func.__name__}",
                        "operationId": func.__name__.lower(),
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {item["name"]: item["schema"] for item in parameters},
                                        "required": [
                                            item["name"]
                                            for item in parameters
                                            if item.get("required", False)
                                        ],
                                    }
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Successful response",
                                "content": {"application/json": {"schema": {"type": "object"}}},
                            }
                        },
                    }
                }
            },
        }

    def convert_multi_function_to_openapi(
        self,
        functions: List[callable],
        api_name: str,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert multiple functions into one OpenAPI document."""
        schema = {
            "openapi": "3.0.0",
            "info": {
                "title": f"{api_name} API",
                "version": "1.0.0",
                "description": description or f"Multi-function API for {api_name}",
            },
            "servers": [{"url": "http://localhost:8000", "description": "Gecko server"}],
            "paths": {},
        }

        for func in functions:
            path = f"/{api_name.lower()}/{func.__name__}"
            schema["paths"][path] = self._convert_function_schema(func)

        return schema

    def convert_class_to_openapi(self, cls: Type, api_name: Optional[str] = None) -> Dict[str, Any]:
        """Convert all public methods in a class into OpenAPI paths."""
        api_name = api_name or cls.__name__
        methods = [getattr(cls, name) for name in dir(cls) if not name.startswith("_") and callable(getattr(cls, name))]
        return self.convert_multi_function_to_openapi(methods, api_name, cls.__doc__ or f"API for {cls.__name__}")

    def _convert_parameter(self, param_name: str, param: inspect.Parameter, type_hints: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        param_type = type_hints.get(param_name, param.annotation)
        if param_type == inspect.Parameter.empty:
            param_type = str

        schema = self._convert_type_to_schema(param_type)
        required = param.default == inspect.Parameter.empty

        param_schema = {
            "name": param_name,
            "schema": schema,
            "required": required,
        }
        if param.default != inspect.Parameter.empty:
            param_schema["schema"]["default"] = param.default
        return param_schema

    def _convert_function_schema(self, func: callable) -> Dict[str, Any]:
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        parameters = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            item = self._convert_parameter(param_name, param, type_hints)
            if item:
                parameters.append(item)

        request_schema = {
            "type": "object",
            "properties": {item["name"]: item["schema"] for item in parameters},
            "required": [item["name"] for item in parameters if item.get("required", False)],
        }

        return {
            "post": {
                "summary": func.__name__.replace("_", " ").title(),
                "description": func.__doc__ or f"Execute {func.__name__}",
                "operationId": func.__name__.lower(),
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": request_schema}},
                },
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    }
                },
            }
        }

    def _convert_type_to_schema(self, param_type: Type) -> Dict[str, Any]:
        if param_type in self.type_mapping:
            return self.type_mapping[param_type].copy()

        origin = getattr(param_type, "__origin__", None)
        if origin is list:
            args = getattr(param_type, "__args__", ())
            return {"type": "array", "items": self._convert_type_to_schema(args[0])} if args else {"type": "array"}
        if origin is dict:
            return {"type": "object"}

        if isinstance(param_type, type) and issubclass(param_type, BaseModel):
            return param_type.model_json_schema()

        if is_dataclass(param_type):
            properties: Dict[str, Any] = {}
            required: List[str] = []
            for item in fields(param_type):
                properties[item.name] = self._convert_type_to_schema(item.type)
                if item.default is inspect.Parameter.empty:
                    required.append(item.name)
            return {"type": "object", "properties": properties, "required": required}

        logger.warning("Unknown type %s, defaulting to string", param_type)
        return {"type": "string"}

    def validate_schema(self, schema: Dict[str, Any]) -> bool:
        """Perform minimal structure checks for generated schema."""
        try:
            if not all(key in schema for key in ("openapi", "info", "paths")):
                return False
            info = schema["info"]
            if not all(key in info for key in ("title", "version")):
                return False
            return isinstance(schema["paths"], dict)
        except Exception as exc:
            logger.error("Schema validation failed: %s", exc)
            return False


_converter_instance: Optional[SchemaConverter] = None


def get_schema_converter() -> SchemaConverter:
    """Return singleton converter instance."""
    global _converter_instance
    if _converter_instance is None:
        _converter_instance = SchemaConverter()
    return _converter_instance

"""Minimal tau2-compatible toolkit abstractions.

This intentionally implements only the pieces needed by the airline/retail
domain tools, the Gecko adapter, and local evaluator replay.
"""

from __future__ import annotations

import inspect
from copy import deepcopy
from enum import Enum
from typing import Annotated, Any, Callable, Dict, Optional, TypeVar

from pydantic import BaseModel, Field, create_model

from benchmarks.taubench.internal.environment.db import DB
from benchmarks.taubench.internal.utils import get_dict_hash, update_pydantic_model_with_dict

TOOL_ATTR = "__tool__"
TOOL_TYPE_ATTR = "__tool_type__"
MUTATES_STATE_ATTR = "__mutates_state__"
DISCOVERABLE_ATTR = "__discoverable__"

T = TypeVar("T", bound=DB)


def _openai_strict_schema(schema: Any) -> Any:
    """Return a schema acceptable to OpenAI strict tool validation.

    Pydantic may emit object schemas inside anyOf/$defs without an explicit
    additionalProperties flag. OpenAI rejects those nested objects even though
    normal JSON Schema allows the omission.
    """
    if isinstance(schema, list):
        return [_openai_strict_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    fixed = {key: _openai_strict_schema(value) for key, value in schema.items()}
    is_object = fixed.get("type") == "object" or isinstance(fixed.get("properties"), dict)
    if is_object:
        fixed["additionalProperties"] = False
        properties = fixed.get("properties")
        if isinstance(properties, dict):
            required = list(fixed.get("required") or [])
            for key in properties:
                if key not in required:
                    required.append(key)
            fixed["required"] = required
    return fixed


class ToolType(str, Enum):
    READ = "read"
    WRITE = "write"
    THINK = "think"
    GENERIC = "generic"


def is_tool(tool_type: ToolType = ToolType.READ, mutates_state: Optional[bool] = None):
    if mutates_state is None:
        mutates_state = tool_type == ToolType.WRITE

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, TOOL_ATTR, True)
        setattr(func, TOOL_TYPE_ATTR, tool_type)
        setattr(func, MUTATES_STATE_ATTR, mutates_state)
        return func

    return decorator


def is_discoverable_tool(
    tool_type: ToolType = ToolType.READ,
    mutates_state: Optional[bool] = None,
):
    if mutates_state is None:
        mutates_state = tool_type == ToolType.WRITE

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, TOOL_ATTR, True)
        setattr(func, TOOL_TYPE_ATTR, tool_type)
        setattr(func, MUTATES_STATE_ATTR, mutates_state)
        setattr(func, DISCOVERABLE_ATTR, True)
        return func

    return decorator


class Tool:
    """Small function wrapper compatible with the fields GATS reads from tau2 tools."""

    def __init__(self, func: Callable[..., Any]) -> None:
        self._func = func
        self.name = func.__name__
        self.__name__ = self.name
        self.__doc__ = func.__doc__
        self.__signature__ = inspect.signature(func)
        self.short_desc, self.long_desc = self._split_doc(func.__doc__ or "")

    @staticmethod
    def _split_doc(doc: str) -> tuple[str, str]:
        cleaned = inspect.cleandoc(doc)
        if not cleaned:
            return "", ""
        parts = cleaned.split("\n\n", 1)
        return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""

    def get_function_name(self) -> str:
        return self.name

    @property
    def openai_tool_schema(self) -> dict[str, Any]:
        return self.openai_schema

    @property
    def openai_schema(self) -> dict[str, Any]:
        fields: dict[str, tuple[Any, Any]] = {}
        for name, param in self.__signature__.parameters.items():
            if name == "self":
                continue
            annotation = param.annotation if param.annotation is not inspect._empty else Any
            default = ... if param.default is inspect._empty else param.default
            fields[name] = (annotation, default)
        params = create_model(f"{self.name}_parameters", **fields)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self._description(),
                "parameters": _openai_strict_schema(params.model_json_schema()),
            },
        }

    def _description(self) -> str:
        if not self.long_desc:
            return self.short_desc or self.name
        return f"{self.short_desc}\n\n{self.long_desc}"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return deepcopy(self._func(*args, **kwargs))

    def __str__(self) -> str:
        return f"def {self.name}{self.__signature__}:\n    \"\"\"{self.__doc__ or ''}\"\"\""


def as_tool(func: Callable[..., Any], **_: Any) -> Tool:
    return Tool(func)


class ToolKitType(type):
    def __init__(cls, name: str, bases: tuple[type, ...], attrs: dict[str, Any]) -> None:
        func_tools = {}
        for attr_name, method in attrs.items():
            if isinstance(method, property):
                method = method.fget
            if hasattr(method, TOOL_ATTR):
                func_tools[attr_name] = method

        @property
        def _func_tools(self) -> Dict[str, Callable[..., Any]]:
            all_func_tools = func_tools.copy()
            try:
                all_func_tools.update(super(cls, self)._func_tools)
            except AttributeError:
                pass
            return all_func_tools

        cls._func_tools = _func_tools
        super().__init__(name, bases, attrs)


class ToolKitBase(metaclass=ToolKitType):
    def __init__(self, db: Optional[T] = None) -> None:
        self.db: Optional[T] = db

    @property
    def tools(self) -> Dict[str, Callable[..., Any]]:
        return {name: getattr(self, name) for name in self._func_tools.keys()}

    def use_tool(self, tool_name: str, **kwargs: Any) -> Any:
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found.")
        return deepcopy(self.tools[tool_name](**kwargs))

    def get_tools(self, include: Optional[list[str]] = None) -> Dict[str, Tool]:
        tools = {
            name: as_tool(tool)
            for name, tool in self.tools.items()
            if not getattr(tool, DISCOVERABLE_ATTR, False)
        }
        if include is not None:
            allowed = set(include)
            unknown = allowed - set(tools.keys())
            if unknown:
                raise ValueError(f"Tool(s) not found: {sorted(unknown)}")
            tools = {name: tool for name, tool in tools.items() if name in allowed}
        return tools

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self.tools

    def tool_type(self, tool_name: str) -> ToolType:
        return getattr(self.tools[tool_name], TOOL_TYPE_ATTR)

    def tool_mutates_state(self, tool_name: str) -> bool:
        return getattr(self.tools[tool_name], MUTATES_STATE_ATTR, True)

    def update_db(self, update_data: Optional[dict[str, Any]] = None) -> None:
        if self.db is None:
            raise ValueError("Database has not been initialized.")
        self.db = update_pydantic_model_with_dict(self.db, update_data or {})

    def get_db_hash(self) -> str:
        if self.db is None:
            raise ValueError("Database has not been initialized.")
        return get_dict_hash(self.db.model_dump(mode="json", exclude_none=False))


class ToolSignature(BaseModel):
    name: Annotated[str, Field(description="The name of the tool")]
    doc: Annotated[str, Field(description="The documentation of the tool")]
    params: Annotated[Optional[dict], Field(default=None)]
    returns: Annotated[Optional[dict], Field(default=None)]


def get_tool_signatures(tools: ToolKitBase) -> dict[str, ToolSignature]:
    signatures = {}
    for name, tool in tools.get_tools().items():
        schema = tool.openai_schema.get("function", {})
        signatures[name] = ToolSignature(
            name=name,
            doc=str(tool),
            params=schema.get("parameters"),
            returns=None,
        )
    return signatures


def get_tool_types(tools: ToolKitBase) -> dict[str, ToolType]:
    return {name: tools.tool_type(name) for name in tools.get_tools().keys()}

import copy
import json
import logging
import os
import re
import unicodedata
import warnings
from typing import Any, Dict, List, Optional
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.toolkits import FunctionTool
from camel.configs import AnthropicConfig, ChatGPTConfig, GeminiConfig
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class _ContextWindowFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "context window size not defined" not in record.getMessage()

logging.getLogger().addFilter(_ContextWindowFilter())


def _model_type(enum_name: str, fallback: str) -> Any:
    """Use CAMEL's enum when available while keeping older installs usable."""
    return getattr(ModelType, enum_name, fallback)


def _config_class_for_platform(platform: ModelPlatformType) -> Any:
    """Return the native CAMEL config class for each provider family."""
    if platform == ModelPlatformType.ANTHROPIC:
        return AnthropicConfig
    if platform == ModelPlatformType.GEMINI:
        return GeminiConfig
    return ChatGPTConfig


def _cache_control_from_env(env_var: str, default: str) -> Optional[str]:
    """Return a prompt-cache TTL from env, or None to disable."""
    value = os.getenv(env_var, default).strip().lower()
    if value in {"", "0", "false", "none", "off", "disable", "disabled"}:
        return None
    if value not in {"5m", "1h"}:
        logger.warning(
            "Invalid %s=%r; falling back to %s",
            env_var,
            value,
            default,
        )
        return default if default in {"5m", "1h"} else None
    return value


def _anthropic_cache_control() -> Optional[str]:
    """Return the prompt-cache TTL for official Anthropic models."""
    return _cache_control_from_env("ANTHROPIC_CACHE_CONTROL", "5m")


def _build_model_config_dict(
    model_name: str,
    platform: ModelPlatformType,
    runtime_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build provider-specific CAMEL config without dropping valid fields."""
    if platform == ModelPlatformType.OPENAI_COMPATIBLE_MODEL:
        return {
            key: value
            for key, value in runtime_config.items()
            if value is not None
        }

    config_class = _config_class_for_platform(platform)
    allowed_fields = set(config_class.model_fields.keys())
    known_config: Dict[str, Any] = {}
    dropped_keys: List[str] = []
    for key, value in runtime_config.items():
        if value is None:
            continue
        if key in allowed_fields:
            known_config[key] = value
        else:
            dropped_keys.append(key)

    if dropped_keys:
        logger.warning(
            "Dropping unsupported model config fields for %s: %s",
            model_name,
            sorted(dropped_keys),
        )

    return config_class(**known_config).as_dict()


def strip_thinking_content(text: str) -> str:
    """Strip model reasoning blocks like <think>...</think> before JSON parsing."""
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()
    if not cleaned:
        return ""

    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()
    elif cleaned.lstrip().startswith("<think>"):
        cleaned = re.sub(r"^\s*<think>\s*", "", cleaned, count=1, flags=re.IGNORECASE).strip()

    return cleaned


def strip_code_fences(text: str) -> str:
    """Remove outer markdown code fences (``` / ```json) if present."""
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()
    if not cleaned:
        return ""

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return cleaned


def sanitize_llm_json_text(text: str) -> str:
    """Normalize LLM output for JSON parsing: remove think blocks then code fences."""
    return strip_code_fences(strip_thinking_content(text))


def _is_pydantic_response_format(response_format: Any) -> bool:
    try:
        return isinstance(response_format, type) and issubclass(
            response_format,
            BaseModel,
        )
    except TypeError:
        return False


def _json_example_from_schema(schema: Dict[str, Any]) -> Any:
    value_type = schema.get("type")
    if isinstance(value_type, list):
        value_type = next((item for item in value_type if item != "null"), None)

    if "anyOf" in schema and not value_type:
        return _json_example_from_schema(schema["anyOf"][0])
    if "oneOf" in schema and not value_type:
        return _json_example_from_schema(schema["oneOf"][0])

    if value_type == "object" or "properties" in schema:
        return {
            name: _json_example_from_schema(prop)
            for name, prop in (schema.get("properties") or {}).items()
        }
    if value_type == "array":
        return []
    if value_type == "boolean":
        return False
    if value_type == "integer":
        return 0
    if value_type == "number":
        return 0.0
    return ""


def _field_value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    value = getattr(obj, key, None)
    if value is not None:
        return value
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(key)
    return None


def _set_message_field(message: Any, key: str, value: Any) -> None:
    if isinstance(message, dict):
        message[key] = value
        return
    try:
        setattr(message, key, value)
    except Exception:
        pass


def _tool_call_ids(tool_calls: Any) -> List[str]:
    ids: List[str] = []
    if not isinstance(tool_calls, list):
        return ids
    for tool_call in tool_calls:
        tool_call_id = _field_value(tool_call, "id")
        if tool_call_id:
            ids.append(str(tool_call_id))
    return ids


def _capture_reasoning_content_for_tool_calls(model: Any, response: Any) -> None:
    cache = getattr(model, "_gecko_reasoning_by_tool_call_id", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(model, "_gecko_reasoning_by_tool_call_id", cache)

    for choice in getattr(response, "choices", []) or []:
        message = getattr(choice, "message", None)
        if message is None:
            continue
        reasoning_content = _field_value(message, "reasoning_content")
        if not reasoning_content:
            continue
        for tool_call_id in _tool_call_ids(_field_value(message, "tool_calls")):
            cache[tool_call_id] = reasoning_content


def _messages_with_cached_reasoning_content(model: Any, messages: List[Any]) -> List[Any]:
    """Restore DeepSeek thinking-mode reasoning_content for in-turn tool calls."""
    request_messages = copy.deepcopy(messages)
    cache = getattr(model, "_gecko_reasoning_by_tool_call_id", None)
    if not isinstance(cache, dict) or not cache:
        return request_messages

    for message in request_messages:
        if _field_value(message, "role") != "assistant":
            continue
        if _field_value(message, "reasoning_content"):
            continue
        tool_call_ids = _tool_call_ids(_field_value(message, "tool_calls"))
        if not tool_call_ids:
            continue
        reasoning_values = [cache.get(tool_call_id) for tool_call_id in tool_call_ids]
        reasoning_values = [value for value in reasoning_values if value]
        if reasoning_values:
            _set_message_field(message, "reasoning_content", reasoning_values[0])

    return request_messages


def _messages_with_json_schema_instruction(
    request_messages: List[Any],
    response_format: type[BaseModel],
) -> List[Any]:
    """Append JSON schema instructions to already-copied request messages."""
    if not request_messages:
        request_messages = [{"role": "user", "content": ""}]

    schema = response_format.model_json_schema()
    schema_text = json.dumps(
        schema,
        indent=2,
        ensure_ascii=False,
    )
    example_text = json.dumps(
        _json_example_from_schema(schema),
        indent=2,
        ensure_ascii=False,
    )
    instruction = (
        "\n\nRespond with JSON only. The JSON object must validate against this "
        "JSON Schema:\n"
        f"{schema_text}\n"
        "Example JSON shape:\n"
        f"{example_text}\n"
        "Do not include markdown fences or explanatory text."
    )

    last_message = request_messages[-1]
    if isinstance(last_message, dict):
        content = last_message.get("content", "")
        if isinstance(content, str):
            last_message["content"] = content + instruction
        elif isinstance(content, list):
            last_message["content"] = content + [
                {"type": "text", "text": instruction},
            ]
        else:
            last_message["content"] = f"{content}{instruction}"
    else:
        try:
            content = getattr(last_message, "content", "")
            setattr(last_message, "content", f"{content}{instruction}")
        except Exception:
            request_messages.append({"role": "user", "content": instruction})

    return request_messages


def _schema_type(schema: Dict[str, Any]) -> Optional[str]:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        for item in schema_type:
            if item != "null":
                return str(item)
        return "null" if "null" in schema_type else None
    return str(schema_type) if schema_type is not None else None


def _enum_value_matches_type(value: Any, schema_type: Optional[str]) -> bool:
    if not schema_type:
        return True
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "null":
        return value is None
    return True


def _merge_unique_json_values(values: List[Any]) -> List[Any]:
    merged: List[Any] = []
    seen = set()
    for value in values:
        try:
            key = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        merged.append(value)
    return merged


_PROVIDER_SCHEMA_PROPERTY_RE = re.compile(r"^[a-zA-Z0-9_.-]{1,64}$")


def _safe_provider_property_name(name: str, existing: set[str]) -> str:
    """Convert a JSON-schema property name to a provider-safe tool arg name."""
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", ascii_name).strip("_")
    if not safe:
        safe = "param"
    safe = safe[:64]
    base = safe
    suffix = 2
    while safe in existing:
        suffix_text = f"_{suffix}"
        safe = f"{base[:64 - len(suffix_text)]}{suffix_text}"
        suffix += 1
    return safe


def _mapping_node_has_entries(node: Dict[str, Any]) -> bool:
    return bool(node.get("keys") or node.get("children"))


def _restore_provider_arg_names(value: Any, mapping_node: Optional[Dict[str, Any]]) -> Any:
    """Map provider-safe argument keys back to canonical benchmark/tool keys."""
    if not mapping_node:
        return value
    if isinstance(value, list):
        return [_restore_provider_arg_names(item, mapping_node) for item in value]
    if not isinstance(value, dict):
        return value

    key_map = mapping_node.get("keys") or {}
    children = mapping_node.get("children") or {}
    restored: Dict[str, Any] = {}
    for key, item in value.items():
        canonical_key = key_map.get(key, key)
        child_map = children.get(key) or children.get(canonical_key)
        restored[canonical_key] = _restore_provider_arg_names(item, child_map)
    return restored


def _sanitize_provider_schema(
    value: Any,
    arg_mapping: Optional[Dict[str, Any]] = None,
) -> Any:
    """Remove local metadata and normalize JSON Schema for strict providers."""
    if isinstance(value, dict):
        result = {
            key: _sanitize_provider_schema(item)
            for key, item in value.items()
            if not str(key).startswith("x-")
        }

        properties = result.get("properties")
        if isinstance(properties, dict):
            sanitized_properties: Dict[str, Any] = {}
            rename_map: Dict[str, str] = {}
            child_maps: Dict[str, Dict[str, Any]] = {}
            existing: set[str] = set()
            for prop_name, prop_schema in properties.items():
                prop_name_str = str(prop_name)
                safe_name = prop_name_str
                if not _PROVIDER_SCHEMA_PROPERTY_RE.match(prop_name_str):
                    safe_name = _safe_provider_property_name(prop_name_str, existing)
                    rename_map[safe_name] = prop_name_str
                elif prop_name_str in existing:
                    safe_name = _safe_provider_property_name(prop_name_str, existing)
                    rename_map[safe_name] = prop_name_str
                existing.add(safe_name)

                child_mapping: Dict[str, Any] = {}
                sanitized_properties[safe_name] = _sanitize_provider_schema(
                    prop_schema,
                    child_mapping,
                )
                if _mapping_node_has_entries(child_mapping):
                    child_maps[safe_name] = child_mapping

            result["properties"] = sanitized_properties

            required = result.get("required")
            if isinstance(required, list) and rename_map:
                inverse_map = {original: safe for safe, original in rename_map.items()}
                result["required"] = [
                    inverse_map.get(item, item)
                    for item in required
                ]

            if arg_mapping is not None:
                if rename_map:
                    arg_mapping.setdefault("keys", {}).update(rename_map)
                if child_maps:
                    arg_mapping.setdefault("children", {}).update(child_maps)

        schema_type = _schema_type(result)
        enum_values = result.get("enum")
        if schema_type == "array" and isinstance(enum_values, list):
            item_schema = result.get("items")
            if not isinstance(item_schema, dict):
                item_schema = {}
            scalar_enum_values = [
                item for item in enum_values
                if not isinstance(item, list)
            ]
            if scalar_enum_values:
                item_type = _schema_type(item_schema)
                compatible_item_values = [
                    item for item in scalar_enum_values
                    if _enum_value_matches_type(item, item_type)
                ]
                if compatible_item_values:
                    existing_item_enum = item_schema.get("enum")
                    if isinstance(existing_item_enum, list):
                        compatible_item_values = existing_item_enum + compatible_item_values
                    item_schema["enum"] = _merge_unique_json_values(compatible_item_values)
                    result["items"] = item_schema
                result.pop("enum", None)
            else:
                compatible_array_values = [
                    item for item in enum_values
                    if _enum_value_matches_type(item, schema_type)
                ]
                if compatible_array_values:
                    result["enum"] = compatible_array_values
                else:
                    result.pop("enum", None)
        elif isinstance(enum_values, list) and schema_type:
            compatible_values = [
                item for item in enum_values
                if _enum_value_matches_type(item, schema_type)
            ]
            if compatible_values:
                result["enum"] = compatible_values
            else:
                result.pop("enum", None)

        return result
    if isinstance(value, list):
        return [_sanitize_provider_schema(item) for item in value]
    return value


def _tool_function_name(tool: Dict[str, Any]) -> Optional[str]:
    function = tool.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        return str(name) if name else None
    name = tool.get("name")
    return str(name) if name else None


def _tool_parameter_schema(tool: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    function = tool.get("function")
    if isinstance(function, dict):
        parameters = function.get("parameters")
        return parameters if isinstance(parameters, dict) else None
    input_schema = tool.get("input_schema")
    return input_schema if isinstance(input_schema, dict) else None


def _set_tool_parameter_schema(tool: Dict[str, Any], schema: Dict[str, Any]) -> None:
    function = tool.get("function")
    if isinstance(function, dict):
        function["parameters"] = schema
    else:
        tool["input_schema"] = schema


def _tool_strict_enabled(tool: Any) -> bool:
    if not isinstance(tool, dict):
        return False
    function = tool.get("function")
    if isinstance(function, dict):
        return function.get("strict") is True
    return tool.get("strict") is True


def _strip_tool_strict_flag(tool: Any) -> None:
    if not isinstance(tool, dict):
        return
    function = tool.get("function")
    if isinstance(function, dict):
        function.pop("strict", None)
    tool.pop("strict", None)


def _should_strip_anthropic_strict_tools(tools: Any) -> bool:
    if not isinstance(tools, list):
        return False
    return any(_tool_strict_enabled(tool) for tool in tools)


def _clean_tools_for_outbound_request(
    tools: Optional[List[Dict[str, Any]]],
    strip_strict: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    cleaned_tools, _ = _clean_tools_for_outbound_request_with_arg_maps(
        tools,
        strip_strict=strip_strict,
    )
    return cleaned_tools


def _clean_tools_for_outbound_request_with_arg_maps(
    tools: Optional[List[Dict[str, Any]]],
    strip_strict: bool = False,
) -> tuple[Optional[List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    if tools is None:
        return None, {}
    cleaned_tools = copy.deepcopy(tools)
    arg_maps: Dict[str, Dict[str, Any]] = {}
    if not isinstance(cleaned_tools, list):
        return cleaned_tools, arg_maps

    for idx, tool in enumerate(cleaned_tools):
        if not isinstance(tool, dict):
            continue
        name = _tool_function_name(tool)
        schema = _tool_parameter_schema(tool)
        if schema is not None:
            _set_tool_parameter_schema(tool, {})
        cleaned_tool = _sanitize_provider_schema(tool)
        if strip_strict:
            _strip_tool_strict_flag(cleaned_tool)
        if not name or not isinstance(schema, dict):
            cleaned_tools[idx] = cleaned_tool
            continue
        arg_mapping: Dict[str, Any] = {}
        _set_tool_parameter_schema(
            cleaned_tool,
            _sanitize_provider_schema(schema, arg_mapping),
        )
        if _mapping_node_has_entries(arg_mapping):
            arg_maps[name] = arg_mapping
        cleaned_tools[idx] = cleaned_tool

    return cleaned_tools, arg_maps


def _set_field_value(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
        return
    try:
        setattr(obj, key, value)
    except Exception:
        extra = getattr(obj, "model_extra", None)
        if isinstance(extra, dict):
            extra[key] = value


def _remap_response_tool_call_arguments(response: Any, arg_maps: Dict[str, Dict[str, Any]]) -> Any:
    if not arg_maps:
        return response
    for choice in getattr(response, "choices", []) or []:
        message = getattr(choice, "message", None)
        tool_calls = _field_value(message, "tool_calls") if message is not None else None
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            function = _field_value(tool_call, "function")
            if function is None:
                continue
            name = _field_value(function, "name")
            mapping = arg_maps.get(str(name))
            if not mapping:
                continue
            arguments = _field_value(function, "arguments")
            if isinstance(arguments, str):
                try:
                    parsed_arguments = json.loads(arguments)
                except Exception:
                    continue
                restored = _restore_provider_arg_names(parsed_arguments, mapping)
                _set_field_value(function, "arguments", json.dumps(restored, ensure_ascii=False))
            elif isinstance(arguments, dict):
                restored = _restore_provider_arg_names(arguments, mapping)
                _set_field_value(function, "arguments", restored)
    return response


def _clean_model_config_tools_for_outbound(
    model: Any,
    strip_strict: bool = False,
) -> tuple[bool, Any]:
    model_config = getattr(model, "model_config_dict", None)
    if not isinstance(model_config, dict) or "tools" not in model_config:
        return False, None

    original_tools = model_config.get("tools")
    model_config["tools"] = _clean_tools_for_outbound_request(
        original_tools,
        strip_strict=strip_strict,
    )
    return True, original_tools


def _restore_model_config_tools(model: Any, had_tools: bool, original_tools: Any) -> None:
    if not had_tools:
        return
    model_config = getattr(model, "model_config_dict", None)
    if isinstance(model_config, dict):
        model_config["tools"] = original_tools


def _tool_call_extra_content(tool_call: Any) -> Any:
    extra = _field_value(tool_call, "extra_content")
    if extra:
        return extra
    function = _field_value(tool_call, "function")
    if function is not None:
        return _field_value(function, "extra_content")
    return None


def _set_tool_call_extra_content(tool_call: Any, extra_content: Any) -> None:
    if isinstance(tool_call, dict):
        tool_call["extra_content"] = extra_content
        return
    try:
        setattr(tool_call, "extra_content", extra_content)
    except Exception:
        pass


def _capture_gemini_tool_extra_content(model: Any, response: Any) -> None:
    cache = getattr(model, "_gecko_gemini_tool_extra_by_id", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(model, "_gecko_gemini_tool_extra_by_id", cache)

    for choice in getattr(response, "choices", []) or []:
        message = getattr(choice, "message", None)
        tool_calls = _field_value(message, "tool_calls") if message is not None else None
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            tool_call_id = _field_value(tool_call, "id")
            extra_content = _tool_call_extra_content(tool_call)
            if tool_call_id and extra_content:
                cache[str(tool_call_id)] = copy.deepcopy(extra_content)


def _install_outbound_tool_schema_cleanup(
    model: Any,
    model_name: str,
    strip_anthropic_strict_limit: bool = False,
) -> None:
    """Strip local tool-schema metadata from outbound request copies."""
    original_run = getattr(model, "_run", None)
    original_arun = getattr(model, "_arun", None)
    if not callable(original_run):
        return
    if getattr(model, "_gecko_outbound_tool_schema_cleanup", False):
        return

    def _run(
        messages: List[Any],
        response_format: Any = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        strip_strict = bool(
            strip_anthropic_strict_limit
            and (
                _should_strip_anthropic_strict_tools(tools)
                or _should_strip_anthropic_strict_tools(
                    getattr(model, "model_config_dict", {}).get("tools")
                    if isinstance(getattr(model, "model_config_dict", None), dict)
                    else None
                )
            )
        )
        had_tools, original_tools = _clean_model_config_tools_for_outbound(
            model,
            strip_strict=strip_strict,
        )
        cleaned_tools, arg_maps = _clean_tools_for_outbound_request_with_arg_maps(
            tools,
            strip_strict=strip_strict,
        )
        try:
            response = original_run(
                messages,
                response_format,
                cleaned_tools,
            )
            return _remap_response_tool_call_arguments(response, arg_maps)
        finally:
            _restore_model_config_tools(model, had_tools, original_tools)

    setattr(model, "_run", _run)

    if callable(original_arun):
        async def _arun(
            messages: List[Any],
            response_format: Any = None,
            tools: Optional[List[Dict[str, Any]]] = None,
        ):
            strip_strict = bool(
                strip_anthropic_strict_limit
                and (
                    _should_strip_anthropic_strict_tools(tools)
                    or _should_strip_anthropic_strict_tools(
                        getattr(model, "model_config_dict", {}).get("tools")
                        if isinstance(getattr(model, "model_config_dict", None), dict)
                        else None
                    )
                )
            )
            had_tools, original_tools = _clean_model_config_tools_for_outbound(
                model,
                strip_strict=strip_strict,
            )
            cleaned_tools, arg_maps = _clean_tools_for_outbound_request_with_arg_maps(
                tools,
                strip_strict=strip_strict,
            )
            try:
                response = await original_arun(
                    messages,
                    response_format,
                    cleaned_tools,
                )
                return _remap_response_tool_call_arguments(response, arg_maps)
            finally:
                _restore_model_config_tools(model, had_tools, original_tools)

        setattr(model, "_arun", _arun)

    setattr(model, "_gecko_outbound_tool_schema_cleanup", True)
    logger.debug("Installed outbound tool-schema cleanup for %s", model_name)


def _install_gemini_compatibility_mode(model: Any, model_name: str) -> None:
    """Install Gemini-specific tool-call compatibility hooks."""
    original_run = getattr(model, "_run", None)
    original_arun = getattr(model, "_arun", None)
    if not callable(original_run):
        return
    setattr(model, "_gecko_gemini_tool_extra_by_id", {})

    original_process_messages = getattr(model, "_process_messages", None)
    if callable(original_process_messages) and not getattr(
        model,
        "_gecko_gemini_process_patch",
        False,
    ):
        def _process_messages(messages: List[Any]) -> List[Any]:
            tool_extra_by_id = dict(
                getattr(model, "_gecko_gemini_tool_extra_by_id", {}) or {}
            )
            for message in messages or []:
                tool_calls = _field_value(message, "tool_calls")
                if not isinstance(tool_calls, list):
                    continue
                for tool_call in tool_calls:
                    tool_call_id = _field_value(tool_call, "id")
                    extra_content = _tool_call_extra_content(tool_call)
                    if tool_call_id and extra_content:
                        tool_extra_by_id[str(tool_call_id)] = copy.deepcopy(extra_content)

            processed_messages = original_process_messages(messages)
            for message in processed_messages or []:
                tool_calls = _field_value(message, "tool_calls")
                if not isinstance(tool_calls, list):
                    continue
                for tool_call in tool_calls:
                    tool_call_id = _field_value(tool_call, "id")
                    if not tool_call_id or _tool_call_extra_content(tool_call):
                        continue
                    extra_content = tool_extra_by_id.get(str(tool_call_id))
                    if extra_content:
                        _set_tool_call_extra_content(tool_call, copy.deepcopy(extra_content))
            return processed_messages

        setattr(model, "_process_messages", _process_messages)
        setattr(model, "_gecko_gemini_process_patch", True)

    def _run(
        messages: List[Any],
        response_format: Any = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        if _is_pydantic_response_format(response_format) and tools:
            request_messages = _messages_with_json_schema_instruction(
                copy.deepcopy(messages),
                response_format,
            )
            response = original_run(request_messages, None, tools)
        else:
            response = original_run(messages, response_format, tools)
        _capture_gemini_tool_extra_content(model, response)
        return response

    setattr(model, "_run", _run)

    if callable(original_arun):
        async def _arun(
            messages: List[Any],
            response_format: Any = None,
            tools: Optional[List[Dict[str, Any]]] = None,
        ):
            if _is_pydantic_response_format(response_format) and tools:
                request_messages = _messages_with_json_schema_instruction(
                    copy.deepcopy(messages),
                    response_format,
                )
                response = await original_arun(request_messages, None, tools)
            else:
                response = await original_arun(messages, response_format, tools)
            _capture_gemini_tool_extra_content(model, response)
            return response

        setattr(model, "_arun", _arun)

    logger.debug("Installed Gemini compatibility mode for %s", model_name)


def _install_json_object_structured_output_mode(model: Any, model_name: str) -> None:
    """Install DeepSeek compatibility hooks on one model instance."""
    original_run = getattr(model, "_run", None)
    original_arun = getattr(model, "_arun", None)
    if not callable(original_run):
        return
    setattr(model, "_gecko_reasoning_by_tool_call_id", {})

    def _json_object_request_config(
        tools: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        request_config = copy.deepcopy(getattr(model, "model_config_dict", {}) or {})
        request_config.pop("stream", None)
        request_config["response_format"] = {"type": "json_object"}
        if tools is not None:
            request_config["tools"] = _clean_tools_for_outbound_request(tools)
        elif "tools" in request_config:
            request_config["tools"] = _sanitize_provider_schema(
                copy.deepcopy(request_config["tools"])
            )
        return request_config

    def _run(
        messages: List[Any],
        response_format: Any = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        request_messages = _messages_with_cached_reasoning_content(model, messages)
        if not _is_pydantic_response_format(response_format):
            response = original_run(request_messages, response_format, tools)
            _capture_reasoning_content_for_tool_calls(model, response)
            return response

        request_messages = _messages_with_json_schema_instruction(
            request_messages,
            response_format,
        )
        request_config = _json_object_request_config(tools)
        logger.debug(
            "Using JSON object structured-output compatibility mode for %s",
            model_name,
        )
        response = model._client.chat.completions.create(
            messages=request_messages,
            model=model.model_type,
            **request_config,
        )
        _capture_reasoning_content_for_tool_calls(model, response)
        return response

    setattr(model, "_run", _run)

    if callable(original_arun):
        async def _arun(
            messages: List[Any],
            response_format: Any = None,
            tools: Optional[List[Dict[str, Any]]] = None,
        ):
            request_messages = _messages_with_cached_reasoning_content(model, messages)
            if not _is_pydantic_response_format(response_format):
                response = await original_arun(request_messages, response_format, tools)
                _capture_reasoning_content_for_tool_calls(model, response)
                return response

            request_messages = _messages_with_json_schema_instruction(
                request_messages,
                response_format,
            )
            request_config = _json_object_request_config(tools)
            logger.debug(
                "Using async JSON object structured-output compatibility mode for %s",
                model_name,
            )
            response = await model._async_client.chat.completions.create(
                messages=request_messages,
                model=model.model_type,
                **request_config,
            )
            _capture_reasoning_content_for_tool_calls(model, response)
            return response

        setattr(model, "_arun", _arun)


def _install_responses_api_compatibility_mode(model: Any, model_name: str) -> None:
    """Apply local compatibility hooks for CAMEL's Responses API mode."""
    if getattr(model, "_gecko_responses_api_compatibility_mode", False):
        return
    if getattr(model, "_api_mode", None) != "responses":
        return

    def _strictify_schema(schema: Any) -> None:
        if isinstance(schema, dict):
            schema.pop("default", None)
            properties = schema.get("properties")
            if isinstance(properties, dict):
                schema["required"] = list(properties.keys())
                schema["additionalProperties"] = False
            for value in schema.values():
                _strictify_schema(value)
        elif isinstance(schema, list):
            for item in schema:
                _strictify_schema(item)

    def _strictify_response_format(request_config: Dict[str, Any]) -> None:
        text_config = request_config.get("text")
        if not isinstance(text_config, dict):
            return
        format_config = text_config.get("format")
        if not isinstance(format_config, dict):
            return
        schema = format_config.get("schema")
        if schema is not None:
            _strictify_schema(schema)
            format_config["strict"] = True

    original_prepare = getattr(model, "_prepare_responses_request_config", None)
    if callable(original_prepare):
        def _prepare_responses_request_config(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Setting `store=False` will disable `previous_response_id` chaining\.",
                    category=UserWarning,
                )
                request_config = original_prepare(*args, **kwargs)
            _strictify_response_format(request_config)
            return request_config

        setattr(
            model,
            "_prepare_responses_request_config",
            _prepare_responses_request_config,
        )

    setattr(model, "_gecko_responses_api_compatibility_mode", True)
    logger.debug("Installed Responses API compatibility mode for %s", model_name)


def create_model(
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    tools: List[FunctionTool] = None,
    timeout: Optional[float] = None,
    max_retries: int = 3,
):
    """
    Create a model instance
    
    Args:
        model_name: Model name
        temperature: Temperature parameter
        max_tokens: Maximum token count
        tools: List of tools
        timeout: Per-request timeout in seconds for model backend
        max_retries: Maximum retries for backend requests
        
    Returns:
        Created model instance
    """
    model_configs = {
        "gpt-4o": {
            "platform": ModelPlatformType.OPENAI,
            "type": ModelType.GPT_4O,
        },
        "gpt-4o-mini": {
            "platform": ModelPlatformType.OPENAI,
            "type": ModelType.GPT_4O_MINI,
        },
        "gpt-4.1-mini": {
            "platform": ModelPlatformType.OPENAI,
            "type": ModelType.GPT_4_1_MINI,
        },
        "gpt-4.1": {
            "platform": ModelPlatformType.OPENAI,
            "type": ModelType.GPT_4_1,
        },
        "gpt-5": {
            "platform": ModelPlatformType.OPENAI,
            "type": ModelType.GPT_5,
            "allow_max_tokens": False,
            "allow_temperature": False,
        },
        "gpt-5.5": {
            "platform": ModelPlatformType.OPENAI,
            "type": "gpt-5.5",
            "allow_max_tokens": False,
            "allow_temperature": False,
        },
        "haivex-gpt-5.5": {
            "platform": ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            "type": "gpt-5.5",
            "url": "https://api.intenext.ai/v1",
            "api_key": "INTENEXT_API_KEY",
            "allow_max_tokens": False,
            "allow_temperature": False,
        },
        "haivex-gpt-5.5-responses": {
            "platform": ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            "type": "gpt-5.5",
            "url": "https://api.intenext.ai/v1",
            "api_key": "INTENEXT_API_KEY",
            "allow_max_tokens": False,
            "allow_temperature": False,
            "api_mode": "responses",
            "extra_config": {
                "store": False,
            },
        },
        "claude-opus-4-7": {
            "platform": ModelPlatformType.ANTHROPIC,
            "type": _model_type("CLAUDE_OPUS_4_7", "claude-opus-4-7"),
            "api_key": "ANTHROPIC_API_KEY",
            "allow_temperature": False,
            "extra_config": {"cache_control": _anthropic_cache_control()},
        },
        "opus-4.7": {
            "platform": ModelPlatformType.ANTHROPIC,
            "type": _model_type("CLAUDE_OPUS_4_7", "claude-opus-4-7"),
            "api_key": "ANTHROPIC_API_KEY",
            "allow_temperature": False,
            "extra_config": {"cache_control": _anthropic_cache_control()},
        },
        "gemini-3.1-pro-preview": {
            "platform": ModelPlatformType.GEMINI,
            "type": "gemini-3.1-pro-preview",
            "api_key": "GEMINI_API_KEY",
            "allow_temperature": False,
        },
        "gemini-3.5-flash": {
            "platform": ModelPlatformType.GEMINI,
            "type": "gemini-3.5-flash",
            "api_key": "GEMINI_API_KEY",
            "allow_temperature": False,
        },
        "gemini-3.1-flash-lite": {
            "platform": ModelPlatformType.GEMINI,
            "type": "gemini-3.1-flash-lite",
            "api_key": "GEMINI_API_KEY",
            "allow_temperature": False,
        },
        "gpt-5-mini": {
            "platform": ModelPlatformType.OPENAI,
            "type": ModelType.GPT_5_MINI,
            "allow_max_tokens": False,
            "allow_temperature": False,
        },
        "DeepSeek-V4.0-Pro-thinking": {
            "platform": ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            "type": "deepseek-v4-pro",
            "url": "https://api.deepseek.com",
            "api_key": "DEEPSEEK_API_KEY",
            "extra_config": {
                "extra_body": {
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": "high",
                },
            },
            "structured_output_mode": "json_object",
        },
        "deepinfra-qwen3-14b": {
            "platform": ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            "type": "Qwen/Qwen3-14B",
            "url": "https://api.deepinfra.com/v1/openai",
            "api_key": "DEEPINFRA_API_KEY",
            "extra_config": {
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True, "max_thinking_tokens": 4096}},
            },
        },
    }


    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}")

    config = model_configs[model_name]
    
    extra_config = dict(config.get("extra_config", {}) or {})
    effective_max_tokens = max_tokens
    model_max_tokens_cap = config.get("max_tokens_cap")
    if isinstance(model_max_tokens_cap, int) and model_max_tokens_cap > 0:
        effective_max_tokens = min(max_tokens, model_max_tokens_cap)

    base_runtime_config: Dict[str, Any] = {
        "temperature": temperature if config.get("allow_temperature", True) else None,
        "max_tokens": effective_max_tokens if config.get("allow_max_tokens", True) else None,
        "tools": tools,
    }
    base_runtime_config.update(extra_config)

    config_dict = _build_model_config_dict(
        model_name,
        config["platform"],
        base_runtime_config,
    )

    model_args = {
        "model_platform": config["platform"],
        "model_type": config["type"],
        "model_config_dict": config_dict,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    if "url" in config:
        model_args["url"] = config["url"]
    if "api_key" in config:
        api_key_val = os.getenv(config["api_key"])
        model_args["api_key"] = api_key_val
    if config.get("api_mode"):
        model_args["api_mode"] = config["api_mode"]

    model = ModelFactory.create(**model_args)

    _install_outbound_tool_schema_cleanup(
        model,
        model_name,
        strip_anthropic_strict_limit=config["platform"] == ModelPlatformType.ANTHROPIC,
    )

    if config["platform"] == ModelPlatformType.GEMINI:
        _install_gemini_compatibility_mode(model, model_name)

    if config.get("structured_output_mode") == "json_object":
        _install_json_object_structured_output_mode(model, model_name)

    if config.get("api_mode") == "responses":
        _install_responses_api_compatibility_mode(model, model_name)

    try:
        if hasattr(model, "_timeout") and timeout is not None:
            setattr(model, "_timeout", timeout)
        if hasattr(model, "_max_retries"):
            setattr(model, "_max_retries", max_retries)

        for client_attr in ("_client", "_async_client"):
            client = getattr(model, client_attr, None)
            if client is None:
                continue
            if timeout is not None:
                for timeout_attr in ("timeout", "_timeout"):
                    try:
                        setattr(client, timeout_attr, timeout)
                    except Exception:
                        pass
            for retry_attr in ("max_retries", "_max_retries"):
                try:
                    setattr(client, retry_attr, max_retries)
                except Exception:
                    pass
    except Exception as exc:
        logger.debug(f"Failed to enforce timeout/retry on model backend: {exc}")

    return model


def load_json(json_str, default_value={}, verbose=True):
    """
    Enhanced JSON loading function that handles various JSON format errors.
    
    Args:
        json_str (str): JSON string to parse
        default_value (Any, optional): Default value to return if parsing fails. Defaults to {}
        verbose (bool, optional): Whether to print detailed error messages. Defaults to True
    
    Returns:
        Any: Parsed JSON object, or default_value if parsing fails
    """
    if not json_str:
        if verbose:
            logger.warning("Empty JSON string provided")
        return default_value

    json_str = sanitize_llm_json_text(json_str)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        if verbose:
            logger.debug(f"Standard JSON parsing failed: {e}")
    
    try:
        import json5
        return json5.loads(json_str)
    except Exception as e:
        if verbose:
            logger.debug(f"JSON5 parsing failed: {e}")
    
    try:
        import demjson3
        return demjson3.decode(json_str, strict=False)
    except Exception as e:
        if verbose:
            logger.debug(f"demjson3 parsing failed: {e}")
    
    try:
        import json_repair
        repaired = json_repair.repair_json(json_str)
        return json.loads(repaired)
    except Exception as e:
        if verbose:
            logger.debug(f"JSON repair failed: {e}")
    
    if verbose:
        logger.warning(f"All JSON parsing attempts failed, returning default value")
    return default_value

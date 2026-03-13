"""Extract ToolIR instances from heterogeneous input payloads."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .models import (
    ToolIR,
    dedup_string_list,
    extract_text_blobs,
    sanitize_identifier,
)
from .schema import SchemaProcessor

# Signature: (system_prompt, user_prompt) -> parsed JSON dict
LLMCaller = Callable[[str, str], Dict[str, Any]]


class ToolDescriptionParser:
    """Payload -> List[ToolIR], using LLM only for gap-filling."""

    def __init__(self, schema_processor: SchemaProcessor, llm_caller: LLMCaller) -> None:
        self._schema = schema_processor
        self._llm = llm_caller

    def extract_tool_irs(self, payload: Any) -> List[ToolIR]:
        candidates: List[Tuple[Any, List[str]]] = []
        self._collect_candidates(payload, context_hints=[], out=candidates)
        if not candidates:
            raise ValueError("No tool candidates found in payload")

        tool_irs: List[ToolIR] = []
        for index, (candidate, hints) in enumerate(candidates, start=1):
            tool_irs.append(self._candidate_to_tool_ir(candidate, hints, index))

        # Deduplicate names
        dedup: Dict[str, ToolIR] = {}
        for tool in tool_irs:
            name = tool.name
            suffix = 2
            while name in dedup:
                name = f"{tool.name}_{suffix}"
                suffix += 1
            if name != tool.name:
                tool.name = name
            dedup[name] = tool
        return list(dedup.values())

    # ------------------------------------------------------------------
    # Candidate collection
    # ------------------------------------------------------------------

    def _collect_candidates(
        self,
        payload: Any,
        context_hints: List[str],
        out: List[Tuple[Any, List[str]]],
    ) -> None:
        if isinstance(payload, list):
            for item in payload:
                self._collect_candidates(item, context_hints, out)
            return

        if isinstance(payload, dict):
            local_hints = list(context_hints)
            local_hints.extend(
                dedup_string_list(
                    extract_text_blobs(
                        {
                            "question": payload.get("question"),
                            "prompt": payload.get("prompt"),
                            "instruction": payload.get("instruction"),
                            "task": payload.get("task"),
                        }
                    )
                )
            )
            identifier = payload.get("id")
            if isinstance(identifier, str) and identifier.strip():
                local_hints.append(f"id={identifier.strip()}")

            if self._looks_like_tool_object(payload):
                out.append((payload, dedup_string_list(local_hints)))

            for key in ("function", "functions", "tools", "tool_definitions", "apis"):
                if key in payload:
                    self._collect_candidates(payload[key], local_hints, out)
            return

        if isinstance(payload, str):
            text = payload.strip()
            if text:
                out.append((text, context_hints))

    # ------------------------------------------------------------------
    # Candidate -> ToolIR
    # ------------------------------------------------------------------

    def _candidate_to_tool_ir(
        self,
        candidate: Any,
        context_hints: List[str],
        index: int,
    ) -> ToolIR:
        if isinstance(candidate, str):
            return self._infer_tool_from_text(candidate, context_hints, index)

        if not isinstance(candidate, dict):
            raise ValueError(f"Unsupported tool candidate type: {type(candidate).__name__}")

        name = self._extract_tool_name(candidate)
        description = self._extract_tool_description(candidate)
        request_schema_raw, explicit_paths = self._extract_request_schema(candidate)
        output_hint_schema = self._extract_output_hint_schema(candidate)

        if not name or not description or request_schema_raw is None:
            inferred = self._infer_tool_from_text(
                json.dumps(candidate, ensure_ascii=False, indent=2),
                context_hints,
                index,
            )
            if not name:
                name = inferred.name
            if not description:
                description = inferred.description
            if request_schema_raw is None:
                request_schema_raw = inferred.request_schema
            if output_hint_schema is None:
                output_hint_schema = inferred.output_hint_schema

        if request_schema_raw is None:
            interface = self._infer_interface(name, description, context_hints)
            request_schema_raw = interface["parameters"]
            if output_hint_schema is None and isinstance(interface.get("output_schema"), dict):
                output_hint_schema = interface["output_schema"]

        request_schema, normalized_paths = self._schema.normalize(
            request_schema_raw,
            default_description="Request payload",
            pointer="",
            protect_explicit=True,
        )
        normalized_paths.update(explicit_paths)
        hints = dedup_string_list(context_hints)
        normalized_name = sanitize_identifier(name)
        if not normalized_name:
            raise ValueError(f"Invalid tool name extracted for candidate #{index}")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Missing tool description for candidate #{index}")

        return ToolIR(
            name=normalized_name,
            description=description.strip(),
            request_schema=request_schema,
            protected_request_paths=normalized_paths,
            output_hint_schema=output_hint_schema,
            context_hints=hints,
        )

    # ------------------------------------------------------------------
    # LLM-powered inference (gap-filling)
    # ------------------------------------------------------------------

    def _infer_tool_from_text(
        self,
        text: str,
        context_hints: List[str],
        index: int,
    ) -> ToolIR:
        system = "You extract structured tool definitions from natural language."
        prompt = f"""Extract one tool definition from text and return only JSON:
{{
  "name": "snake_case_name",
  "description": "one sentence",
  "parameters": {{
    "type": "object",
    "properties": {{}},
    "required": []
  }},
  "output_schema": {{}}
}}

Rules:
- Keep provided facts strict; infer only missing fields.
- parameters.type must be object.
- If output constraints are not explicit, infer a realistic schema.
- Do not include confidence or reasoning fields.

Text:
{text}

Context hints:
{json.dumps(context_hints, ensure_ascii=False)}
"""
        parsed = self._llm(system, prompt)
        name = parsed.get("name")
        description = parsed.get("description")
        params = parsed.get("parameters")
        output_schema = parsed.get("output_schema")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"LLM must provide non-empty tool name for candidate #{index}")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"LLM must provide non-empty tool description for candidate #{index}")
        if not isinstance(params, dict):
            raise ValueError(f"LLM must provide object 'parameters' for candidate #{index}")
        normalized_name = sanitize_identifier(name)
        if not normalized_name:
            raise ValueError(f"Invalid tool name generated by LLM for candidate #{index}: {name!r}")

        request_schema, protected = self._schema.normalize(
            params,
            default_description="Request payload",
            pointer="",
            protect_explicit=True,
        )
        return ToolIR(
            name=normalized_name,
            description=description.strip(),
            request_schema=request_schema,
            protected_request_paths=protected,
            output_hint_schema=output_schema if isinstance(output_schema, dict) else None,
            context_hints=dedup_string_list(context_hints),
        )

    def _infer_interface(
        self,
        name: str,
        description: str,
        context_hints: List[str],
    ) -> Dict[str, Any]:
        system = "You infer missing function interfaces from concise descriptions."
        prompt = f"""Return JSON only:
{{
  "parameters": {{
    "type": "object",
    "properties": {{}},
    "required": []
  }},
  "output_schema": {{}}
}}

Function name: {name}
Description: {description}
Context hints: {json.dumps(context_hints, ensure_ascii=False)}

Rules:
- Preserve realism; keep parameters minimal.
- If information is absent, use optional parameters instead of hard constraints.
- No confidence fields, no reasoning fields.
"""
        parsed = self._llm(system, prompt)
        params = parsed.get("parameters")
        if not isinstance(params, dict):
            raise ValueError(f"LLM must return object 'parameters' for inferred interface of {name}")
        return parsed

    # ------------------------------------------------------------------
    # Static extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_tool_object(obj: Dict[str, Any]) -> bool:
        name = ToolDescriptionParser._extract_tool_name(obj)
        if not name:
            return False
        return any(
            key in obj
            for key in (
                "description",
                "summary",
                "docstring",
                "parameters",
                "inputs",
                "input_schema",
                "args",
                "output",
                "response",
                "returns",
            )
        )

    @staticmethod
    def _extract_tool_name(obj: Dict[str, Any]) -> str:
        for key in ("name", "function_name", "operationId", "tool_name"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return sanitize_identifier(value.strip())
        return ""

    @staticmethod
    def _extract_tool_description(obj: Dict[str, Any]) -> str:
        for key in ("description", "summary", "docstring", "purpose"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _extract_request_schema(
        self,
        obj: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Set[str]]:
        schema_keys = ("parameters", "inputs", "input_schema", "arguments_schema", "args")
        for key in schema_keys:
            raw = obj.get(key)
            if isinstance(raw, dict):
                if "type" in raw or "properties" in raw or "required" in raw:
                    return raw, set()
                if "properties" in raw.get("parameters", {}):
                    return raw["parameters"], set()
            if isinstance(raw, list):
                schema = self._schema_from_parameter_list(raw)
                return schema, set()
        return None, set()

    def _schema_from_parameter_list(self, parameters: List[Any]) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for entry in parameters:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            prop_schema, _ = self._schema.normalize(
                entry.get("schema", entry),
                default_description=f"Parameter {name}",
                pointer="",
                protect_explicit=True,
            )
            properties[name] = prop_schema
            if entry.get("required", False):
                required.append(name)
        return {
            "type": "object",
            "properties": properties,
            "required": dedup_string_list(required),
            "additionalProperties": False,
            "description": "Request payload",
        }

    @staticmethod
    def _extract_output_hint_schema(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for key in (
            "output_schema",
            "response_schema",
            "returns",
            "return",
            "output",
            "response",
            "result_schema",
        ):
            value = obj.get(key)
            if isinstance(value, dict):
                return value
        return None

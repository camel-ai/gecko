"""LLM-powered endpoint enrichment and API metadata generation."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import (
    EndpointEnrichment,
    ToolIR,
    dedup_string_list,
    humanize_identifier,
    sanitize_identifier,
)
from .schema import SchemaProcessor

logger = logging.getLogger(__name__)

LLMCaller = Callable[[str, str], Dict[str, Any]]


class EndpointEnricher:
    """Builds enriched OpenAPI endpoints via LLM calls."""

    def __init__(self, schema_processor: SchemaProcessor, llm_caller: LLMCaller) -> None:
        self._schema = schema_processor
        self._llm = llm_caller

    def build_endpoint(
        self,
        tool: ToolIR,
        *,
        include_tool_state: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Build an OpenAPI endpoint and its tool-state metadata.

        Returns:
            (endpoint_dict, tool_state_dict)
        """
        seed_success_schema: Optional[Dict[str, Any]] = None
        if isinstance(tool.output_hint_schema, dict) and tool.output_hint_schema:
            seed_success_schema = self._normalize_success_schema(tool.output_hint_schema, tool)

        endpoint: Dict[str, Any] = {
            "operationId": tool.name,
            "summary": humanize_identifier(tool.name),
            "description": tool.description or f"Execute {tool.name}.",
            "requestBody": {
                "required": bool(tool.request_schema.get("properties")),
                "content": {
                    "application/json": {
                        "schema": deepcopy(tool.request_schema),
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Operation result",
                    "content": {
                        "application/json": {"schema": {}}
                    },
                }
            },
        }

        enrichment = self._enrich(tool, seed_success_schema)
        endpoint["summary"] = enrichment.summary
        endpoint["description"] = enrichment.description

        schema_updates = self._schema.sanitize_fragment(enrichment.request_schema_updates)
        request_schema = endpoint["requestBody"]["content"]["application/json"]["schema"]
        endpoint["requestBody"]["content"]["application/json"]["schema"] = (
            self._schema.merge_with_protection(
                request_schema,
                schema_updates,
                tool.protected_request_paths,
            )
        )

        success_schema = self._normalize_success_schema(enrichment.success_schema, tool)
        endpoint["responses"]["200"]["content"]["application/json"]["schema"] = success_schema

        tool_state: Dict[str, Any] = {}
        if include_tool_state:
            if enrichment.validation_rules:
                tool_state["validation_rules"] = enrichment.validation_rules
            if enrichment.behavior_hints:
                tool_state["behavior_hints"] = enrichment.behavior_hints

        return endpoint, tool_state

    def generate_api_metadata(
        self,
        tools: List[ToolIR],
        requested_api_name: Optional[str],
    ) -> Tuple[str, str]:
        requested = requested_api_name.strip() if isinstance(requested_api_name, str) else ""
        if requested:
            title = self._normalize_api_title(requested)
            description = self._generate_api_description(tools, title)
            return title, description

        summary_lines = [f"- {tool.name}: {tool.description}" for tool in tools[:20]]
        prompt = f"""Generate API metadata for this tool collection.
Tools:
{chr(10).join(summary_lines)}

Return JSON only:
{{
  "title": "compact_api_title",
  "description": "40-100 word paragraph"
}}

Constraints:
- title must be concise, human-readable, and toolkit-level.
- description should summarize scope and usage style.
- no markdown.
"""
        parsed = self._llm(
            "You generate OpenAPI metadata for tool collections.",
            prompt,
        )
        title = self._normalize_api_title(parsed.get("title"))
        description = parsed.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("LLM must provide non-empty API metadata description")
        return title, " ".join(description.split())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enrich(
        self,
        tool: ToolIR,
        seed_success_schema: Optional[Dict[str, Any]],
    ) -> EndpointEnrichment:
        prompt_payload = {
            "name": tool.name,
            "description": tool.description,
            "request_schema": tool.request_schema,
            "output_hint_schema": tool.output_hint_schema,
            "context_hints": tool.context_hints,
            "seed_success_schema": seed_success_schema,
        }
        system = "You enrich OpenAPI endpoints while preserving explicit input constraints."
        prompt = f"""Return JSON only:
{{
  "summary": "string",
  "description": "string",
  "request_schema_updates": {{}},
  "success_schema": {{}},
  "validation_rules": [{{"condition":"...", "result":{{...}}}}],
  "behavior_hints": ["..."]
}}

Rules:
- Keep explicit request facts unchanged (names, required fields, explicit types/defaults/enums/descriptions).
- request_schema_updates must only contain additive refinements for unconstrained fields.
- CRITICAL: Do NOT add enum constraints to request parameters unless the original specification explicitly lists allowed values. If the description only mentions a default or a few examples, those are hints, not an exhaustive enum.
- CRITICAL: Do NOT add pattern or format constraints (e.g., date formats, regex) to string parameters that accept natural language or free-text input. Only add pattern/format if the original specification explicitly requires a specific format.
- CRITICAL: Do NOT add minLength, maxLength, minimum, or maximum constraints that are not in the original specification unless they are obvious safety bounds (e.g., array maxItems for sanity).
- CRITICAL: Do NOT add new properties/parameters in request_schema_updates that do not exist in the original specification. Only refine descriptions or add constraints to EXISTING properties.
- CRITICAL: Do NOT fabricate unit specifications (e.g., "in cents", "in smallest unit", "in milliseconds") unless the original specification explicitly states the unit. If the unit is not specified, describe the parameter without assuming a unit system.
- CRITICAL: Do NOT rewrite or rephrase any description that is already present in request_schema properties — existing descriptions must be copied verbatim into request_schema_updates if you reference them at all.
- success_schema must be executable OpenAPI schema with no $ref.
- Do not include error wrapper/object in success_schema; return only the successful payload structure.
- Do not output confidence or reasoning fields.
- Keep summary concise and description simulation-focused.
- success_schema is mandatory.
- Unless explicitly required by the provided specification, responses MUST NOT include input-echo or debug fields (e.g., "inputs").

Input:
{json.dumps(prompt_payload, indent=2, ensure_ascii=False)}
"""
        parsed = self._llm(system, prompt)
        summary = parsed.get("summary")
        description = parsed.get("description")
        request_schema_updates = parsed.get("request_schema_updates")
        success_schema = parsed.get("success_schema")
        validation_rules = parsed.get("validation_rules")
        behavior_hints = parsed.get("behavior_hints")

        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(f"LLM must provide non-empty summary for tool '{tool.name}'")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"LLM must provide non-empty description for tool '{tool.name}'")
        if not isinstance(success_schema, dict):
            raise ValueError(f"LLM must provide success_schema object for tool '{tool.name}'")
        if request_schema_updates is None:
            request_schema_updates = {}
        if not isinstance(request_schema_updates, dict):
            raise ValueError(f"LLM must provide object request_schema_updates for tool '{tool.name}'")

        valid_rules: List[Dict[str, Any]] = []
        if isinstance(validation_rules, list):
            for rule in validation_rules:
                if isinstance(rule, dict) and rule:
                    valid_rules.append(rule)

        return EndpointEnrichment(
            summary=summary.strip(),
            description=description.strip(),
            request_schema_updates=request_schema_updates,
            success_schema=success_schema,
            validation_rules=valid_rules,
            behavior_hints=dedup_string_list(behavior_hints if isinstance(behavior_hints, list) else []),
        )

    def _normalize_success_schema(self, raw_schema: Any, tool: ToolIR) -> Dict[str, Any]:
        if not isinstance(raw_schema, dict):
            raise ValueError(f"Missing valid success schema for tool '{tool.name}'")
        sanitized = self._schema.sanitize_fragment(raw_schema)
        normalized, _ = self._schema.normalize(
            sanitized,
            default_description="Successful response payload",
            pointer="",
            protect_explicit=False,
        )
        return normalized

    def _generate_api_description(self, tools: List[ToolIR], api_name: str) -> str:
        summary_lines = [f"- {tool.name}: {tool.description}" for tool in tools[:20]]
        prompt = f"""Write one concise paragraph for an API collection.
API name: {api_name}
Tools:
{chr(10).join(summary_lines)}

Requirements:
- 40-100 words.
- Focus on overall scope and usage style.
- No markdown or list in output.
"""
        parsed = self._llm(
            "You produce concise API descriptions.",
            f'Return JSON only: {{"description":"..."}}\n\n{prompt}',
        )
        description = parsed.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("LLM must provide non-empty API description")
        return " ".join(description.split())

    @staticmethod
    def _normalize_api_title(raw_title: Any) -> str:
        if not isinstance(raw_title, str):
            raise ValueError("API title must be a string")
        cleaned = sanitize_identifier(raw_title.strip())
        if len(cleaned) < 2:
            raise ValueError(f"API title is too short after normalization: {raw_title!r}")
        if len(cleaned) > 64:
            raise ValueError(f"API title is too long after normalization: {cleaned!r}")
        return cleaned

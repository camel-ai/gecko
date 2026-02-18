"""
Response Generator - Generates mock API responses based on OpenAPI schemas.

This module uses LLM to generate realistic API responses that conform to OpenAPI
schemas while validating against the current system state.
"""

import json
import logging
from datetime import datetime
from copy import deepcopy
from typing import Any, Dict, Optional

from camel.agents import ChatAgent
from fastapi import Request

from .request_details import RequestDetails
from .config_updater import update_state
from .schema_utils import (
    resolve_refs,
    extract_parameter_descriptions,
    extract_response_descriptions,
    extract_toolkit_info,
)
from utils.model_utils import create_model
import json_repair

logger = logging.getLogger(__name__)


RESPONSE_SYSTEM_PROMPT = """
You are an API simulation engine that generates JSON responses strictly following OpenAPI 3.1 schemas.

CORE PRINCIPLES:

1. **Schema Adherence** — Always match the schema exactly (structure, names, types, formats, required fields).

2. **System State Validation** — Validate all operations against the current System State (including toolkit-specific runtime_state). If requirements are not met, return an error.

3. **Semantic Validation** — Referenced entities must exist, be accessible, and be valid for the requested operation.

4. **Reasonable Defaults** — If schema-required values are missing from System State, synthesize realistic values (UUIDs, ISO 8601 timestamps, tokens, etc.) that do not contradict System State.

5. **State Consistency** — Reflect mutations consistently; subsequent operations must observe prior successful changes.

6. **No Extra Rules** — Do not invent constraints beyond the tool definition and System State.

STATE PRIORITY RULES:

- Apply this priority when deciding values and constraints:
  1) Current System State (latest runtime/config state)
  2) Request + Operation context
  3) Schema Default State (`info.x-default-state`) where operation-specific defaults override global defaults
  4) Reasonable synthesized values
- If Current System State conflicts with `x-default-state`, trust Current System State.
- If `x-default-state` provides fixed defaults or validation-like constraints, follow them unless overridden by higher-priority state.
- Canonical business truth is top-level toolkit state (`<ToolkitName>.*`); `runtime_state.toolkits.<ToolkitName>` is runtime context only.
- If the same business key appears in both top-level toolkit state and runtime_state with conflicting values, trust top-level toolkit state.


VALIDATION GUIDELINES:

7. **Exact Matching** — Entity names, identifiers, and paths must match EXACTLY. Similar names are NOT the same (e.g., 'user_123' ≠ 'user_124', 'notes.md' ≠ 'note.md').

8. **Navigating Nested Structures** — When checking if an entity exists in a nested structure:
   a) Identify the relevant path/location from runtime_state (may be nested under runtime_state.toolkits.<ToolkitName> or flat)
   b) Parse the path into components if needed
   c) Navigate step-by-step through the structure, following the nesting pattern (e.g., parent → .contents → child → .contents)
   d) Check existence at the final level only - do not assume entities from parent/sibling/child levels

9. **Scope Boundaries** — Operations with scope constraints (e.g., "current directory", "active workspace", "selected items") can ONLY access direct members of that scope, not nested or related scopes.

10. **Case Sensitivity** — All identifiers, names, and keys are case-sensitive unless explicitly stated otherwise in the tool definition.

11. **Schema Branch Selection** — If the response schema uses oneOf/anyOf, choose EXACTLY ONE branch and output a concrete instance of that branch. Never output the schema itself (no oneOf/anyOf/type/properties/description in the response).
12. **Condition Evaluation Discipline** — Treat validation rules as executable conditions:
   - Trigger an error branch only when the condition is positively true from request + state + schema defaults.
   - If a condition is not provably true, do not assume failure.
   - If all known failure conditions are false, prefer the success branch.
13. **Use Called-Method Static Data** — When x-default-state provides called_method_static_data/static_data (e.g., canonical lists, lookup tables), use it as authoritative for validations instead of guessing.
14. **Auth/Login Continuity** — After a successful auth/login response, treat the toolkit as authenticated for subsequent operations unless a later explicit logout/failure changes that state.
15. **Write Success Consistency** — For successful write operations, ensure response semantics are consistent with persisted canonical state mutations (e.g., created records are retrievable by returned IDs).

Example: For a file system toolkit with current directory "/root/alex/workspace/Projects", checking if "notes.md" exists:
- Navigate: GorillaFileSystem.root → alex.contents → workspace.contents → Projects.contents
- Check: "notes.md" exists as a direct key in Projects.contents
- Scope: Only direct children of Projects are accessible, not files in parent (workspace) or subdirectories
"""


CONTEXT_EXTRACTION_SYSTEM_PROMPT = """
You are a state analysis expert. Your task is to extract and summarize the relevant system state for an API operation.

Your role:
1. Analyze the current system state from the configuration
2. Identify what parts of the state are relevant to this specific operation
3. Extract and clearly present this information
4. DO NOT generate responses or results - only analyze and extract state

Key principles:
- Prefer canonical top-level toolkit state for business fields; use runtime_state only for transient context.
- If top-level and runtime_state conflict on business fields (e.g., authenticated flags, counters, records), treat top-level as authoritative.
- For operations that access/modify resources, identify those resources' current state
- For operations with source/destination, check BOTH locations
- Always specify if collections/directories are EMPTY or list their contents
- Include any constraints or validation rules from the operation description
- Focus on what EXISTS vs what DOESN'T EXIST in the relevant scope

Output format:
## Relevant System State
[Extracted configuration relevant to this operation]

## Operation Constraints
[Any constraints or rules from the operation description]

## State Analysis
[Your analysis of the current state relevant to this operation]

Be thorough but concise. Extract ONLY what's needed for this specific operation.
"""


class ResponseGenerator:
    """Generates mock responses based on OpenAPI schema using LLM."""

    def __init__(self, response_model: str = "gpt-5-mini", state_model: str = "gpt-5-mini"):
        """Initialize the response generator with configurable models.

        Args:
            response_model: LLM model for response generation (default: gpt-5-mini)
            state_model: LLM model for state update (default: gpt-5-mini)
        """
        self.response_model = response_model
        self.state_model = state_model
        self.system_prompt = RESPONSE_SYSTEM_PROMPT

    @staticmethod
    def extract_toolkit_runtime_state(config: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extract runtime state for the specific toolkit from config.

        Supports both new nested structure (runtime_state.toolkits.<name>) and
        old flat structure (runtime_state) for backward compatibility.

        Args:
            config: Full system configuration
            schema: OpenAPI schema containing toolkit info

        Returns:
            Dictionary of runtime state variables for this toolkit
        """
        runtime_state = config.get('runtime_state', {})

        # Try to extract toolkit name from schema
        toolkit_name = None
        if schema and isinstance(schema, dict):
            info = schema.get('info', {})
            toolkit_name = info.get('title', '')  # e.g., "GorillaFileSystem", "TwitterAPI"

        # Try new nested structure first
        if 'toolkits' in runtime_state and toolkit_name:
            toolkit_state = runtime_state.get('toolkits', {}).get(toolkit_name, {})
            if toolkit_state:
                return toolkit_state

        # Fallback to old flat structure for backward compatibility
        if 'toolkits' not in runtime_state:
            return runtime_state

        return {}

    @staticmethod
    def _operation_description_map(schema: Dict[str, Any]) -> Dict[str, str]:
        """Build operationId -> description map from schema paths."""
        descriptions: Dict[str, str] = {}
        if not isinstance(schema, dict):
            return descriptions
        paths = schema.get("paths", {})
        if not isinstance(paths, dict):
            return descriptions

        for methods in paths.values():
            if not isinstance(methods, dict):
                continue
            for operation in methods.values():
                if not isinstance(operation, dict):
                    continue
                operation_id = operation.get("operationId")
                if not isinstance(operation_id, str) or not operation_id:
                    continue
                description = operation.get("description")
                descriptions[operation_id] = description if isinstance(description, str) else ""
        return descriptions

    @staticmethod
    def _toolkit_name_from_schema(schema: Dict[str, Any]) -> str:
        if not isinstance(schema, dict):
            return ""
        info = schema.get("info", {})
        if not isinstance(info, dict):
            return ""
        title = info.get("title")
        return title.strip() if isinstance(title, str) else ""

    def _extract_schema_default_state(
        self,
        schema: Dict[str, Any],
        operation: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract operation-filtered x-default-state in NEW format only.

        Expected schema shape:
        info.x-default-state = {
          "global": {"runtime_defaults": {...}},
          "tools": {"<operationId>": {...}}
        }
        """
        if not isinstance(schema, dict):
            return {}
        info = schema.get("info", {})
        if not isinstance(info, dict):
            return {}
        x_default_state = info.get("x-default-state")
        if not isinstance(x_default_state, dict):
            return {}

        global_block = x_default_state.get("global")
        tools_block = x_default_state.get("tools")
        if not isinstance(global_block, dict) or not isinstance(tools_block, dict):
            return {}

        runtime_defaults = global_block.get("runtime_defaults")
        global_runtime_defaults = runtime_defaults if isinstance(runtime_defaults, dict) else {}

        operation_defaults: Dict[str, Any] = {}
        operation_id = operation.get("operationId") if isinstance(operation, dict) else ""
        if isinstance(operation_id, str) and operation_id:
            tool_entry = tools_block.get(operation_id)
            if isinstance(tool_entry, dict):
                static_data = tool_entry.get("static_data")
                if isinstance(static_data, dict) and static_data:
                    operation_defaults["static_data"] = static_data

                validation_rules = tool_entry.get("validation_rules")
                if isinstance(validation_rules, list) and validation_rules:
                    operation_defaults["validation_rules"] = validation_rules

                state_effects = tool_entry.get("state_effects")
                if isinstance(state_effects, list) and state_effects:
                    operation_defaults["state_effects"] = state_effects

                state_effects_on_success = tool_entry.get("state_effects_on_success")
                if isinstance(state_effects_on_success, list) and state_effects_on_success:
                    operation_defaults["state_effects_on_success"] = state_effects_on_success

                state_effects_on_error = tool_entry.get("state_effects_on_error")
                if isinstance(state_effects_on_error, list) and state_effects_on_error:
                    operation_defaults["state_effects_on_error"] = state_effects_on_error

                state_effects_always = tool_entry.get("state_effects_always")
                if isinstance(state_effects_always, list) and state_effects_always:
                    operation_defaults["state_effects_always"] = state_effects_always

                behavior_hints = tool_entry.get("behavior_hints")
                if isinstance(behavior_hints, list) and behavior_hints:
                    operation_defaults["behavior_hints"] = behavior_hints

                success_string_templates = tool_entry.get("success_string_templates")
                if isinstance(success_string_templates, dict) and success_string_templates:
                    operation_defaults["success_string_templates"] = success_string_templates

                called_method_static_data = tool_entry.get("called_method_static_data")
                if isinstance(called_method_static_data, dict) and called_method_static_data:
                    operation_defaults["called_method_static_data"] = called_method_static_data

                # Only pass method call descriptions (no other method metadata).
                method_calls = tool_entry.get("method_calls")
                if isinstance(method_calls, list) and method_calls:
                    op_desc_map = self._operation_description_map(schema)
                    method_call_descriptions = []
                    for method_name in method_calls:
                        if isinstance(method_name, str):
                            desc = op_desc_map.get(method_name, "").strip()
                            if desc:
                                method_call_descriptions.append(desc)
                    if method_call_descriptions:
                        operation_defaults["method_call_descriptions"] = method_call_descriptions

        result: Dict[str, Any] = {}
        if global_runtime_defaults:
            result["global_runtime_defaults"] = global_runtime_defaults
        if operation_defaults:
            result["operation_defaults"] = operation_defaults
        return result

    def _materialize_missing_toolkit_state(
        self,
        current_state: Dict[str, Any],
        schema: Dict[str, Any],
        schema_default_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """If toolkit top-level state is missing, seed it from schema defaults."""
        if not isinstance(current_state, dict):
            return {}
        toolkit_name = self._toolkit_name_from_schema(schema)
        if not toolkit_name:
            return current_state

        defaults = schema_default_state.get("global_runtime_defaults")
        if not isinstance(defaults, dict) or not defaults:
            return current_state

        existing = current_state.get(toolkit_name)
        if isinstance(existing, dict):
            merged = deepcopy(current_state)
            toolkit_state = merged.get(toolkit_name)
            if not isinstance(toolkit_state, dict):
                merged[toolkit_name] = deepcopy(defaults)
                return merged
            for key, value in defaults.items():
                if key not in toolkit_state:
                    toolkit_state[key] = deepcopy(value)
            return merged

        merged = deepcopy(current_state)
        merged[toolkit_name] = deepcopy(defaults)
        logger.info(
            "[RESPONSE] Seeded missing toolkit state from x-default-state defaults: %s",
            toolkit_name,
        )
        return merged

    @staticmethod
    def _format_schema_default_state(schema_default_state: Dict[str, Any]) -> str:
        if not schema_default_state:
            return "(No operation-relevant x-default-state found)"
        try:
            return json.dumps(schema_default_state, indent=2, ensure_ascii=False)
        except Exception:
            return str(schema_default_state)

    async def generate_response(
        self,
        response_schema: Dict[str, Any],
        schema: Dict[str, Any],
        request: Request,
        operation: Dict[str, Any] = None,
    ) -> Any:
        """Generate a mock response based on the response schema.

        Args:
            response_schema: Expected response schema from OpenAPI spec
            schema: Full OpenAPI schema document
            request: FastAPI request object
            operation: OpenAPI operation details

        Returns:
            Generated response matching the schema
        """
        try:
            state_history, current_state, session_id = self._load_session_state(request)
            request_info = await self._extract_request_info(request)
            toolkit_info = extract_toolkit_info(schema)
            param_info_for_user = self._build_param_info(operation)
            schema_default_state = self._extract_schema_default_state(schema, operation)
            schema_default_state_text = self._format_schema_default_state(schema_default_state)
            effective_state = self._materialize_missing_toolkit_state(
                current_state=current_state,
                schema=schema,
                schema_default_state=schema_default_state,
            )

            context_info = self._build_context_info(
                current_state=effective_state,
                operation=operation,
                request_info=request_info,
                toolkit_info=toolkit_info,
                schema=schema,
                schema_default_state=schema_default_state,
                session_id=session_id,
            )

            enhanced_system_prompt = f"""{self.system_prompt}

## Toolkit Information
{toolkit_info}

IMPORTANT: You will receive the relevant system state in the user message. Base your response on that state.
"""

            resolved_schema = resolve_refs(response_schema, schema)
            tool_definition = self._build_tool_definition(operation, request)
            user_message = self._build_user_message(
                context_info=context_info,
                tool_definition=tool_definition,
                param_info=param_info_for_user,
                request_info=request_info,
                resolved_schema=resolved_schema,
                schema_default_state_text=schema_default_state_text,
            )

            response_str = self._call_response_llm(
                enhanced_system_prompt=enhanced_system_prompt,
                user_message=user_message,
                session_id=session_id,
            )

            result = self._parse_response(response_str)

            if len(state_history) > 0:
                tool_descriptions = self._build_tool_descriptions(
                    operation,
                    tool_definition,
                    request,
                    schema=schema,
                )
                tool_calls = self._build_tool_calls(operation, request_info, request, result)

                previous_state = effective_state if isinstance(effective_state, dict) else (
                    state_history[-1] if len(state_history) > 0 else {}
                )
                update_state(
                    previous_state=previous_state,
                    tool_calls=tool_calls,
                    tool_descriptions=tool_descriptions,
                    session_id=session_id,
                    state_model=self.state_model,
                )

            return result

        except Exception as e:
            logger.exception(f"Error generating response: {str(e)}")
            raise

    def _load_session_state(self, request: Request) -> tuple[list, Dict[str, Any], Optional[str]]:
        """Load session state and ensure there is at least one state snapshot."""
        state_history = getattr(request.state, "session_state", [])
        session_id = request.headers.get("X-Session-ID")
        if len(state_history) == 0 and session_id:
            from ..handlers.session_handler import session_handler
            session_handler.add_to_state(session_id, {})
            state_history = [{}]
        current_state = state_history[-1] if len(state_history) > 0 else {}
        return state_history, current_state, session_id

    async def _extract_request_info(self, request: Request) -> Dict[str, Any]:
        """Extract and sanitize request info for prompt usage."""
        request_info = await RequestDetails.extract(request)
        request_info.pop("headers", None)
        request_info.pop("client", None)
        request_info.pop("path", None)
        return request_info

    def _build_param_info(self, operation: Optional[Dict[str, Any]]) -> str:
        """Build parameter + response field descriptions string for the prompt."""
        if not operation:
            return ""
        param_desc = extract_parameter_descriptions(operation)
        resp_desc = extract_response_descriptions(operation)
        return f"{param_desc}{resp_desc}" if (param_desc or resp_desc) else ""

    def _build_context_info(
        self,
        current_state: Dict[str, Any],
        operation: Optional[Dict[str, Any]],
        request_info: Dict[str, Any],
        toolkit_info: str,
        schema: Dict[str, Any],
        schema_default_state: Dict[str, Any],
        session_id: Optional[str],
    ) -> str:
        if not current_state:
            if schema_default_state:
                return (
                    "## Relevant System State\n"
                    "(Empty runtime state; use Schema Default State below as fallback constraints)\n"
                )
            return "## Relevant System State\n(Empty state - no constraints)\n"
        return self.extract_operation_context_with_llm(
            config=current_state,
            operation=operation,
            request_info=request_info,
            toolkit_info=toolkit_info,
            schema=schema,
            model=self.response_model,
            session_id=session_id,
        )

    def _build_tool_definition(self, operation: Optional[Dict[str, Any]], request: Request) -> Dict[str, Any]:
        if not operation:
            return {}
        return {
            "operation_id": operation.get("operationId", ""),
            "summary": operation.get("summary", ""),
            "description": operation.get("description", ""),
            "method": getattr(request, "method", "").upper() if hasattr(request, "method") else "",
            "path": request.url.path if hasattr(request, "url") else "",
        }

    def _build_user_message(
        self,
        context_info: str,
        tool_definition: Dict[str, Any],
        param_info: str,
        request_info: Dict[str, Any],
        resolved_schema: Dict[str, Any],
        schema_default_state_text: str,
    ) -> str:
        return f"""
{context_info}

## Operation Being Performed
**Operation ID**: {tool_definition.get('operation_id', 'Unknown')}
**Summary**: {tool_definition.get('summary', 'No summary available')}
**Description**: {tool_definition.get('description', 'No description available')}
**Method**: {tool_definition.get('method', '')}
**Path**: {tool_definition.get('path', '')}
{param_info}

## Schema Default State (x-default-state, operation-filtered)
{schema_default_state_text}

## Actual Request
            {request_info.get('body', '{}')}


Return a pure JSON object matching the response schema.

## Expected Response Schema
{json.dumps(resolved_schema, indent=2)}
"""

    def _call_response_llm(
        self,
        enhanced_system_prompt: str,
        user_message: str,
        session_id: Optional[str],
    ) -> str:
        agent = ChatAgent(
            enhanced_system_prompt,
            model=create_model(self.response_model, max_tokens=16384, temperature=0.001),
        )

        _rt0 = datetime.now()
        logger.debug("[RESPONSE] LLM START (model=%s)", self.response_model)
        response = agent.step(user_message)
        _rt1 = datetime.now()
        logger.debug(
            "[RESPONSE] LLM END (elapsed=%.3fs, model=%s)",
            (_rt1 - _rt0).total_seconds(),
            self.response_model,
        )

        try:
            if session_id:
                from ..utils.llm_usage import extract_token_usage

                usage = extract_token_usage(response)
                from ..handlers.session_handler import session_handler

                session_handler.record_llm_usage(
                    session_id,
                    category="response_generation",
                    model=str(self.response_model),
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                )
        except Exception as usage_exc:
            logger.warning(f"Failed to record response generation token usage: {usage_exc}")

        response_str = response.msg.content
        return self._strip_wrappers(response_str)

    def _strip_wrappers(self, response_str: str) -> str:
        if response_str.startswith("```json"):
            response_str = response_str[len("```json"):]
        if response_str.endswith("```"):
            response_str = response_str[:-len("```")]
        if response_str.startswith("\n") and response_str.endswith("\n"):
            response_str = response_str[1:-1]
        if "</think>" in response_str:
            response_str = response_str.split("</think>")[-1]
        return response_str

    def _parse_response(self, response_str: str) -> Any:
        """Parse model output as generic JSON value (object/null/string/etc.)."""
        cleaned = response_str.strip()
        try:
            return json_repair.loads(cleaned)
        except Exception:
            # Keep backward compatibility for plain text outputs.
            return {"message": cleaned}

    def _build_tool_descriptions(
        self,
        operation: Optional[Dict[str, Any]],
        tool_definition: Dict[str, Any],
        request: Request,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not operation:
            return None
        canonical_name = self._canonical_operation_name(operation, request)
        enriched_definition = dict(tool_definition)

        toolkit_info: Dict[str, Any] = {}
        if isinstance(schema, dict):
            info = schema.get("info")
            if isinstance(info, dict):
                title = info.get("title")
                description = info.get("description")
                if isinstance(title, str) and title:
                    toolkit_info["name"] = title
                if isinstance(description, str) and description:
                    toolkit_info["description"] = description

        if toolkit_info:
            enriched_definition["toolkit"] = toolkit_info

        if isinstance(schema, dict):
            default_state_hint = self._extract_schema_default_state(schema, operation)
            if isinstance(default_state_hint, dict):
                operation_defaults = default_state_hint.get("operation_defaults")
                if isinstance(operation_defaults, dict) and operation_defaults:
                    enriched_definition["state_hints"] = operation_defaults

        return {canonical_name: enriched_definition}

    def _build_tool_calls(
        self,
        operation: Optional[Dict[str, Any]],
        request_info: Dict[str, Any],
        request: Request,
        result: Any,
    ) -> list[Dict[str, Any]]:
        if not operation:
            return []
        canonical_name = self._canonical_operation_name(operation, request)
        arguments = self._extract_arguments(request_info)
        return [
            {
                "name": canonical_name,
                "arguments": arguments,
                "result": result,
            }
        ]

    def _canonical_operation_name(self, operation: Dict[str, Any], request: Request) -> str:
        operation_id = operation.get("operationId")
        path_parts = request.url.path.split('/') if hasattr(request, "url") else []
        fallback_function_name = path_parts[-1] if len(path_parts) >= 1 else ""
        return operation_id or fallback_function_name

    def _extract_arguments(self, request_info: Dict[str, Any]) -> Dict[str, Any]:
        arguments: Dict[str, Any] = {}
        body_raw = request_info.get('body')
        if body_raw:
            try:
                body_obj = json.loads(body_raw)
                if (
                    isinstance(body_obj, dict)
                    and 'requestBody' in body_obj
                    and isinstance(body_obj['requestBody'], dict)
                ):
                    body_obj = body_obj['requestBody']
                if isinstance(body_obj, dict):
                    arguments.update(body_obj)
            except json.JSONDecodeError:
                pass
        if request_info.get('query_params') and isinstance(request_info['query_params'], dict):
            arguments.update(request_info['query_params'])
        if request_info.get('path_params') and isinstance(request_info['path_params'], dict):
            arguments.update(request_info['path_params'])
        return arguments

    def extract_operation_context_with_llm(
        self,
        config: Dict[str, Any],
        operation: Dict[str, Any],
        request_info: Dict[str, Any],
        toolkit_info: str = "",
        schema: Dict[str, Any] = None,
        model: Optional[str] = None,
        session_id: str | None = None,
    ) -> str:
        """Extract relevant context and state for the operation using LLM.

        Uses a specialized LLM agent to analyze the full system configuration and
        extract only the parts relevant to the current operation.

        Args:
            config: Full system configuration
            operation: Operation details from OpenAPI spec
            request_info: Request details including body and parameters
            toolkit_info: Toolkit description string
            schema: Full OpenAPI schema
            model: LLM model to use for extraction. If None, use self.response_model.

        Returns:
            Formatted string with relevant system state and analysis
        """
        try:
            param_descriptions = extract_parameter_descriptions(operation) if operation else ""
            response_desc = extract_response_descriptions(operation) if operation else ""

            extraction_query = f"""
## Toolkit Information
{toolkit_info}

## Operation Details
- Operation ID: {operation.get('operationId', 'unknown') if operation else 'unknown'}
- Summary: {operation.get('summary', '') if operation else ''}
- Description: {operation.get('description', '') if operation else ''}
{param_descriptions}
{response_desc}

## Request Information
- Method: {operation.get('method', request_info.get('method', '')) if operation else ''}
- Path: {operation.get('path', request_info.get('path', '')) if operation else ''}
- Actual Request Body: {request_info.get('body', '{}') if request_info else '{}'}

## Full System Configuration
{json.dumps(config, indent=2) if isinstance(config, dict) else str(config)}

## Task
Extract and summarize the parts of the system configuration that are relevant to this operation.

Consider:
1. What is the current state/context (from runtime_state)?
2. What resources does this operation need to access or modify?
3. For operations with source/destination: what exists at each location?
4. What constraints apply based on the operation description?
5. Are there any potential conflicts or validation issues?

Provide a clear, structured summary of the relevant state. DO NOT generate a response. Only analyze and extract.
"""

            model_name = model or self.response_model
            extract_model = create_model(model_name, max_tokens=16384, temperature=0.001)
            extract_agent = ChatAgent(CONTEXT_EXTRACTION_SYSTEM_PROMPT, model=extract_model)
            response = extract_agent.step(extraction_query)
            extracted_context = response.msg.content
            try:
                if session_id:
                    from ..utils.llm_usage import extract_token_usage
                    usage = extract_token_usage(response)
                    from ..handlers.session_handler import session_handler
                    session_handler.record_llm_usage(
                        session_id,
                        category="context_extraction",
                        model=str(model_name),
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                    )
            except Exception as usage_exc:
                logger.warning(f"Failed to record context extraction token usage: {usage_exc}")

            if extracted_context.startswith("```"):
                lines = extracted_context.split('\n')
                extracted_context = '\n'.join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            return extracted_context

        except Exception as e:
            logger.exception(f"LLM context extraction failed: {e}")
            raise

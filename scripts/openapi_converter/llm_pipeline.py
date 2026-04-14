#!/usr/bin/env python3
import ast
import json
import logging
import re
import sys
import textwrap
import threading
import time
from itertools import count
from pathlib import Path

# Ensure project root is on sys.path for utils imports
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json_repair
from dotenv import load_dotenv
from camel.agents import ChatAgent
from utils.model_utils import create_model
from .openapi_utils import (
    ensure_descriptions_inplace,
    BatchEndpointResult,
    ClassInfo,
    EnhancedPythonParser,
    ExtractedData,
    FunctionInfo,
    build_data_models_info,
    clean_json_response,
    extract_rule_based_state_data,
    unwrap_http_method,
    fix_endpoint_structure,
    fix_responses_generic,
    remove_examples_inplace,
    remove_refs_generic,
    sanitize_openapi_schema,
    validate_and_fix_endpoint_structure,
    validate_and_fix_paths,
    validate_spec_with_camel,
)

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Suppress very verbose third-party logs in normal runs.
for noisy_logger in (
    "camel.base_model",
    "camel.camel.agents.chat_agent",
    "camel.camel.utils.token_counting",
):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def _reorder_root_fields(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy with root-level `info`/`servers` moved to the tail."""
    if not isinstance(spec, dict):
        return spec

    ordered: Dict[str, Any] = {}
    for key, value in spec.items():
        if key in {"info", "servers"}:
            continue
        ordered[key] = value

    if "info" in spec:
        ordered["info"] = spec["info"]
    if "servers" in spec:
        ordered["servers"] = spec["servers"]
    return ordered


# ---------------------------------------------------------------------------
# RuleBasedDataExtractor
# ---------------------------------------------------------------------------

class RuleBasedDataExtractor:
    """Extract state and static data via deterministic rule-based analysis."""

    def extract_state_and_data(self, extracted_data: ExtractedData,
                               class_info: Optional[ClassInfo]) -> Dict[str, Any]:
        return extract_rule_based_state_data(extracted_data, class_info)


# ---------------------------------------------------------------------------
# EnhancedOpenAPIGenerator (core refactored class)
# ---------------------------------------------------------------------------

class EnhancedOpenAPIGenerator:
    """Enhanced OpenAPI generator with per-method concurrency."""

    def __init__(
        self,
        model_name: str = 'gpt-4.1-mini',
        workers: int = 10,
        llm_timeout: float = 120.0,
    ):
        self.model = create_model(
            model_name,
            max_tokens=16384,
            temperature=0.001,
            timeout=llm_timeout,
            max_retries=0,
        )
        self.workers = workers
        self.llm_timeout = llm_timeout
        self.endpoint_validation_retries = 2
        self.spec = {}
        self.data_extractor = RuleBasedDataExtractor()
        self._components_schemas = {}  # Saved before removal, for $ref resolution
        self._llm_call_counter = count(1)
        self._llm_counter_lock = threading.Lock()

    def _next_llm_call_id(self) -> int:
        with self._llm_counter_lock:
            return next(self._llm_call_counter)

    def _call_llm(self, system_message: str, prompt: str, stage: str, context: str = "") -> str:
        call_id = self._next_llm_call_id()
        context_text = f" context={context}" if context else ""
        logger.info(
            f"[LLM {call_id}] START stage={stage}{context_text} prompt_chars={len(prompt)}"
        )
        started = time.perf_counter()
        try:
            agent = ChatAgent(
                system_message,
                model=self.model,
                step_timeout=self.llm_timeout,
                retry_attempts=0,
                retry_delay=0.0,
            )
            response = agent.step(prompt)
            content = response.msg.content
            elapsed = time.perf_counter() - started
            logger.info(
                f"[LLM {call_id}] END stage={stage} elapsed={elapsed:.2f}s "
                f"response_chars={len(content)}"
            )
            return content
        except Exception:
            elapsed = time.perf_counter() - started
            logger.exception(f"[LLM {call_id}] ERROR stage={stage} elapsed={elapsed:.2f}s")
            raise

    @staticmethod
    def _extract_helper_state_writes(helper: FunctionInfo) -> List[str]:
        """Extract self.* attributes written by a helper method."""
        dedented = textwrap.dedent(helper.source_code or "")
        try:
            tree = ast.parse(dedented)
        except Exception:
            return []

        writes = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        writes.add(target.attr)
            elif isinstance(node, ast.AnnAssign):
                target = node.target
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    writes.add(target.attr)
            elif isinstance(node, ast.AugAssign):
                target = node.target
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    writes.add(target.attr)
        return sorted(writes)

    def generate(
        self,
        class_info: Optional[ClassInfo],
        extracted_data: ExtractedData,
        description_only: bool = False,
    ) -> Dict:
        """Generate OpenAPI spec with enhanced structure"""

        if not class_info:
            logger.error("No class information provided")
            return {}

        pipeline_start = time.perf_counter()
        logger.info(
            f"Pipeline start class={class_info.name} methods={len(class_info.methods)} "
            f"workers={self.workers} llm_timeout={self.llm_timeout}s"
        )

        # Extract structured state data
        state_start = time.perf_counter()
        logger.info("Extracting state and preset data...")
        state_data = self.data_extractor.extract_state_and_data(extracted_data, class_info)
        logger.info(f"State extraction finished in {time.perf_counter() - state_start:.2f}s")

        # Generate API description
        desc_start = time.perf_counter()
        logger.info("Generating API description with full context...")
        api_description = self._generate_api_description_with_helpers(class_info, state_data)
        logger.info(f"API description finished in {time.perf_counter() - desc_start:.2f}s")

        # Initialize spec
        self.spec = {
            "openapi": "3.1.0",
            "info": {
                "title": class_info.name,
                "version": "1.0.0",
                "description": api_description,
                "x-default-state": state_data
            },
            "servers": [{"url": "http://localhost:8000"}],
            "paths": {}
        }

        if description_only:
            compact_state = self._compact_state_data_for_output(state_data)
            self.spec["info"]["x-default-state"] = compact_state
            logger.info("Description-only mode enabled; skipping endpoint generation.")
            logger.info(f"Pipeline finished in {time.perf_counter() - pipeline_start:.2f}s")
            return self.spec

        # Build method index for cross-method context
        method_index = self._build_method_index(class_info.methods)
        method_lookup = {m.name: m for m in class_info.methods}

        # Generate endpoints concurrently (one task per method)
        endpoint_start = time.perf_counter()
        logger.info(
            f"Generating {len(class_info.methods)} endpoints (workers={self.workers})..."
        )
        results = self._generate_endpoints_concurrent(
            class_info.methods, api_description, state_data, method_index, method_lookup
        )
        logger.info(f"Endpoint generation finished in {time.perf_counter() - endpoint_start:.2f}s")
        state_data = self._merge_llm_state_updates(state_data, results)
        self.spec["info"]["x-default-state"] = state_data
        result_by_method = {r.method_name: r for r in results}
        validated_paths: Dict[str, Dict[str, Dict[str, Any]]] = {}
        validate_start = time.perf_counter()
        for method in class_info.methods:
            if method.name not in result_by_method:
                raise RuntimeError(f"Missing generated endpoint for method {method.name}")
            endpoint = result_by_method[method.name].endpoint
            validated_endpoint = self._validate_endpoint_with_retry(
                method=method,
                endpoint=endpoint,
                validated_paths=validated_paths,
                api_description=api_description,
                state_data=state_data,
                method_index=method_index,
            )
            path = f"/{method.name}"
            validated_paths[path] = {"post": validated_endpoint}
        logger.info(
            f"Per-endpoint validation finished in {time.perf_counter() - validate_start:.2f}s"
        )

        self.spec['paths'] = validated_paths

        # Consolidated post-processing
        post_start = time.perf_counter()
        self._post_process_spec()
        compact_state = self._compact_state_data_for_output(state_data)
        self.spec.setdefault("info", {})["x-default-state"] = compact_state
        logger.info(f"Post-processing finished in {time.perf_counter() - post_start:.2f}s")
        logger.info(f"Pipeline finished in {time.perf_counter() - pipeline_start:.2f}s")

        return self.spec

    # ---- API description ----

    def _generate_api_description_with_helpers(self, class_info: ClassInfo,
                                               state_data: Dict[str, Any]) -> str:
        """Generate concise, toolkit-level API description."""
        public_method_details: List[str] = []
        for method in class_info.methods[:20]:
            params = ", ".join(
                f"{p['name']}: {p.get('type_hint', 'Any')}" for p in method.parameters
            )
            ret = f" -> {method.return_type}" if method.return_type else ""
            signature = f"{method.name}({params}){ret}"
            method_doc = (method.docstring or "None").strip() or "None"
            public_method_details.append(
                f"- {signature}\n"
                f"  docstring:\n{textwrap.indent(method_doc, '    ')}"
            )

        context_parts = [
            f"Class: {class_info.name}",
            f"Docstring: {class_info.docstring if class_info.docstring else 'None'}",
            f"Instance variables: {list(class_info.instance_variables.keys())[:10]}",
            "Public methods (up to first 20, with signature + docstring):\n"
            + ("\n".join(public_method_details) if public_method_details else "None"),
        ]

        if 'data_models' in state_data and state_data['data_models']:
            model_names = [
                mn for mn, mi in state_data['data_models'].items()
                if isinstance(mi, dict) and 'fields' in mi
            ]
            if model_names:
                context_parts.append(f"Data Models Available: {model_names[:15]}")

        helper_analysis = []
        if class_info.helper_functions:
            helper_analysis.append("Helper/Private methods (full content):")
            for helper in class_info.helper_functions:
                params = ", ".join(
                    f"{p['name']}: {p.get('type_hint', 'Any')}" for p in helper.parameters
                )
                ret = f" -> {helper.return_type}" if helper.return_type else ""
                signature = f"{helper.name}({params}){ret}"
                helper_doc = (helper.docstring or "None").strip() or "None"
                helper_source = (helper.source_code or "None").strip() or "None"
                helper_analysis.append(
                    f"\n--- helper: {helper.name} ---\n"
                    f"signature: {signature}\n"
                    f"docstring:\n{textwrap.indent(helper_doc, '  ')}\n"
                    f"source_code:\n{textwrap.indent(helper_source, '  ')}"
                )

        full_context = "\n\n".join(
            [part for part in context_parts + (["\n".join(helper_analysis)] if helper_analysis else []) if part]
        )
        global_state = state_data.get("global", {})
        runtime_defaults = global_state.get("runtime_defaults", {})
        tool_states = state_data.get("tools", {})
        try:
            global_state_text = json.dumps(global_state, indent=2, ensure_ascii=False)
        except Exception:
            global_state_text = str(global_state)

        prompt = f"""Write a concise toolkit-level description for this API class.

{full_context}

State Data Summary:
- x-default-state.global (full):
{global_state_text}
- Global runtime defaults keys: {list(runtime_defaults.keys())[:20]}
- Tool-level state extracted for: {list(tool_states.keys())[:15]}

Requirements:
- Focus on general, cross-tool behavior only (shared semantics).
- Include: overall purpose, shared state model, common conventions, common error style.
- Include global initialization rules when loading config/scenario/setup data, if present.
  You MUST use x-default-state.global and helper source code when deriving those rules.
  Example of the kind of rule to capture: how initial runtime context is chosen (e.g., initial cwd).
- Mention operation-specific details ONLY if they are truly global conventions.
- Do NOT provide per-method walkthroughs.
- Do NOT enumerate all methods, constants, or long data tables.
- Do NOT invent response fields or error-schema keys that are not explicitly supported by context.
- If uncertain about a detail, omit it.
- Keep output to one short paragraph, around 80-160 words, plain text.
- No headings, no numbered list, no markdown.

The output will be stored as info.description in OpenAPI and used as shared context."""

        description = self._call_llm(
            system_message=(
                "You are an API documentation expert. Produce concise, general-purpose toolkit "
                "descriptions only. Avoid method-by-method detail."
            ),
            prompt=prompt,
            stage="api_description",
            context=f"class={class_info.name}",
        )

        normalized = " ".join((description or "").split())
        return normalized

    # ---- Method index for cross-method context ----

    def _build_method_index(self, methods: List[FunctionInfo]) -> str:
        """Build compact listing of ALL method signatures for cross-method context."""
        lines = []
        for m in methods:
            params = ", ".join(
                f"{p['name']}: {p.get('type_hint', 'Any')}" for p in m.parameters
            )
            ret = f" -> {m.return_type}" if m.return_type else ""
            doc_line = m.docstring.split('\n')[0] if m.docstring else ""
            lines.append(f"  - {m.name}({params}){ret}  # {doc_line}")
        return "ALL API methods in this class:\n" + "\n".join(lines)

    def _build_state_hint_for_methods(
        self, state_data: Dict[str, Any], method_names: List[str]
    ) -> Dict[str, Any]:
        """Build rich state hint for one or more methods."""
        global_state = state_data.get("global", {})
        runtime_defaults = global_state.get("runtime_defaults") or {}
        tool_states = state_data.get("tools", {})
        tools_hint: Dict[str, Any] = {}
        for method_name in method_names:
            tool_entry = tool_states.get(method_name, {})
            hint_entry: Dict[str, Any] = {
                "static_data": tool_entry.get("static_data", {}),
                "static_data_keys": sorted(list((tool_entry.get("static_data") or {}).keys())),
                "validation_rules": tool_entry.get("validation_rules", []),
            }
            if tool_entry.get("called_method_static_data"):
                hint_entry["called_method_static_data"] = tool_entry["called_method_static_data"]
            if tool_entry.get("method_calls"):
                hint_entry["method_calls"] = tool_entry["method_calls"]
            if tool_entry.get("state_effects"):
                hint_entry["state_effects"] = tool_entry["state_effects"]
            if tool_entry.get("state_effects_on_success"):
                hint_entry["state_effects_on_success"] = tool_entry["state_effects_on_success"]
            if tool_entry.get("state_effects_on_error"):
                hint_entry["state_effects_on_error"] = tool_entry["state_effects_on_error"]
            if tool_entry.get("state_effects_always"):
                hint_entry["state_effects_always"] = tool_entry["state_effects_always"]
            if tool_entry.get("behavior_hints"):
                hint_entry["behavior_hints"] = tool_entry["behavior_hints"]
            if tool_entry.get("success_string_templates"):
                hint_entry["success_string_templates"] = tool_entry["success_string_templates"]
            tools_hint[method_name] = hint_entry
        return {
            "global_runtime_defaults": runtime_defaults,
            "tools": tools_hint,
        }

    @staticmethod
    def _normalize_text_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        result: List[str] = []
        seen = set()
        for item in value:
            if not isinstance(item, str):
                continue
            normalized = " ".join(item.split()).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result

    @staticmethod
    def _normalize_validation_rules(value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        normalized: List[Dict[str, Any]] = []
        seen = set()
        for item in value:
            if not isinstance(item, dict):
                continue
            rule = deepcopy(item)
            marker = json.dumps(rule, ensure_ascii=False, sort_keys=True, default=str)
            if marker in seen:
                continue
            seen.add(marker)
            normalized.append(rule)
        return normalized

    def _collect_called_method_context(
        self,
        method_name: str,
        state_data: Dict[str, Any],
        method_lookup: Dict[str, FunctionInfo],
    ) -> List[Dict[str, Any]]:
        tools_state = state_data.get("tools", {})
        tool_entry = tools_state.get(method_name, {}) if isinstance(tools_state, dict) else {}
        method_calls = tool_entry.get("method_calls", [])
        if not isinstance(method_calls, list):
            return []
        contexts: List[Dict[str, Any]] = []
        for called_name in method_calls:
            if not isinstance(called_name, str):
                continue
            called_method = method_lookup.get(called_name)
            if not called_method:
                continue
            contexts.append(
                {
                    "name": called_method.name,
                    "docstring": called_method.docstring or "",
                    "source_code": called_method.source_code or "",
                    "parameters": called_method.parameters,
                    "return_type": called_method.return_type or "None",
                }
            )
        return contexts

    def _extract_state_updates_from_enrichment(
        self,
        method: FunctionInfo,
        parsed: Dict[str, Any],
    ) -> Dict[str, Any]:
        on_success = self._normalize_text_list(parsed.get("state_effects_on_success"))
        on_error = self._normalize_text_list(parsed.get("state_effects_on_error"))
        on_always = self._normalize_text_list(parsed.get("state_effects_always"))
        validation_rules = self._normalize_validation_rules(parsed.get("validation_rules"))

        updates: Dict[str, Any] = {}
        if on_success:
            updates["state_effects_on_success"] = on_success
            # Backward-compatible field consumed by Gecko today:
            # required effects for successful calls.
            updates["state_effects"] = on_success + [e for e in on_always if e not in on_success]
        elif on_always:
            updates["state_effects"] = list(on_always)
        else:
            updates["state_effects"] = []

        if on_error:
            updates["state_effects_on_error"] = on_error
        if on_always:
            updates["state_effects_always"] = on_always
        if validation_rules:
            updates["validation_rules"] = validation_rules
        return updates

    def _merge_llm_state_updates(
        self,
        state_data: Dict[str, Any],
        results: List[BatchEndpointResult],
    ) -> Dict[str, Any]:
        merged = deepcopy(state_data)
        tools = merged.setdefault("tools", {})
        for result in results:
            if not isinstance(result.state_updates, dict) or not result.state_updates:
                continue
            entry = tools.get(result.method_name)
            if not isinstance(entry, dict):
                entry = {}
                tools[result.method_name] = entry
            for key, value in result.state_updates.items():
                entry[key] = value
        return merged

    def _compact_state_data_for_output(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compact tool-state keys in x-default-state for OpenAPI output."""
        compact = deepcopy(state_data) if isinstance(state_data, dict) else {}
        tools_state = compact.get("tools", {})
        if not isinstance(tools_state, dict):
            compact["tools"] = {}
            return compact

        normalized_tools: Dict[str, Any] = {}
        for method_name, raw_entry in tools_state.items():
            if not isinstance(raw_entry, dict):
                continue

            entry: Dict[str, Any] = {}

            static_data = raw_entry.get("static_data")
            if isinstance(static_data, dict) and static_data:
                entry["static_data"] = static_data

            validation_rules = self._normalize_validation_rules(
                raw_entry.get("validation_rules", [])
            )
            if validation_rules:
                entry["validation_rules"] = validation_rules

            merged_effects: List[str] = []
            for key in (
                "state_effects",
                "state_effects_on_success",
                "state_effects_on_error",
                "state_effects_always",
            ):
                for effect in self._normalize_text_list(raw_entry.get(key, [])):
                    if effect not in merged_effects:
                        merged_effects.append(effect)
            entry["state_effects"] = merged_effects

            behavior_hints = self._normalize_text_list(raw_entry.get("behavior_hints", []))
            templates_map = raw_entry.get("success_string_templates", {})
            if isinstance(templates_map, dict):
                for field_name, templates in templates_map.items():
                    if not isinstance(templates, list):
                        continue
                    cleaned_templates = [
                        t.strip() for t in templates if isinstance(t, str) and t.strip()
                    ]
                    if not cleaned_templates:
                        continue
                    has_existing_template_hint = any(
                        isinstance(h, str)
                        and f"Success field '{field_name}'" in h
                        for h in behavior_hints
                    )
                    if has_existing_template_hint:
                        continue
                    behavior_hints.append(
                        f"Success field '{field_name}' may use template(s): "
                        + " | ".join(cleaned_templates)
                    )
            behavior_hints = self._normalize_text_list(behavior_hints)
            if behavior_hints:
                entry["behavior_hints"] = behavior_hints

            related_method: Dict[str, Any] = {}
            method_calls = raw_entry.get("method_calls", [])
            if isinstance(method_calls, list):
                method_names = [
                    name.strip() for name in method_calls if isinstance(name, str) and name.strip()
                ]
                if method_names:
                    related_method["method_calls"] = method_names
            called_method_static_data = raw_entry.get("called_method_static_data")
            if isinstance(called_method_static_data, dict) and called_method_static_data:
                related_method["static_data"] = called_method_static_data
            if related_method:
                entry["related_method"] = related_method

            normalized_tools[method_name] = entry

        compact["tools"] = normalized_tools
        return compact

    def _review_state_contract_with_llm(
        self,
        method: FunctionInfo,
        parsed_enrichment: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Second-pass LLM review focused on branch-correct state semantics."""
        contract_input = {
            "state_effects_on_success": parsed_enrichment.get("state_effects_on_success", []),
            "state_effects_on_error": parsed_enrichment.get("state_effects_on_error", []),
            "state_effects_always": parsed_enrichment.get("state_effects_always", []),
            "validation_rules": parsed_enrichment.get("validation_rules", []),
        }
        prompt = f"""Review and correct this method state/validation contract.

Method: {method.name}

Method source:
{method.source_code}

Method context payload:
{json.dumps(payload, indent=2, ensure_ascii=False)}

Current proposed contract:
{json.dumps(contract_input, indent=2, ensure_ascii=False)}

Strict definitions:
- state_effects_on_success: mutations that occur ONLY on successful returns.
- state_effects_on_error: mutations that occur ONLY on failure/error paths.
- state_effects_always: mutations that occur regardless of success/failure.
- validation_rules: explicit checks and outcomes from source.

Hard constraints:
- NEVER place error-only mutations in state_effects_on_success.
- If a mutation occurs only when token/auth validation fails, it belongs to state_effects_on_error.
- Keep effects concrete, source-grounded, and concise.
- Do not invent behaviors absent from source.

Return ONLY JSON object:
{{
  "state_effects_on_success": ["..."],
  "state_effects_on_error": ["..."],
  "state_effects_always": ["..."],
  "validation_rules": [{{"condition":"...","result":{{...}}}}]
}}"""

        content = clean_json_response(
            self._call_llm(
                system_message=(
                    "You are a strict code semantics reviewer. "
                    "Correct branch-specific state effects and validation rules. "
                    "Return JSON only."
                ),
                prompt=prompt,
                stage="state_contract_review",
                context=f"method={method.name}",
            )
        )
        reviewed = json_repair.loads(content)
        if not isinstance(reviewed, dict):
            raise TypeError(
                f"State contract review for {method.name} must return object, got "
                f"{type(reviewed).__name__}"
            )
        return reviewed

    # ---- Skeleton-first endpoint generation ----

    @staticmethod
    def _humanize_operation_id(operation_id: str) -> str:
        text = operation_id.replace("_", " ").strip()
        if not text:
            return "Operation"
        return text[0].upper() + text[1:]

    @staticmethod
    def _clean_param_description(text: str) -> str:
        """Remove fabricated examples and implementation details from param descriptions."""
        # Strip "For example: 'ABC123'." patterns and trailing fragments
        cleaned = re.sub(
            r"\s*[Ff]or example[:\s]*['\"][^'\"]*['\"]\.?\s*", " ", text
        )
        # Strip trailing "Use a human-readable name here." type filler
        cleaned = re.sub(
            r"\s*Use a human-readable name here\.?\s*", " ", cleaned
        )
        # Strip "This should be a ..." filler that often follows fabricated examples
        cleaned = re.sub(
            r"\s*This should be a [^.]*\.\s*", " ", cleaned
        )
        return " ".join(cleaned.split()).strip()

    def _normalize_endpoint_description(self, text: str) -> str:
        """Normalize endpoint description text without hard truncation."""
        normalized = " ".join((text or "").split()).strip()
        return normalized

    def _strip_semantic_sections(self, text: str) -> str:
        """Strip auto-generated semantic suffixes (state/validation) from description."""
        normalized = self._normalize_endpoint_description(text)
        lower = normalized.lower()
        candidates = []
        for marker in ("state transitions:", "validation rules:"):
            idx = lower.find(marker)
            if idx >= 0:
                candidates.append(idx)
        if not candidates:
            return normalized
        cut = min(candidates)
        return normalized[:cut].strip()

    def _sanitize_success_mutation_claims(
        self,
        text: str,
        method: FunctionInfo,
        state_data: Dict[str, Any],
    ) -> str:
        """Remove likely-invalid 'success mutates state' claims when success effects are empty."""
        tool_entry = (state_data.get("tools", {}).get(method.name, {}) or {})
        on_success = self._normalize_text_list(tool_entry.get("state_effects_on_success", []))
        legacy_effects = self._normalize_text_list(tool_entry.get("state_effects", []))
        if on_success or legacy_effects:
            return text

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if not sentences:
            return text

        # Generic mutation verbs commonly used in generated descriptions.
        mutation_re = re.compile(
            r"\b(on success|successful(?:ly)?|success)\b.*\b("
            r"set|update|append|remove|delete|increment|decrement|decrease|increase|"
            r"consume|mutat|write|store|record|create"
            r")\b",
            re.IGNORECASE,
        )
        kept = [s for s in sentences if not mutation_re.search(s)]
        cleaned = " ".join(kept).strip()
        return cleaned if cleaned else text

    def _extract_param_descriptions_from_docstring(self, docstring: str) -> Dict[str, str]:
        descriptions: Dict[str, str] = {}
        if not docstring:
            return descriptions

        in_args = False
        current_param = ""
        for raw_line in docstring.splitlines():
            line = raw_line.strip()
            if not in_args:
                if line in {"Args:", "Arguments:"}:
                    in_args = True
                continue
            if line in {"Returns:", "Raises:"}:
                break
            if not line:
                current_param = ""
                continue

            match = re.match(r"^([a-zA-Z_]\w*)\s*(?:\([^)]*\))?\s*:\s*(.+)$", line)
            if match:
                current_param = match.group(1)
                descriptions[current_param] = match.group(2).strip()
                continue
            if current_param and line and not line.endswith(":"):
                descriptions[current_param] = f"{descriptions[current_param]} {line}".strip()

        return descriptions

    def _schema_from_type_hint(self, type_hint: str) -> Dict[str, Any]:
        hint = (type_hint or "Any").strip()
        hint_lower = hint.lower()

        if "optional[" in hint_lower or ("union[" in hint_lower and "none" in hint_lower):
            hint = hint.replace("Optional[", "").replace("optional[", "").rstrip("]")
            hint_lower = hint.lower()

        if "list" in hint_lower or "tuple" in hint_lower:
            items_type = self._extract_inner_type(hint)
            return {"type": "array", "items": {"type": items_type}}
        if "dict" in hint_lower or "mapping" in hint_lower:
            return {"type": "object", "additionalProperties": True}

        inferred = self._infer_json_type(hint)
        if inferred == "array":
            items_type = self._extract_inner_type(hint)
            return {"type": "array", "items": {"type": items_type}}
        if inferred == "object":
            return {"type": "object", "additionalProperties": True}
        schema = {"type": inferred}
        # Preserve float distinction: OpenAPI "number" + format "float"
        if inferred == "number" and hint_lower.startswith("float"):
            schema["format"] = "float"
        return schema

    def _extract_inner_type(self, type_hint: str) -> str:
        """Extract and convert the inner type from a generic like List[float] -> 'number'."""
        import re
        m = re.search(r'\[([^\[\]]+)\]', type_hint)
        if m:
            inner = m.group(1).strip().split(",")[0].strip()  # Take first type for Tuple[T, ...]
            return self._infer_json_type(inner)
        return "string"

    def _schema_from_ast_value(self, node: Optional[ast.AST], depth: int = 0) -> Dict[str, Any]:
        if node is None:
            return {"type": "object"}

        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool):
                return {"type": "boolean"}
            if isinstance(value, int):
                return {"type": "integer"}
            if isinstance(value, float):
                return {"type": "number", "format": "float"}
            if isinstance(value, str):
                return {"type": "string"}
            if value is None:
                return {"type": "null"}
            return {"type": "string"}

        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            item_schema = {"type": "string"}
            elements = getattr(node, "elts", []) or []
            if elements:
                item_schema = self._schema_from_ast_value(elements[0], depth + 1)
            return {"type": "array", "items": item_schema}

        if isinstance(node, ast.Dict):
            properties: Dict[str, Any] = {}
            required: List[str] = []
            for key_node, value_node in zip(node.keys, node.values):
                if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                    key = key_node.value
                    properties[key] = self._schema_from_ast_value(value_node, depth + 1)
                    required.append(key)
            schema: Dict[str, Any] = {"type": "object", "properties": properties}
            if required:
                schema["required"] = sorted(required)
            schema["additionalProperties"] = True
            return schema

        if isinstance(node, ast.Name):
            name = node.id.lower()
            if "list" in name or "items" in name:
                return {"type": "array", "items": {"type": "string"}}
            if "count" in name or "num" in name or name.endswith("_id"):
                return {"type": "integer"}
            if "cost" in name or "price" in name or "amount" in name:
                return {"type": "number", "format": "float"}
            if name.startswith("is_") or name.startswith("has_"):
                return {"type": "boolean"}
            return {"type": "string"}

        if isinstance(node, ast.Call):
            call_name = ""
            if isinstance(node.func, ast.Name):
                call_name = node.func.id.lower()
            elif isinstance(node.func, ast.Attribute):
                call_name = node.func.attr.lower()
            if call_name in {"list", "sorted"}:
                return {"type": "array", "items": {"type": "string"}}
            if call_name in {"dict"}:
                return {"type": "object", "additionalProperties": True}
            if call_name in {"str"}:
                return {"type": "string"}
            if call_name in {"int", "randint"}:
                return {"type": "integer"}
            if call_name in {"float"}:
                return {"type": "number", "format": "float"}
            if call_name in {"bool"}:
                return {"type": "boolean"}
            return {"type": "string"}

        if isinstance(node, ast.UnaryOp):
            return self._schema_from_ast_value(node.operand, depth + 1)

        return {"type": "object"}

    def _infer_success_schema_from_method(self, method: FunctionInfo) -> Dict[str, Any]:
        dedented = textwrap.dedent(method.source_code or "")
        try:
            tree = ast.parse(dedented)
        except Exception:
            return {"type": "object", "description": "Successful response payload"}

        function_node = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                function_node = node
                break
        if function_node is None:
            return {"type": "object", "description": "Successful response payload"}

        raw_schemas: List[Dict[str, Any]] = []
        object_schemas: List[Dict[str, Any]] = []
        key_presence: Dict[str, int] = {}

        for node in ast.walk(function_node):
            if not isinstance(node, ast.Return) or node.value is None:
                continue

            if isinstance(node.value, ast.Dict):
                keys = []
                has_error_key = False
                for key_node in node.value.keys:
                    if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                        keys.append(key_node.value)
                        if key_node.value == "error":
                            has_error_key = True
                if has_error_key:
                    continue

                schema = self._schema_from_ast_value(node.value)
                raw_schemas.append(schema)
                if schema.get("type") == "object" and isinstance(schema.get("properties"), dict):
                    object_schemas.append(schema)
                    for key in schema["properties"]:
                        key_presence[key] = key_presence.get(key, 0) + 1
                continue

            raw_schemas.append(self._schema_from_ast_value(node.value))

        if object_schemas:
            merged_properties: Dict[str, Any] = {}
            for schema in object_schemas:
                for key, prop_schema in schema.get("properties", {}).items():
                    if key not in merged_properties:
                        merged_properties[key] = prop_schema
            required = sorted(
                key for key, count in key_presence.items() if count == len(object_schemas)
            )
            merged_schema: Dict[str, Any] = {
                "type": "object",
                "properties": merged_properties,
                "additionalProperties": True,
            }
            if required:
                merged_schema["required"] = required
            return merged_schema

        if raw_schemas:
            return raw_schemas[0]

        return {"type": "object", "description": "Successful response payload"}

    def _apply_success_schema(self, endpoint: Dict[str, Any], success_schema: Dict[str, Any]) -> None:
        endpoint["responses"] = fix_responses_generic(endpoint.get("responses", {}))
        schema = endpoint["responses"]["200"]["content"]["application/json"].setdefault("schema", {})
        one_of = schema.get("oneOf")
        if not isinstance(one_of, list):
            one_of = []
        while len(one_of) < 2:
            one_of.append({})

        success_branch = deepcopy(success_schema)
        if "description" not in success_branch:
            success_branch["description"] = "Success response"
        one_of[0] = success_branch

        error_branch = one_of[1]
        if not isinstance(error_branch, dict) or "properties" not in error_branch:
            error_branch = {
                "type": "object",
                "properties": {
                    "error": {"type": "string", "description": "Error message"},
                },
                "required": ["error"],
                "description": "Error response",
            }
        one_of[1] = error_branch
        schema["oneOf"] = one_of

    def _build_endpoint_skeleton(self, method: FunctionInfo) -> Dict[str, Any]:
        param_descriptions = self._extract_param_descriptions_from_docstring(method.docstring)

        properties: Dict[str, Any] = {}
        required: List[str] = []
        for param in method.parameters:
            name = param["name"]
            param_schema = self._schema_from_type_hint(param.get("type_hint", "Any"))
            param_schema["description"] = param_descriptions.get(name, f"Parameter {name}")
            if "default" in param:
                param_schema["default"] = param["default"]
            properties[name] = param_schema
            if param.get("required", True):
                required.append(name)

        request_schema: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
            "description": "Request payload",
        }
        if required:
            request_schema["required"] = sorted(required)

        description = (method.docstring or "").strip()
        endpoint: Dict[str, Any] = {
            "operationId": method.name,
            "summary": self._humanize_operation_id(method.name),
            "description": self._normalize_endpoint_description(
                description if description else f"Execute `{method.name}`."
            ),
            "requestBody": {
                "required": bool(method.parameters),
                "content": {"application/json": {"schema": request_schema}},
            },
            "responses": {},
        }
        self._apply_success_schema(endpoint, self._infer_success_schema_from_method(method))
        return endpoint

    def _merge_endpoint_override(
        self,
        method: FunctionInfo,
        base_endpoint: Dict[str, Any],
        override: Dict[str, Any],
    ) -> Dict[str, Any]:
        endpoint = deepcopy(base_endpoint)
        if not isinstance(override, dict):
            raise TypeError(
                f"LLM override for {method.name} must be object, got {type(override).__name__}"
            )

        operation_id = override.get("operationId")
        if isinstance(operation_id, str) and operation_id and operation_id != method.name:
            raise ValueError(
                f"LLM override operationId mismatch for {method.name}: {operation_id}"
            )

        summary = override.get("summary")
        if isinstance(summary, str) and summary.strip():
            endpoint["summary"] = summary.strip()

        description = override.get("description")
        if isinstance(description, str) and description.strip():
            endpoint["description"] = self._normalize_endpoint_description(description.strip())

        parameter_descriptions = override.get("param_descriptions")
        if parameter_descriptions is not None and not isinstance(parameter_descriptions, dict):
            raise TypeError(
                f"param_descriptions for {method.name} must be object when provided"
            )
        if isinstance(parameter_descriptions, dict):
            schema = (
                endpoint.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema", {})
            )
            properties = schema.get("properties", {})
            for key, value in parameter_descriptions.items():
                if key in properties and isinstance(value, str) and value.strip():
                    properties[key]["description"] = self._clean_param_description(value.strip())

        success_schema = override.get("success_schema")
        if success_schema is not None:
            if not isinstance(success_schema, dict):
                raise TypeError(f"success_schema for {method.name} must be an object when provided")
            self._apply_success_schema(endpoint, success_schema)

        endpoint["operationId"] = method.name
        return endpoint

    def _apply_behavior_hints_to_endpoint(
        self,
        endpoint: Dict[str, Any],
        method: FunctionInfo,
        state_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Keep endpoint descriptions concise and free of injected hint details."""
        result = deepcopy(endpoint)
        base_desc = self._normalize_endpoint_description(result.get("description", ""))
        result["description"] = self._strip_semantic_sections(base_desc)
        return result

    def _apply_state_effects_to_endpoint(
        self,
        endpoint: Dict[str, Any],
        method: FunctionInfo,
        state_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Strip generated transition suffixes from endpoint descriptions."""
        result = deepcopy(endpoint)
        base_desc = self._strip_semantic_sections(result.get("description", ""))
        result["description"] = self._normalize_endpoint_description(
            self._sanitize_success_mutation_claims(base_desc, method, state_data)
        )
        return result

    @staticmethod
    def _summarize_validation_rule(rule: Dict[str, Any]) -> Optional[str]:
        if not isinstance(rule, dict):
            return None
        if "condition" in rule and "result" in rule:
            condition = str(rule.get("condition", "")).strip()
            result = rule.get("result")
            try:
                result_text = json.dumps(result, ensure_ascii=False, sort_keys=True)
            except Exception:
                result_text = str(result)
            if condition and result_text:
                return f"if {condition}, returns {result_text}"
        if "raises" in rule:
            exc_type = str(rule.get("raises", "")).strip()
            message = str(rule.get("message", "")).strip()
            if exc_type and message:
                return f"raises {exc_type}: {message}"
            if exc_type:
                return f"raises {exc_type}"
        return None

    def _apply_validation_rules_to_endpoint(
        self,
        endpoint: Dict[str, Any],
        method: FunctionInfo,
        state_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Attach concise validation constraints derived from state hints."""
        result = deepcopy(endpoint)
        base_desc = self._strip_semantic_sections(result.get("description", ""))
        result["description"] = self._normalize_endpoint_description(base_desc)

        tools_state = state_data.get("tools", {})
        tool_entry = tools_state.get(method.name, {}) if isinstance(tools_state, dict) else {}
        validation_rules = tool_entry.get("validation_rules", [])
        if not isinstance(validation_rules, list) or not validation_rules:
            return result

        schema = (
            result.get("requestBody", {})
            .get("content", {})
            .get("application/json", {})
            .get("schema", {})
        )
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        if not isinstance(properties, dict):
            properties = {}

        condition_texts: List[str] = []
        for rule in validation_rules:
            if not isinstance(rule, dict):
                continue
            cond = str(rule.get("condition", "")).strip()
            if cond:
                condition_texts.append(cond)

        if not condition_texts:
            return result

        # 1) Add concise endpoint-level validation summary.
        preview = "; ".join(condition_texts[:2])
        if len(condition_texts) > 2:
            preview += "; ..."
        validation_summary = f"Validation constraints include: {preview}."
        if "Validation constraints include:" not in result["description"] and validation_summary not in result["description"]:
            result["description"] = self._normalize_endpoint_description(
                f"{result['description']} {validation_summary}".strip()
            )

        # 2) Add parameter-level disambiguation for numeric constraints when possible.
        def _append_once(text: str, suffix: str) -> str:
            return text if suffix in text else f"{text} {suffix}".strip()

        for param_name, param_schema in properties.items():
            if not isinstance(param_schema, dict):
                continue
            if param_schema.get("type") not in {"number", "integer", "string"}:
                continue

            related = [c for c in condition_texts if re.search(rf"\\b{re.escape(param_name)}\\b", c)]
            if not related:
                continue

            # Prefer condition with explicit numeric ceiling/floor (e.g. > 50).
            preferred = None
            for cond in related:
                nums = re.findall(r"[-+]?(?:\\d+\\.\\d+|\\d+)", cond)
                if any(float(n) >= 1.0 for n in nums):
                    preferred = cond
                    break
            if preferred is None:
                preferred = related[0]

            desc = str(param_schema.get("description", "") or "").strip()
            # Keep short but concrete; include one representative rule verbatim.
            desc = _append_once(
                desc,
                f"Validation rule: must not violate `{preferred}`.",
            )
            param_schema["description"] = self._normalize_endpoint_description(desc)

        return result

    @staticmethod
    def _template_to_regex_pattern(template: str) -> str:
        """Convert a template string with {...} placeholders into regex pattern."""
        placeholder_token = "__TPL_PLACEHOLDER__"
        normalized = re.sub(r"\{[^{}]+\}", placeholder_token, template)
        escaped = re.escape(normalized)
        escaped = escaped.replace(re.escape(placeholder_token), r".+")
        return f"^{escaped}$"

    def _apply_success_template_patterns(
        self,
        endpoint: Dict[str, Any],
        method: FunctionInfo,
        state_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply regex constraints to success string fields based on source templates."""
        tools_state = state_data.get("tools", {})
        tool_entry = tools_state.get(method.name, {}) if isinstance(tools_state, dict) else {}
        templates_map = tool_entry.get("success_string_templates", {})
        if not isinstance(templates_map, dict) or not templates_map:
            return endpoint

        result = deepcopy(endpoint)
        success_schema = (
            result.get("responses", {})
            .get("200", {})
            .get("content", {})
            .get("application/json", {})
            .get("schema", {})
            .get("oneOf", [{}])[0]
        )
        if not isinstance(success_schema, dict):
            return result
        properties = success_schema.get("properties", {})
        if not isinstance(properties, dict):
            return result

        for field_name, templates in templates_map.items():
            if field_name not in properties:
                continue
            prop_schema = properties.get(field_name)
            if not isinstance(prop_schema, dict):
                continue
            if prop_schema.get("type") != "string":
                continue
            if not isinstance(templates, list):
                continue

            normalized_templates = [t.strip() for t in templates if isinstance(t, str) and t.strip()]
            if not normalized_templates:
                continue

            patterns = [self._template_to_regex_pattern(t) for t in normalized_templates]
            combined_pattern = patterns[0] if len(patterns) == 1 else f"^(?:{'|'.join(p[1:-1] for p in patterns)})$"
            prop_schema["pattern"] = combined_pattern

            template_note = " or ".join(normalized_templates[:3])
            current_desc = prop_schema.get("description", "")
            if isinstance(current_desc, str):
                if template_note in current_desc:
                    merged_desc = current_desc
                else:
                    merged_desc = (
                        f"{current_desc} Expected format template: {template_note}"
                        if current_desc
                        else f"Expected format template: {template_note}"
                    )
                prop_schema["description"] = self._normalize_endpoint_description(merged_desc)

        return result

    def _build_llm_enrichment_payload(
        self,
        method: FunctionInfo,
        state_data: Dict[str, Any],
        base_endpoint: Dict[str, Any],
        method_lookup: Dict[str, FunctionInfo],
    ) -> Dict[str, Any]:
        tools_state = state_data.get("tools", {})
        tool_state = tools_state.get(method.name, {}) if isinstance(tools_state, dict) else {}
        baseline_success_schema = (
            base_endpoint.get("responses", {})
            .get("200", {})
            .get("content", {})
            .get("application/json", {})
            .get("schema", {})
            .get("oneOf", [{}])[0]
        )
        called_method_context = self._collect_called_method_context(
            method_name=method.name,
            state_data=state_data,
            method_lookup=method_lookup,
        )
        return {
            "operationId": method.name,
            "parameters": method.parameters,
            "return_type": method.return_type or "None",
            "docstring": (method.docstring or ""),
            "source_excerpt": (method.source_code or ""),
            "state_hint": {
                "static_data": tool_state.get("static_data", {}),
                "static_data_keys": sorted(list((tool_state.get("static_data") or {}).keys())),
                "called_method_static_data": tool_state.get("called_method_static_data", {}),
                "method_calls": tool_state.get("method_calls", []),
                "validation_rules": tool_state.get("validation_rules", []),
                "state_effects": tool_state.get("state_effects", []),
                "state_effects_on_success": tool_state.get("state_effects_on_success", []),
                "state_effects_on_error": tool_state.get("state_effects_on_error", []),
                "state_effects_always": tool_state.get("state_effects_always", []),
                "behavior_hints": tool_state.get("behavior_hints", []),
                "success_string_templates": tool_state.get("success_string_templates", {}),
            },
            "called_method_context": called_method_context,
            "baseline_success_schema": baseline_success_schema,
        }

    def _generate_single_endpoint_single_call(
        self,
        method: FunctionInfo,
        api_description: str,
        state_data: Dict[str, Any],
        method_index: str,
        method_lookup: Dict[str, FunctionInfo],
    ) -> BatchEndpointResult:
        """Build deterministic skeleton and let LLM enrich semantic fields."""
        data_models_info = build_data_models_info(state_data)
        state_hint = self._build_state_hint_for_methods(state_data, [method.name])
        base_endpoint = self._build_endpoint_skeleton(method)
        payload = self._build_llm_enrichment_payload(
            method, state_data, base_endpoint, method_lookup
        )

        prompt = f"""Generate endpoint enrichment data for ONE method.

You are NOT generating a full endpoint. The endpoint skeleton is deterministic and already built.
You MUST return ONE JSON object with this schema:
{{
  "operationId": "<method name>",
  "summary": "<short summary>",
  "description": "<behavior and constraints>",
  "param_descriptions": {{"param": "description"}},
  "success_schema": {{ ... OpenAPI JSON schema for SUCCESS payload only, no error branch ... }},
  "state_effects_on_success": ["..."],
  "state_effects_on_error": ["..."],
  "state_effects_always": ["..."],
  "validation_rules": [{{"condition": "...", "result": {{...}}}}]
}}

Rules:
- operationId must exactly match the method name.
- NEVER use $ref.
- success_schema must describe only the successful payload (no 'error' field as primary shape).
- If a method might return null/None on success, do NOT model success as null. Instead, provide a structured success object with explicit, informative fields (for example status/result/message and key outputs). Failures should be explicit error objects.
- For counting operations (like wc), explicitly specify how counts are calculated (e.g., lines by len(content.splitlines()), words by len(content.split()), characters by len(content)) in behavior_hints.
- If a method does not mutate state (e.g., read-only operations like cat, wc, ls, pwd), `state_effects_on_success`, `state_effects_on_error`, and `state_effects_always` MUST be empty lists `[]`. Do NOT write "No state mutation occurs".
- param_descriptions keys must come from that method's parameters.
- param_descriptions must only describe what the parameter represents and its expected format/unit. Do NOT include fabricated examples (e.g., "For example: 'ABC123'"), internal validation logic, implementation details, or constraints not visible in the method's docstring or signature.
- Keep description concise and usage-focused (usually 1-2 sentences).
- Description should explain what the tool does and the practical call scope/inputs.
- Do NOT include explicit mutation statements, branch-level validation details, or template details in description.
- Put detailed constraints into behavior_hints/validation_rules/state_effects* fields instead of description text.
- Non-negotiable behavior hints in state_hint.behavior_hints MUST be preserved in behavior_hints output.
- Non-negotiable success transitions in state_hint.state_effects MUST be preserved in state_effects* output.
- CRITICAL branch separation:
  - Put mutations that happen only on successful returns into state_effects_on_success.
  - Put mutations that happen only on error/failure paths into state_effects_on_error.
  - Put mutations that happen regardless of outcome into state_effects_always.
  - NEVER place failure-only mutations into state_effects_on_success.
- Include key validation rules that determine success/failure; do not omit explicit checks visible in source.
- If this method calls other methods (state_hint.tools[method].method_calls and called_method_context), inherit their required constraints/data dependencies (for example route tables, enum/airport validation, cached lookup semantics) when they affect this method.
- Avoid long step-by-step walkthroughs and avoid repeating obvious type/schema details.
- Do not invent constraints or fields that are absent from source/called method context.

API Overview:
{api_description}

{method_index}

Available State Data:
{json.dumps(state_hint, indent=2)}{data_models_info}

Method payload:
{json.dumps(payload, indent=2, ensure_ascii=False)}

Return ONLY the JSON object."""

        content = clean_json_response(
            self._call_llm(
                system_message=(
                    "You enrich deterministic OpenAPI endpoints with semantic details. "
                    "Return only JSON."
                ),
                prompt=prompt,
                stage="endpoint_enrichment_single",
                context=f"method={method.name}",
            )
        )
        try:
            parsed = json_repair.loads(content)
        except Exception as exc:
            raise ValueError(f"Endpoint enrichment JSON parsing failed for {method.name}: {exc}") from exc

        if isinstance(parsed, list):
            if len(parsed) != 1 or not isinstance(parsed[0], dict):
                raise TypeError(
                    f"Endpoint enrichment for {method.name} must be object or single-item object array"
                )
            parsed = parsed[0]
        if not isinstance(parsed, dict):
            raise TypeError(
                f"Endpoint enrichment for {method.name} must return object, got {type(parsed).__name__}"
            )

        try:
            reviewed_contract = self._review_state_contract_with_llm(method, parsed, payload)
            parsed.update(reviewed_contract)
        except Exception as exc:
            logger.warning(
                f"State contract review failed for {method.name}; "
                f"falling back to single-pass enrichment: {exc}"
            )

        state_updates = self._extract_state_updates_from_enrichment(method, parsed)
        endpoint = self._merge_endpoint_override(method, base_endpoint, parsed)
        endpoint = self._apply_behavior_hints_to_endpoint(endpoint, method, state_data)
        endpoint = self._apply_state_effects_to_endpoint(endpoint, method, state_data)
        endpoint = self._apply_validation_rules_to_endpoint(endpoint, method, state_data)
        endpoint = self._apply_success_template_patterns(endpoint, method, state_data)
        endpoint = unwrap_http_method(endpoint)
        endpoint["operationId"] = method.name
        fixed = self._fix_endpoint_structure(endpoint)
        fixed = unwrap_http_method(fixed)
        return BatchEndpointResult(
            method.name,
            fixed,
            "skeleton_plus_llm",
            state_updates=state_updates,
        )

    # ---- Concurrent endpoint processing ----

    def _generate_endpoints_concurrent(self, methods: List[FunctionInfo],
                                       api_description: str,
                                       state_data: Dict[str, Any],
                                       method_index: str,
                                       method_lookup: Dict[str, FunctionInfo]) -> List[BatchEndpointResult]:
        """Process endpoints concurrently using ThreadPoolExecutor."""
        all_results: List[BatchEndpointResult] = []

        def process_method(method_idx: int, method: FunctionInfo) -> BatchEndpointResult:
            logger.info(
                f"  Method {method_idx + 1}/{len(methods)} START name={method.name}"
            )
            started = time.perf_counter()
            method_result = self._generate_single_endpoint_single_call(
                method, api_description, state_data, method_index, method_lookup
            )
            logger.info(
                f"  Method {method_idx + 1}/{len(methods)} END name={method.name} "
                f"elapsed={time.perf_counter() - started:.2f}s"
            )
            return method_result

        if self.workers <= 1 or len(methods) <= 1:
            # Sequential processing
            for idx, method in enumerate(methods):
                all_results.append(process_method(idx, method))
        else:
            # Concurrent processing
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {
                    executor.submit(process_method, idx, method): idx
                    for idx, method in enumerate(methods)
                }
                method_results_map: Dict[int, BatchEndpointResult] = {}
                for future in as_completed(futures):
                    idx = futures[future]
                    method_results_map[idx] = future.result()
                # Collect in order
                for idx in range(len(methods)):
                    all_results.append(method_results_map[idx])

        # Verify parameters against AST
        method_map = {m.name: m for m in methods}
        for result in all_results:
            if result.method_name in method_map:
                self._verify_parameters(result.endpoint, method_map[result.method_name])

        return all_results

    def _normalize_endpoint_for_validation(self, endpoint: Dict, operation_id: str) -> Dict:
        """Normalize an endpoint before per-endpoint CAMEL validation."""
        normalized = deepcopy(endpoint)
        normalized = unwrap_http_method(normalized)
        normalized['operationId'] = operation_id
        normalized = self._fix_endpoint_structure(normalized)
        normalized = unwrap_http_method(normalized)
        self._remove_refs(normalized)
        if 'responses' in normalized:
            normalized['responses'] = self._fix_responses(normalized['responses'])
        self._remove_examples(normalized)
        self._ensure_descriptions(normalized)
        self._validate_and_fix_endpoint_structure(normalized)
        return normalized

    def _build_partial_spec_for_validation(
        self,
        validated_paths: Dict[str, Dict[str, Dict[str, Any]]],
        path: str,
        endpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a partial spec containing validated paths plus one candidate path."""
        spec = {
            "openapi": "3.1.0",
            "info": deepcopy(self.spec.get("info", {})),
            "servers": deepcopy(self.spec.get("servers", [{"url": "http://localhost:8000"}])),
            "paths": deepcopy(validated_paths),
        }
        spec["paths"][path] = {"post": endpoint}
        return spec

    def _validate_endpoint_with_retry(
        self,
        method: FunctionInfo,
        endpoint: Dict[str, Any],
        validated_paths: Dict[str, Dict[str, Dict[str, Any]]],
        api_description: str,
        state_data: Dict[str, Any],
        method_index: str,
    ) -> Dict[str, Any]:
        """Validate one endpoint with CAMEL using partial spec; retry regeneration on failure."""
        path = f"/{method.name}"
        candidate = endpoint
        last_error = ""

        for attempt in range(self.endpoint_validation_retries + 1):
            candidate = self._apply_behavior_hints_to_endpoint(candidate, method, state_data)
            candidate = self._apply_state_effects_to_endpoint(candidate, method, state_data)
            candidate = self._apply_validation_rules_to_endpoint(candidate, method, state_data)
            candidate = self._apply_success_template_patterns(candidate, method, state_data)
            self._verify_parameters(candidate, method)
            normalized = self._normalize_endpoint_for_validation(candidate, method.name)
            partial_spec = self._build_partial_spec_for_validation(validated_paths, path, normalized)
            ok, error = self._validate_spec_with_camel(partial_spec)
            if ok:
                if attempt > 0:
                    logger.info(f"Endpoint {method.name} passed CAMEL validation after retry {attempt}")
                else:
                    logger.info(f"Endpoint {method.name} passed CAMEL validation")
                return normalized

            last_error = error or "Unknown CAMEL validation error"
            logger.warning(
                f"Endpoint {method.name} failed CAMEL validation (attempt {attempt + 1}/"
                f"{self.endpoint_validation_retries + 1}): {last_error}"
            )
            if attempt >= self.endpoint_validation_retries:
                break
            candidate = self._regenerate_single_endpoint_with_feedback(
                method=method,
                api_description=api_description,
                state_data=state_data,
                method_index=method_index,
                current_endpoint=normalized,
                validation_error=last_error,
            )

        raise RuntimeError(f"Endpoint {method.name} failed CAMEL validation: {last_error}")

    def _regenerate_single_endpoint_with_feedback(
        self,
        method: FunctionInfo,
        api_description: str,
        state_data: Dict[str, Any],
        method_index: str,
        current_endpoint: Dict[str, Any],
        validation_error: str,
    ) -> Dict[str, Any]:
        """Regenerate a single endpoint enrichment with explicit validation error feedback."""
        data_models_info = build_data_models_info(state_data)
        state_hint = self._build_state_hint_for_methods(state_data, [method.name])
        trimmed_error = validation_error
        base_endpoint = self._build_endpoint_skeleton(method)
        payload = self._build_llm_enrichment_payload(method, state_data, base_endpoint)

        prompt = f"""Regenerate enrichment fields for ONE endpoint.

Previous validation error (must fix):
{trimmed_error}

API Overview:
{api_description}

{method_index}

Available State Data:
{json.dumps(state_hint, indent=2)}{data_models_info}

Current endpoint (failed validation):
{json.dumps(current_endpoint, indent=2, ensure_ascii=False)}

Method payload:
{json.dumps(payload, indent=2, ensure_ascii=False)}

CRITICAL REQUIREMENTS:
- Return ONE JSON object only (not array) with fields:
  operationId, summary, description, param_descriptions, success_schema
- operationId must be exactly {method.name}
- NEVER use $ref
- success_schema must describe SUCCESS payload only
- Keep description concise and usage-focused (usually 1-2 sentences)
- Do not include detailed branch-level state/validation text in description.
- Put detailed constraints into behavior_hints/validation_rules/state_effects* fields.
- Avoid long step-by-step walkthroughs and avoid repeating obvious type/schema details

Return ONLY the enrichment object JSON."""

        content = clean_json_response(
            self._call_llm(
                system_message="You fix endpoint enrichment fields only. Return only valid JSON object.",
                prompt=prompt,
                stage="endpoint_enrichment_retry",
                context=f"method={method.name}",
            )
        )
        try:
            parsed = json_repair.loads(content)
        except Exception as exc:
            raise ValueError(f"Single endpoint regeneration JSON parsing failed: {exc}") from exc
        if not isinstance(parsed, dict):
            raise TypeError(
                f"Single endpoint regeneration must return object, got {type(parsed).__name__}"
            )
        return self._merge_endpoint_override(method, base_endpoint, parsed)

    # ---- Parameter verification ----

    def _verify_parameters(self, endpoint: Dict, method: FunctionInfo) -> None:
        """Compare LLM-generated parameters against AST-extracted parameters."""
        ast_params = {p['name']: p for p in method.parameters}
        ast_param_names = set(ast_params.keys())

        # Extract parameter names from endpoint requestBody
        rb = endpoint.get('requestBody', {})
        content = rb.get('content', {}).get('application/json', {})
        schema = content.get('schema', {})
        properties = schema.get('properties', {})
        llm_param_names = set(properties.keys())
        required_list = schema.get('required', [])

        corrections = []

        # Add missing params
        for name in ast_param_names - llm_param_names:
            param = ast_params[name]
            properties[name] = {
                "type": self._infer_json_type(param.get('type_hint', 'Any')),
                "description": f"Parameter {name}"
            }
            if param.get('required', True) and name not in required_list:
                required_list.append(name)
            corrections.append(f"added missing param '{name}'")

        # Remove hallucinated params
        for name in llm_param_names - ast_param_names:
            del properties[name]
            if name in required_list:
                required_list.remove(name)
            corrections.append(f"removed hallucinated param '{name}'")

        # Fix required flags
        for name in ast_param_names & llm_param_names:
            param = ast_params[name]
            should_be_required = param.get('required', True)
            is_required = name in required_list
            if should_be_required and not is_required:
                required_list.append(name)
                corrections.append(f"marked '{name}' as required")
            elif not should_be_required and is_required:
                required_list.remove(name)
                corrections.append(f"marked '{name}' as optional")

        if corrections:
            logger.info(f"    Parameter verification [{method.name}]: {', '.join(corrections)}")

        # Write back
        schema['properties'] = properties
        if required_list:
            schema['required'] = required_list
        elif 'required' in schema:
            del schema['required']

    # ---- Rule-based normalization wrappers ----

    def _fix_endpoint_structure(self, endpoint: Dict) -> Dict:
        return fix_endpoint_structure(endpoint)

    def _remove_refs(self, obj: Any) -> None:
        remove_refs_generic(obj, self._components_schemas)

    def _remove_examples(self, obj: Any) -> None:
        remove_examples_inplace(obj)

    def _ensure_descriptions(self, obj: Any) -> None:
        ensure_descriptions_inplace(obj)

    @staticmethod
    def _summarize_success_schema_fields(endpoint: Dict[str, Any]) -> Optional[str]:
        """Build a short success-response field summary from 200->oneOf[0] schema."""
        try:
            success_schema = (
                endpoint.get("responses", {})
                .get("200", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema", {})
                .get("oneOf", [{}])[0]
            )
            if not isinstance(success_schema, dict):
                return None
            props = success_schema.get("properties", {})
            if not isinstance(props, dict) or not props:
                return None

            fields: List[str] = []
            for key, prop in props.items():
                if not isinstance(prop, dict):
                    fields.append(str(key))
                    continue
                ptype = prop.get("type")
                extras: List[str] = []
                if isinstance(prop.get("enum"), list) and prop["enum"]:
                    extras.append("enum")
                if "minimum" in prop:
                    extras.append(f"min={prop['minimum']}")
                if "maximum" in prop:
                    extras.append(f"max={prop['maximum']}")
                if ptype:
                    details = ",".join(extras)
                    fields.append(f"{key} ({ptype}{', ' + details if details else ''})")
                else:
                    fields.append(str(key))

            if not fields:
                return None
            preview = ", ".join(fields[:4])
            if len(fields) > 4:
                preview += ", ..."
            return f"Success response fields: {preview}."
        except Exception:
            return None

    def _enhance_string_param_descriptions(self, endpoint: Dict[str, Any]) -> None:
        """Make free-form string parameters less ambiguous with examples and contrasts."""
        schema = (
            endpoint.get("requestBody", {})
            .get("content", {})
            .get("application/json", {})
            .get("schema", {})
        )
        if not isinstance(schema, dict):
            return
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return

        prop_names = set(properties.keys())
        id_like_names = {
            n for n in prop_names if n == "id" or n.endswith("_id") or n.endswith("Id")
        }

        def _append_once(text: str, suffix: str) -> str:
            if suffix in text:
                return text
            return f"{text} {suffix}".strip()

        for name, prop in properties.items():
            if not isinstance(prop, dict):
                continue
            if prop.get("type") != "string":
                continue

            desc = str(prop.get("description", "") or "").strip()
            lname = name.lower()

            # Disambiguate name-like fields when id-like sibling params exist.
            is_name_like = (
                ("name" in lname and not lname.endswith("_name_id"))
                or (lname == "user")
                or (lname.endswith("_user"))
                or (lname == "username")
            )
            has_id_sibling = any(
                sname for sname in id_like_names if sname != name and (lname in sname or "user" in lname)
            )
            if is_name_like and has_id_sibling:
                desc = _append_once(
                    desc,
                    "Use a human-readable name here. Do not pass an identifier value from *_id fields (e.g., 'USR001').",
                )
            # Even without explicit sibling fields, user/name inputs are frequently confused with IDs.
            if lname in {"user", "username", "user_name"}:
                desc = _append_once(
                    desc,
                    "This field expects a human-readable username, not a user_id (e.g., use 'John', not 'USR001').",
                )

            # Disambiguate id-like fields from names.
            if lname.endswith("_id") or lname == "id":
                desc = _append_once(
                    desc,
                    "Use the identifier value, not a human-readable name.",
                )

            prop["description"] = self._normalize_endpoint_description(desc)

    def _fix_responses(self, responses: Dict) -> Dict:
        return fix_responses_generic(responses)

    @staticmethod
    def _is_error_schema_branch(schema_obj: Any) -> bool:
        """Heuristically detect an error-branch schema in oneOf."""
        if not isinstance(schema_obj, dict):
            return False
        props = schema_obj.get("properties")
        if isinstance(props, dict) and "error" in props:
            return True
        required = schema_obj.get("required")
        if isinstance(required, list) and "error" in required:
            return True
        desc = str(schema_obj.get("description", "") or "").lower()
        return "error response" in desc or desc == "error"

    def _prune_error_branch_when_no_failure_path(
        self,
        endpoint: Dict[str, Any],
        tool_entry: Dict[str, Any],
    ) -> None:
        """
        Remove synthetic error branch for tools without explicit failure paths.

        Rule:
        - Keep error branch when there are explicit validation rules OR error-only state effects.
        - Otherwise keep success-only schema to align with real tool behavior.
        """
        if not isinstance(endpoint, dict):
            return
        if not isinstance(tool_entry, dict):
            tool_entry = {}

        validation_rules = tool_entry.get("validation_rules", [])
        state_effects_on_error = tool_entry.get("state_effects_on_error", [])
        has_fail_path = bool(validation_rules) or bool(state_effects_on_error)
        if has_fail_path:
            return

        schema = (
            endpoint.get("responses", {})
            .get("200", {})
            .get("content", {})
            .get("application/json", {})
            .get("schema", {})
        )
        if not isinstance(schema, dict):
            return
        one_of = schema.get("oneOf")
        if not isinstance(one_of, list) or len(one_of) <= 1:
            return

        filtered = [branch for branch in one_of if not self._is_error_schema_branch(branch)]
        if not filtered:
            return
        schema["oneOf"] = filtered

    @staticmethod
    def _ensure_float_format(obj: Any) -> None:
        """Add format:'float' to every type:'number' schema that lacks it."""
        if isinstance(obj, dict):
            if obj.get("type") == "number" and "format" not in obj:
                obj["format"] = "float"
            for v in obj.values():
                EnhancedOpenAPIGenerator._ensure_float_format(v)
        elif isinstance(obj, list):
            for item in obj:
                EnhancedOpenAPIGenerator._ensure_float_format(item)

    # ---- Consolidated post-processing ----

    def _post_process_spec(self) -> None:
        """Consolidated post-processing pipeline:
        1. Save components/schemas for $ref resolution
        2. Generic $ref removal
        3. Fix responses (ensure 200 + oneOf)
        4. Remove examples
        5. Ensure descriptions
        6. Validate and fix structure
        7. Remove components section
        """
        # Step 1: Save components/schemas before removal (for $ref resolution)
        self._components_schemas = deepcopy(
            self.spec.get('components', {}).get('schemas', {})
        )

        # Steps 2-5 on each endpoint
        tools_state = (
            self.spec.get("info", {})
            .get("x-default-state", {})
            .get("tools", {})
        )
        for path, methods in self.spec.get('paths', {}).items():
            for method_key, endpoint in methods.items():
                if not isinstance(endpoint, dict):
                    continue
                # Generic $ref removal
                self._remove_refs(endpoint)
                # Fix responses
                if 'responses' in endpoint:
                    endpoint['responses'] = self._fix_responses(endpoint['responses'])
                operation_id = endpoint.get("operationId")
                tool_entry = (
                    tools_state.get(operation_id, {})
                    if isinstance(tools_state, dict) and isinstance(operation_id, str)
                    else {}
                )
                self._prune_error_branch_when_no_failure_path(endpoint, tool_entry)
                # Remove examples
                self._remove_examples(endpoint)
                # Ensure descriptions
                self._ensure_descriptions(endpoint)
                if isinstance(endpoint.get("description"), str):
                    endpoint["description"] = self._normalize_endpoint_description(endpoint["description"])
                # Make free-form string parameters less ambiguous.
                self._enhance_string_param_descriptions(endpoint)
                # Expose concise success response schema details in endpoint description
                # so FC models can leverage response semantics.
                summary = self._summarize_success_schema_fields(endpoint)
                if summary:
                    base_desc = endpoint.get("description", "")
                    if isinstance(base_desc, str):
                        if summary not in base_desc:
                            endpoint["description"] = self._normalize_endpoint_description(
                                f"{base_desc} {summary}".strip()
                            )
                    else:
                        endpoint["description"] = summary

        # Ensure all number properties have format: "float"
        self._ensure_float_format(self.spec.get("paths", {}))

        # Step 6: Validate and fix structure
        self._validate_and_fix_spec()

        # Step 7: Remove components section
        if 'components' in self.spec:
            del self.spec['components']

    def _validate_and_fix_endpoint_structure(self, endpoint: Dict[str, Any]) -> int:
        return validate_and_fix_endpoint_structure(endpoint)

    def _validate_and_fix_spec(self) -> None:
        """Validate and fix the complete spec."""
        issues_fixed = validate_and_fix_paths(self.spec.get('paths', {}))
        if issues_fixed > 0:
            logger.info(f"Fixed {issues_fixed} structural issues")

    # ---- Type inference ----

    def _infer_json_type(self, type_hint: str) -> str:
        """Infer JSON type from Python type hint"""
        type_mapping = {
            'str': 'string',
            'int': 'integer',
            'float': 'number',
            'bool': 'boolean',
            'list': 'array',
            'dict': 'object',
            'List': 'array',
            'Dict': 'object',
            'Optional': 'string',
            'Any': 'object'
        }

        for py_type, json_type in type_mapping.items():
            if py_type.lower() in type_hint.lower():
                return json_type

        return 'string'

    # ---- CAMEL validation ----

    def _validate_spec_with_camel(self, spec: Dict[str, Any]) -> Tuple[bool, str]:
        paths_count = len(spec.get("paths", {}))
        started = time.perf_counter()
        logger.info(f"[CAMEL] START validate paths={paths_count}")
        ok, err = validate_spec_with_camel(spec)
        elapsed = time.perf_counter() - started
        if ok:
            logger.info(f"[CAMEL] END validate success elapsed={elapsed:.2f}s")
        else:
            logger.warning(
                f"[CAMEL] END validate failure elapsed={elapsed:.2f}s error={err}"
            )
        return ok, err

    def validate_with_camel(self) -> bool:
        """Validate with CAMEL OpenAPIToolkit"""
        ok, err = self._validate_spec_with_camel(self.spec)
        if ok:
            logger.info("CAMEL validation: success")
            return True
        logger.warning(f"CAMEL validation error: {err}")
        return False


def convert_python_to_openapi(
    input_file: str,
    output: str = "openapi_spec.json",
    model: str = "gpt-4.1-mini",
    additional_files: Optional[List[str]] = None,
    workers: int = 10,
    llm_timeout: float = 120.0,
    class_name: Optional[str] = None,
    description_only: bool = False,
    state_output: Optional[str] = None,
) -> int:
    """Run the full conversion flow and write the output spec to disk."""
    total_start = time.perf_counter()
    additional_files = additional_files or []
    logger.info(f"Parsing {input_file}...")
    if additional_files:
        logger.info(f"Also parsing additional files: {additional_files}")

    parse_start = time.perf_counter()
    py_parser = EnhancedPythonParser()
    class_info, extracted_data = py_parser.parse_multiple_files(
        input_file,
        additional_files,
        preferred_class_name=class_name,
    )
    logger.info(f"Parsing finished in {time.perf_counter() - parse_start:.2f}s")
    if not class_info:
        logger.error("No class found in file")
        return 1

    logger.info(f"Found class {class_info.name} with {len(class_info.methods)} methods")
    logger.info(f"Extracted {len(extracted_data.preset_databases)} preset databases")
    logger.info(f"Extracted {len(extracted_data.constants)} constants")
    if extracted_data.data_models_content:
        logger.info(f"Loaded {len(extracted_data.data_models_content)} characters of data model definitions")

    logger.info(
        f"Initializing generator model={model} workers={workers} "
        f"llm_timeout={llm_timeout}s class_name={class_name or 'auto'} "
        f"description_only={description_only}"
    )
    generator = EnhancedOpenAPIGenerator(
        model_name=model,
        workers=workers,
        llm_timeout=llm_timeout,
    )
    generate_start = time.perf_counter()
    spec = generator.generate(
        class_info,
        extracted_data,
        description_only=description_only,
    )
    logger.info(f"Spec generation finished in {time.perf_counter() - generate_start:.2f}s")
    if not spec:
        logger.error("Failed to generate OpenAPI spec")
        return 1

    write_start = time.perf_counter()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_spec = sanitize_openapi_schema(_reorder_root_fields(spec))
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(output_spec, output_file, indent=2, ensure_ascii=False)
    logger.info(f"OpenAPI spec saved to {output_path}")
    logger.info(f"Spec write finished in {time.perf_counter() - write_start:.2f}s")

    if state_output:
        state_path = Path(state_output)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_payload = spec.get("info", {}).get("x-default-state", {})
        with open(state_path, "w", encoding="utf-8") as state_file:
            json.dump(state_payload, state_file, indent=2, ensure_ascii=False)
        logger.info(f"Extracted state data saved to {state_path}")

    final_validate_start = time.perf_counter()
    if generator.validate_with_camel():
        logger.info(
            f"Final CAMEL validation finished in {time.perf_counter() - final_validate_start:.2f}s"
        )
        logger.info("Spec validated successfully (mandatory CAMEL validation)")
        logger.info(f"Total conversion finished in {time.perf_counter() - total_start:.2f}s")
        return 0

    logger.error("Mandatory CAMEL validation failed")
    logger.info(f"Total conversion finished in {time.perf_counter() - total_start:.2f}s")
    return 1

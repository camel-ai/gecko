import copy
import importlib
import inspect
import json
import logging
import re
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from benchmarks.base.test_case import TestCase
from inference.core.sim_solver import SimSolver, TurnOutcome, SimEvent
from inference.utils.log_context import bind_log_context
from utils.reasoning import strip_reasoning
from utils.bfcl_multi_turn_tool_names import (
    canonicalize_bfcl_multi_turn_toolkit,
    normalize_bfcl_multi_turn_function_tool,
    normalize_bfcl_multi_turn_tool_call,
)

from gats.core.task import GATSTask, GATSAttempt, GATSTurn

logger = logging.getLogger(__name__)

REAL_TASK_AGENT_TIMEOUT_SECONDS = 3600

CLASS_MODULE_MAPPING: Dict[str, Dict[str, Any]] = {
    "GorillaFileSystem": {
        "module": "gorilla_file_system",
        "class": "GorillaFileSystem",
    },
    "MathAPI": {
        "module": "math_api",
        "class": "MathAPI",
    },
    "MessageAPI": {
        "module": "message_api",
        "class": "MessageAPI",
    },
    "TwitterAPI": {
        "module": "posting_api",
        "class": "TwitterAPI",
    },
    "PostingAPI": {
        "module": "posting_api",
        "class": "TwitterAPI",
    },
    "TicketAPI": {
        "module": "ticket_api",
        "class": "TicketAPI",
    },
    "TradingBot": {
        "module": "trading_bot",
        "class": "TradingBot",
    },
    "TravelAPI": {
        "module": "travel_booking",
        "class": "TravelAPI",
    },
    "TravelBooking": {
        "module": "travel_booking",
        "class": "TravelAPI",
    },
    "VehicleControlAPI": {
        "module": "vehicle_control",
        "class": "VehicleControlAPI",
    },
    "VehicleControl": {
        "module": "vehicle_control",
        "class": "VehicleControlAPI",
    },
}

CLASS_FUNC_DOC_MAPPING: Dict[str, str] = {
    "GorillaFileSystem": "gorilla_file_system",
    "MathAPI": "math_api",
    "MessageAPI": "message_api",
    "TwitterAPI": "posting_api",
    "PostingAPI": "posting_api",
    "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "TravelBooking": "travel_booking",
    "VehicleControlAPI": "vehicle_control",
    "VehicleControl": "vehicle_control",
}

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_FUNC_SOURCE_DIR = os.path.join(
    _PROJECT_ROOT, "benchmarks", "bfcl", "multi_turn", "func_source_code"
)
_FUNC_DOC_DIR = os.path.join(
    _PROJECT_ROOT, "data", "bfcl", "multi_turn_func_doc"
)
_MULTI_TURN_SCHEMA_DIR = os.path.join(
    _PROJECT_ROOT, "data", "bfcl", "openapi", "multi_turn"
)

_PRECONDITION_INDEX: Optional[Dict[str, List[Dict[str, Any]]]] = None
_STATE_ACCESS_INDEX: Optional[Dict[str, str]] = None
_STATE_EFFECTS_INDEX: Optional[Dict[str, List[str]]] = None


def _iter_multi_turn_schema_paths(schema_paths: Optional[List[str]] = None) -> List[str]:
    paths: List[str] = []
    seen = set()
    if schema_paths:
        candidates = schema_paths
    elif os.path.isdir(_MULTI_TURN_SCHEMA_DIR):
        candidates = [
            os.path.join(_MULTI_TURN_SCHEMA_DIR, fname)
            for fname in os.listdir(_MULTI_TURN_SCHEMA_DIR)
            if fname.endswith(".json")
        ]
    else:
        candidates = []

    for path in candidates:
        if not isinstance(path, str) or not path.endswith(".json"):
            continue
        if not os.path.exists(path) or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def _load_precondition_index(
    schema_paths: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Scan every multi-turn OpenAPI schema and index tool preconditions.

    Returns a mapping ``tool_name -> list of entries``, where each entry is
    ``{"toolkit": <Toolkit>, "state_path": str, "required_value": Any,
        "remedy": {"name": str, "arguments": dict}}``.
    """
    global _PRECONDITION_INDEX
    if schema_paths is None and _PRECONDITION_INDEX is not None:
        return _PRECONDITION_INDEX
    index: Dict[str, List[Dict[str, Any]]] = {}
    schema_files = _iter_multi_turn_schema_paths(schema_paths)
    if not schema_files:
        if schema_paths is None:
            _PRECONDITION_INDEX = index
        return index
    for path in schema_files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                schema = json.load(fh)
        except Exception:
            continue
        toolkit = os.path.splitext(os.path.basename(path))[0]
        tools = (
            (schema.get("info") or {})
            .get("x-default-state", {})
            .get("tools", {})
        )
        if not isinstance(tools, dict):
            continue
        for tool_name, entry in tools.items():
            if not isinstance(entry, dict):
                continue
            preconds = entry.get("preconditions")
            if not isinstance(preconds, list):
                continue
            for pc in preconds:
                if not isinstance(pc, dict):
                    continue
                remedy = pc.get("remedy")
                state_path = pc.get("state_path")
                if not isinstance(remedy, dict) or not isinstance(state_path, str):
                    continue
                remedy_name = remedy.get("name")
                if not isinstance(remedy_name, str):
                    continue
                index.setdefault(tool_name, []).append({
                    "toolkit": toolkit,
                    "state_path": state_path.strip(),
                    "required_value": pc.get("required_value"),
                    "remedy": {
                        "name": remedy_name,
                        "arguments": remedy.get("arguments") or {},
                    },
                    "rationale": pc.get("rationale", ""),
                })
    if schema_paths is None:
        _PRECONDITION_INDEX = index
    return index


def _load_state_access_index(
    schema_paths: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Return schema-declared state_access keyed by tool name."""
    global _STATE_ACCESS_INDEX
    if schema_paths is None and _STATE_ACCESS_INDEX is not None:
        return _STATE_ACCESS_INDEX
    index: Dict[str, str] = {}
    schema_files = _iter_multi_turn_schema_paths(schema_paths)
    if not schema_files:
        if schema_paths is None:
            _STATE_ACCESS_INDEX = index
        return index
    for path in schema_files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                schema = json.load(fh)
        except Exception:
            continue
        tools = (
            (schema.get("info") or {})
            .get("x-default-state", {})
            .get("tools", {})
        )
        if not isinstance(tools, dict):
            continue
        for tool_name, entry in tools.items():
            if not isinstance(tool_name, str) or not isinstance(entry, dict):
                continue
            state_access = entry.get("state_access")
            if isinstance(state_access, str) and state_access:
                index[tool_name] = state_access.lower()
    if schema_paths is None:
        _STATE_ACCESS_INDEX = index
    return index


def _load_state_effects_index(
    schema_paths: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Return schema-declared state effects keyed by tool name."""
    global _STATE_EFFECTS_INDEX
    if schema_paths is None and _STATE_EFFECTS_INDEX is not None:
        return _STATE_EFFECTS_INDEX
    index: Dict[str, List[str]] = {}
    schema_files = _iter_multi_turn_schema_paths(schema_paths)
    if not schema_files:
        if schema_paths is None:
            _STATE_EFFECTS_INDEX = index
        return index
    for path in schema_files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                schema = json.load(fh)
        except Exception:
            continue
        tools = (
            (schema.get("info") or {})
            .get("x-default-state", {})
            .get("tools", {})
        )
        if not isinstance(tools, dict):
            continue
        for tool_name, entry in tools.items():
            if not isinstance(tool_name, str) or not isinstance(entry, dict):
                continue
            effects = entry.get("state_effects_on_success")
            if not isinstance(effects, list) or not effects:
                effects = entry.get("state_effects")
            if isinstance(effects, list) and any(str(effect).strip() for effect in effects):
                index[tool_name] = [str(effect) for effect in effects if str(effect).strip()]
    if schema_paths is None:
        _STATE_EFFECTS_INDEX = index
    return index


def _fix_schema_types(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Fix BFCL func_doc schema types for OpenAI strict mode."""
    result = copy.deepcopy(schema)
    type_map = {"float": "number", "dict": "object", "tuple": "array"}
    original_type = result.get("type")
    if original_type in type_map:
        result["type"] = type_map[original_type]
        if original_type == "float":
            result["format"] = "float"
    result.pop("default", None)
    if "properties" in result and isinstance(result["properties"], dict):
        result["properties"] = {
            k: _fix_schema_types(v) for k, v in result["properties"].items()
        }
    if "items" in result and isinstance(result["items"], dict):
        result["items"] = _fix_schema_types(result["items"])
    if isinstance(result.get("additionalProperties"), dict):
        result["additionalProperties"] = _fix_schema_types(result["additionalProperties"])
    is_obj = result.get("type") == "object" or "properties" in result
    if is_obj:
        result.setdefault("type", "object")
        props = result.get("properties", {})
        if props:
            result["required"] = list(props.keys())
        result.setdefault("additionalProperties", False)
    return result


def _build_openai_tool_schema(func_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert BFCL func_doc to OpenAI tool schema format."""
    params = _fix_schema_types(copy.deepcopy(func_doc.get("parameters", {})))
    return {
        "type": "function",
        "function": {
            "name": func_doc["name"],
            "description": func_doc.get("description", ""),
            "strict": True,
            "parameters": params,
        },
    }


def _build_test_case(task: GATSTask) -> TestCase:
    """Create a minimal TestCase from a GATSTask for SimSolver compatibility."""
    return TestCase(
        id=task.id,
        metadata={
            "initial_config": deepcopy(task.initial_config) if task.initial_config else {},
            **task.metadata,
        },
    )


def _outcome_to_gats_turn(
    outcome: TurnOutcome,
    question: str,
    real_tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> GATSTurn:
    """Convert SimSolver TurnOutcome + events into a GATSTurn."""
    attempts_by_idx: Dict[int, Dict[str, Any]] = {}
    checklist: List[Dict[str, Any]] = []

    for ev in outcome.events:
        if ev.type == "checklist":
            raw = ev.data.get("checklist", []) or []
            checklist = [
                {"description": it.get("description", "")}
                for it in raw
                if isinstance(it, dict)
            ]
            continue
        if ev.attempt is None:
            continue
        idx = int(ev.attempt)
        attempts_by_idx.setdefault(idx, {
            "index": idx,
            "tool_calls": [],
            "score": 0.0,
            "feedback": {},
            "config_after": {},
            "agent_response": "",
            "execution_time": 0.0,
        })
        rec = attempts_by_idx[idx]
        if ev.type == "agent_response":
            rec["agent_response"] = ev.data.get("response", "") or ""
        elif ev.type == "tool_calls":
            rec["tool_calls"] = ev.data.get("tool_calls", []) or []
        elif ev.type == "attempt_config":
            rec["config_after"] = ev.data.get("config", {}) or {}
        elif ev.type == "judge":
            rec["score"] = float(ev.data.get("score") or 0.0)
            rec["feedback"] = ev.data.get("feedback", {}) or {}
        elif ev.type == "attempt_end":
            rec["execution_time"] = float(ev.data.get("execution_time") or 0.0)

    gats_attempts = [
        GATSAttempt(
            index=a["index"],
            tool_calls=a["tool_calls"],
            score=a["score"],
            feedback=a["feedback"],
            config_after=a["config_after"],
            agent_response=a["agent_response"],
            execution_time=a["execution_time"],
        )
        for a in (attempts_by_idx[k] for k in sorted(attempts_by_idx))
    ]

    if (
        outcome.final_tool_calls
        and 0 <= outcome.best_attempt < len(gats_attempts)
        and not gats_attempts[outcome.best_attempt].tool_calls
    ):
        gats_attempts[outcome.best_attempt].tool_calls = deepcopy(
            outcome.final_tool_calls
        )

    return GATSTurn(
        index=outcome.turn_idx,
        question=question,
        best_attempt=outcome.best_attempt,
        score=outcome.score,
        attempts=gats_attempts,
        checklist=checklist,
        config_after=deepcopy(outcome.final_config),
        execution_time=outcome.execution_time,
        real_tool_calls=deepcopy(real_tool_calls or []),
    )




class GATSSolver:
    """Wraps the existing SimSolver with GATS types.

    One GATSSolver is created per task. Call ``process_turn(question)``
    for each turn; SimSolver maintains internal state across turns.

    When ``config.enable_real_execution`` is True (multi-turn), each turn
    follows a two-stage flow:
      1. SimSolver (mock tools) → best attempt
      2. Real task agent (real Python tools) follows the ICL plan → final answer
    """

    def __init__(
        self,
        task: GATSTask,
        *,
        model: str = "gpt-4.1-mini",
        max_retries: int = 3,
        agent_timeout: Optional[int] = None,
        gecko_url: str = "http://localhost:8000",
        override_openapi_servers: bool = True,
        agent_max_iterations: int = 10,
        enable_checklist: bool = True,
        agent_prompt: Optional[str] = None,
        judge_prompt: Optional[str] = None,
        triage_judge_prompt: Optional[str] = None,
        checklist_prompt: Optional[str] = None,
        base_checklist_items: Optional[List[str]] = None,
        agent_persistence: bool = False,
        include_agent_response_in_judge: bool = True,
        enable_tool_result_folding: bool = True,
        append_base_checklist_to_generated: bool = True,
        debug: bool = False,
        verbose: bool = False,
        enable_real_execution: bool = False,
        multi_agent_prompt: Optional[str] = None,
    ):
        self._task = task
        self._model = model
        self._precondition_index = _load_precondition_index(task.tool_schemas)
        self._state_access_index = _load_state_access_index(task.tool_schemas)
        self._state_effects_index = _load_state_effects_index(task.tool_schemas)
        test_case = _build_test_case(task)

        effective_agent_prompt = (
            task.agent_prompt
            if task.agent_prompt is not None
            else agent_prompt
        )

        self._solver = SimSolver(
            test_case=test_case,
            initial_config=task.initial_config,
            model_name=model,
            max_retries=max_retries,
            agent_timeout=agent_timeout,
            mock_server_url=gecko_url,
            override_openapi_server=override_openapi_servers,
            agent_max_iteration=agent_max_iterations,
            enable_evaluation=(max_retries > 0),
            enable_checklist=enable_checklist,
            agent_system_prompt=effective_agent_prompt,
            judge_system_prompt=judge_prompt,
            triage_judge_prompt=triage_judge_prompt,
            openapi_tool_paths=task.tool_schemas,
            agent_persistence_mode=agent_persistence,
            base_checklist_items=base_checklist_items,
            checklist_system_prompt=checklist_prompt,
            include_agent_response_in_judge=include_agent_response_in_judge,
            enable_tool_result_folding=enable_tool_result_folding,
            append_base_checklist_to_generated=append_base_checklist_to_generated,
            fetch_attempt_state=task.metadata.get("type") != "single_turn",
            enable_debug=debug,
            verbose_debug=verbose,
        )

        self._enable_real = enable_real_execution
        self._multi_agent_prompt = multi_agent_prompt or agent_prompt
        self._agent_timeout = agent_timeout
        self._agent_max_iterations = agent_max_iterations
        self._real_task_agent = None
        self._real_tools = []
        self._real_tool_instances: Dict[str, Any] = {}
        self._real_toolkit_by_name: Dict[str, str] = {}
        self._real_history_items: List[Dict[str, Any]] = []
        self._real_current_config: Dict[str, Any] = deepcopy(
            task.initial_config or {}
        )
        self._real_state_seeded = False
        self._real_session_id: Optional[str] = None

        if self._enable_real:
            self._init_real_execution(task)

    def _init_real_execution(self, task: GATSTask) -> None:
        """Initialize real Python class instances and wrap them as CAMEL tools."""
        involved_classes = task.metadata.get("involved_classes", [])
        if not involved_classes:
            logger.warning("No involved_classes for task %s; real execution disabled", task.id)
            self._enable_real = False
            return

        if _FUNC_SOURCE_DIR not in sys.path:
            sys.path.insert(0, _FUNC_SOURCE_DIR)
        self._patch_bfcl_dependencies()

        initial_config = task.initial_config or {}
        for class_name in involved_classes:
            domain = CLASS_MODULE_MAPPING.get(class_name)
            if not domain:
                logger.warning("Unknown class %s; skipping", class_name)
                continue
            try:
                module = importlib.import_module(domain["module"])
                cls = getattr(module, domain["class"])
                instance = cls()
                scenario = self._find_scenario(class_name, initial_config)
                if hasattr(instance, "_load_scenario") and isinstance(scenario, dict):
                    instance._load_scenario(deepcopy(scenario))
                self._real_tool_instances[class_name] = instance
                logger.info("Initialized real instance: %s", class_name)
            except Exception as e:
                logger.error("Failed to initialize %s: %s", class_name, e, exc_info=True)

        self._real_tools = self._build_real_tools(involved_classes)

        if self._real_tools:
            self._real_task_agent = self._create_real_task_agent()
            logger.info(
                "Real execution ready for %s: %d tools, %d instances",
                task.id, len(self._real_tools), len(self._real_tool_instances),
            )
        else:
            logger.warning("No real tools built for %s; real execution disabled", task.id)
            self._enable_real = False

    @staticmethod
    def _find_scenario(
        class_name: str, initial_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Find the scenario config for a class, trying aliases."""
        aliases = [class_name]
        domain = CLASS_MODULE_MAPPING.get(class_name, {})
        real_class = domain.get("class", class_name) if isinstance(domain, dict) else class_name
        if real_class != class_name:
            aliases.append(real_class)
        for key in aliases:
            if key in initial_config and isinstance(initial_config[key], dict):
                return initial_config[key]
        return {}

    @staticmethod
    def _patch_bfcl_dependencies() -> None:
        """Patch long_context import aliases expected by BFCL source modules."""
        source_dir = Path(_FUNC_SOURCE_DIR)
        long_context_path = source_dir / "long_context.py"
        if long_context_path.exists() and "long_context" not in sys.modules:
            import importlib.util
            spec = importlib.util.spec_from_file_location("long_context", str(long_context_path))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["long_context"] = mod
                try:
                    spec.loader.exec_module(mod)
                except Exception:
                    pass

    def _build_real_tools(self, involved_classes: List[str]) -> list:
        """Build CAMEL FunctionTools from real class instances + func_doc schemas."""
        from camel.toolkits import FunctionTool

        tools = []
        seen_names = set()
        self._real_toolkit_by_name = {}

        for class_name in involved_classes:
            instance = self._real_tool_instances.get(class_name)
            if not instance:
                continue
            toolkit_name = canonicalize_bfcl_multi_turn_toolkit(class_name) or class_name

            doc_name = CLASS_FUNC_DOC_MAPPING.get(class_name)
            if not doc_name:
                continue
            doc_path = os.path.join(_FUNC_DOC_DIR, f"{doc_name}.json")
            if not os.path.exists(doc_path):
                logger.warning("func_doc not found: %s", doc_path)
                continue

            func_docs = []
            with open(doc_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        func_docs.append(json.loads(line))
            doc_by_name = {d["name"]: d for d in func_docs}

            method_names = [
                name for name, _ in inspect.getmembers(instance, predicate=inspect.ismethod)
                if not name.startswith("_")
            ]

            for method_name in method_names:
                if method_name in seen_names:
                    continue
                doc = doc_by_name.get(method_name)
                if not doc:
                    continue

                method = getattr(instance, method_name)
                schema = _build_openai_tool_schema(doc)

                def _make_wrapper(m):
                    def wrapper(**kwargs):
                        result = m(**kwargs)
                        if result is None:
                            return {"success": True}
                        return result
                    wrapper.__name__ = m.__name__
                    wrapper.__doc__ = m.__doc__ or ""
                    return wrapper

                wrapped = _make_wrapper(method)
                tool = FunctionTool(func=wrapped, openai_tool_schema=schema)
                tool = normalize_bfcl_multi_turn_function_tool(
                    tool,
                    default_toolkit=toolkit_name,
                )
                tools.append(tool)
                seen_names.add(method_name)
                self._real_toolkit_by_name[method_name] = toolkit_name

        return tools

    def _create_real_task_agent(self):
        """Create the real task agent (ChatAgent with real tools)."""
        from inference.agents.chat_agent import ChatAgent

        agent = ChatAgent(
            model_name=self._model,
            timeout=(
                self._agent_timeout
                if self._agent_timeout is not None
                else REAL_TASK_AGENT_TIMEOUT_SECONDS
            ),
            system_message=self._multi_agent_prompt,
            max_iteration=self._agent_max_iterations,
            agent_role="real_task_agent",
        )
        agent.set_tools(self._real_tools)
        return agent

    def _extract_best_attempt(self, outcome: TurnOutcome) -> Dict[str, Any]:
        """Extract best attempt data from SimSolver outcome events."""
        best_idx = outcome.best_attempt
        best = {
            "tool_calls": [],
            "score": outcome.score,
            "feedback": {},
            "agent_response": "",
        }
        for ev in outcome.events:
            if ev.attempt is None or int(ev.attempt) != best_idx:
                continue
            if ev.type == "tool_calls":
                best["tool_calls"] = ev.data.get("tool_calls", []) or []
            elif ev.type == "judge":
                best["feedback"] = ev.data.get("feedback", {}) or {}
            elif ev.type == "agent_response":
                best["agent_response"] = ev.data.get("response", "") or ""
        return best

    def _build_best_attempt_icl(
        self,
        best_attempt: Dict[str, Any],
        current_question: str = "",
    ) -> str:
        """Build ICL text from SimSolver best attempt.

        For successful plans (score >= 1.0), render the mock plan as-is and
        separately list arguments that should be preserved or rebound during
        real execution. Keeping guidance outside the call text avoids
        teaching the real agent to copy source tags as literal arguments.
        """
        if not best_attempt:
            return ""

        score = float(best_attempt.get("score", 0.0) or 0.0)
        tool_calls = best_attempt.get("tool_calls", []) or []

        if score >= 1.0:
            question_corpus_parts: List[str] = []
            if current_question:
                question_corpus_parts.append(current_question)
            for item in self._real_history_items:
                if item.get("role") == "user":
                    content = item.get("content")
                    if isinstance(content, str) and content:
                        question_corpus_parts.append(content)
            question_corpus = "\n".join(question_corpus_parts)

            prior_results: List[Any] = []
            for item in self._real_history_items:
                if item.get("role") in ("tool_call", "tool_result"):
                    if "result" in item:
                        prior_results.append(item.get("result"))

            initial_state = getattr(self._task, "initial_config", None)

            if os.getenv("GATS_ENABLE_PRECONDITION_INJECTION") == "1":
                tool_calls = self._augment_plan_with_preconditions(
                    tool_calls, initial_state=initial_state,
                )

            rendered_lines: List[str] = []
            preserve_items: List[str] = []
            rebind_items: List[str] = []
            successful_count = 0
            seen_state_change_signatures = set()
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                if self._should_skip_plan_call_for_current_question(
                    call,
                    current_question,
                ):
                    continue
                normalized_call = normalize_bfcl_multi_turn_tool_call(
                    call,
                    rename=True,
                    preserve_original=False,
                )
                fn = str(
                    normalized_call.get("function", "")
                    or normalized_call.get("name", "")
                    or ""
                )
                args = call.get("arguments", {})
                result = call.get("result")
                err = self._extract_error_text(result)
                if err:
                    continue
                duplicate_signature = self._state_changing_duplicate_signature(call)
                if duplicate_signature and duplicate_signature in seen_state_change_signatures:
                    logger.info(
                        "Task %s: dropping duplicate state-changing plan call %s",
                        self._task.id,
                        fn,
                    )
                    continue
                if duplicate_signature:
                    seen_state_change_signatures.add(duplicate_signature)
                successful_count += 1
                if successful_count > 8:
                    break
                preserve_keys = self._append_argument_guidance_items(
                    step_number=successful_count,
                    args=args,
                    question_corpus=question_corpus,
                    prior_tool_results=prior_results,
                    initial_state=initial_state,
                    preserve_items=preserve_items,
                    rebind_items=rebind_items,
                )
                args_block = self._compact_args(
                    args,
                    max_items=6,
                    full_value_keys=preserve_keys,
                )
                rendered_lines.append(f"{successful_count}. {fn}({args_block})")
                prior_results.append(result)

            successful_section = "\n".join(rendered_lines) if rendered_lines else "<none>"
            preserve_section = "\n".join(preserve_items) if preserve_items else "- <none>"
            rebind_section = "\n".join(rebind_items) if rebind_items else "- <none>"

            icl_text = f"""Executable plan:
{successful_section}

Argument guidance:
Preserve exactly (likely user-provided text or fixed initial-state value; changing it may alter the user's intent):
{preserve_section}

Rebind from real execution (likely mock-derived or dependent on prior tool results; use real tool results when available):
{rebind_section}

Guidance:
1) Follow the plan's tool sequence and argument keys exactly. Do not add calls outside the plan or skip its prerequisites.
2) Keep values listed under Preserve exactly unchanged.
3) Treat values listed under Rebind from real execution as candidates to update from real tool results instead of trusting mock-only values.
4) Do not change argument data types (e.g. dict vs scalar); only update values.
5) Do not add optional/default arguments that are absent from the executable plan unless the user explicitly requested that value.
6) Execute each listed plan call. A read/check/list call, prior result, or current state is not a substitute for a listed state-changing call.
"""
            return "\n\n".join(
                section for section in icl_text.split("\n\n") if section.strip()
            )

        tool_calls_block = "- tool calls:\n" if tool_calls else "- tool calls: none"
        if tool_calls:
            for idx, call in enumerate(tool_calls[:8], start=1):
                normalized_call = normalize_bfcl_multi_turn_tool_call(
                    call,
                    rename=True,
                    preserve_original=False,
                )
                fn = normalized_call.get("function", "") or normalized_call.get("name", "")
                args = call.get("arguments", {})
                result = call.get("result")
                call_status = "invalid" if self._extract_error_text(result) else "valid"
                tool_calls_block += (
                    f"  {idx}. [{call_status}] {fn}({self._compact_args(args)})\n"
                    f"     result: {self._compact_result(result)}\n"
                )
            tool_calls_block = tool_calls_block.rstrip()

        feedback = best_attempt.get("feedback", {})
        unresolved_block = ""
        unresolved = self._extract_unresolved_issues(feedback, tool_calls)
        if unresolved:
            unresolved_block = "- unresolved issues from judge:\n"
            for idx, issue in enumerate(unresolved[:5], start=1):
                unresolved_block += f"  {idx}. {issue}\n"
            unresolved_block = unresolved_block.rstrip()

        icl_text = f"""Fail example

{tool_calls_block}

{unresolved_block}

- guidance: this example failed; avoid repeating invalid calls and adjust using real tool results.
"""
        return "\n\n".join(
            section for section in icl_text.split("\n\n") if section.strip()
        )

    def _build_task_agent_prompt(self, turn_question: str, icl_text: str) -> str:
        """Build the user prompt for the real task agent."""
        parts = []

        if self._real_history_items:
            parts.append("=== Previous Conversation History ===")
            for item in self._real_history_items:
                role = item.get("role", "")
                if role == "user":
                    parts.append(f"User: {item.get('content', '')}")
                elif role == "assistant":
                    parts.append(f"Assistant: {item.get('content', '')}")
                elif role == "tool_call":
                    normalized_item = normalize_bfcl_multi_turn_tool_call(
                        item,
                        rename=True,
                        preserve_original=False,
                    )
                    fn = normalized_item.get("function") or normalized_item.get("name", "unknown")
                    args = normalized_item.get("arguments") or normalized_item.get("args", {})
                    args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                    parts.append(f"Tool Call: {fn}")
                    parts.append(f"Arguments: {args_str}")
                    if "result" in normalized_item:
                        result = normalized_item["result"]
                        result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                        parts.append(f"Result: {result_str}")
            parts.append("")

        parts.append(f"Current task:\n{turn_question}")
        parts.append("")
        parts.append(
            "Argument rules:\n"
            "- Use only argument values supported by the current user request, prior real tool results, or the executable plan when one is provided.\n"
            "- Do not add optional/default arguments that are not shown in the executable plan and not explicitly requested by the user; omit empty lists, empty strings, null/None, \"None\", and default numeric values used only as placeholders.\n"
            "- For update/patch dictionaries, include only the fields the user explicitly asked to change. Pass an empty string or empty list only when the user explicitly requested that empty value."
        )

        if icl_text:
            parts.append("")
            parts.append(icl_text)

        return "\n".join(parts)

    def _execute_real_turn(
        self, turn_question: str, icl_text: str, *, turn_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run the real task agent for one turn.

        Returns dict with 'tool_calls' and 'assistant_text'.
        """
        prompt = self._build_task_agent_prompt(turn_question, icl_text)

        if hasattr(self._real_task_agent, "set_log_context"):
            self._real_task_agent.set_log_context(turn_idx=turn_idx, attempt=None)
        with bind_log_context(task_id=self._task.id):
            response = self._real_task_agent.generate_response(
                prompt, context={"temperature": 0.1}
            )

        tool_calls = []
        assistant_text = ""
        if response:
            tool_calls = [
                self._normalize_real_tool_call(tc)
                for tc in (response.tool_calls or [])
            ]
            assistant_text = response.raw_response or ""

        return {
            "tool_calls": tool_calls,
            "assistant_text": assistant_text,
        }

    @staticmethod
    def _flat_arguments(arguments: Any) -> Dict[str, Any]:
        if isinstance(arguments, dict) and isinstance(arguments.get("requestBody"), dict):
            return dict(arguments["requestBody"])
        if isinstance(arguments, dict):
            return dict(arguments)
        return {}

    @classmethod
    def _canonicalize_duplicate_arg_value(cls, value: Any) -> Any:
        if isinstance(value, bool):
            return value
        if isinstance(value, float):
            return int(value) if value.is_integer() else value
        if isinstance(value, dict):
            return {
                key: cls._canonicalize_duplicate_arg_value(val)
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [cls._canonicalize_duplicate_arg_value(item) for item in value]
        return value

    @staticmethod
    def _with_flat_arguments(original_arguments: Any, flat_arguments: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(original_arguments, dict) and isinstance(original_arguments.get("requestBody"), dict):
            updated = dict(original_arguments)
            updated["requestBody"] = dict(flat_arguments)
            return updated
        return dict(flat_arguments)

    @staticmethod
    def _call_function_name(call: Dict[str, Any]) -> str:
        normalized = normalize_bfcl_multi_turn_tool_call(
            call,
            rename=True,
            preserve_original=False,
        )
        return str(normalized.get("function") or normalized.get("name") or "").strip()

    def _real_instance_for_call(
        self,
        function_name: str,
        toolkit_name: Optional[str],
    ) -> Optional[Any]:
        candidates: List[str] = []
        if toolkit_name:
            candidates.append(toolkit_name)
        mapped_toolkit = self._real_toolkit_by_name.get(function_name)
        if mapped_toolkit and mapped_toolkit not in candidates:
            candidates.append(mapped_toolkit)

        for candidate in candidates:
            instance = self._real_tool_instances.get(candidate)
            if instance is not None and hasattr(instance, function_name):
                return instance

        canonical_candidates = {
            canonicalize_bfcl_multi_turn_toolkit(candidate) or candidate
            for candidate in candidates
            if candidate
        }
        for class_name, instance in self._real_tool_instances.items():
            canonical_class = canonicalize_bfcl_multi_turn_toolkit(class_name) or class_name
            if canonical_class in canonical_candidates and hasattr(instance, function_name):
                return instance

        for instance in self._real_tool_instances.values():
            if hasattr(instance, function_name):
                return instance
        return None

    def _execute_plan_calls_on_real_tools(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        executed: List[Dict[str, Any]] = []
        error_count = 0

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            normalized = self._normalize_real_tool_call(tool_call)
            function_name = self._call_function_name(normalized)
            if not function_name:
                continue

            arguments = self._flat_arguments(normalized.get("arguments", {}))
            normalized["function"] = function_name
            normalized["arguments"] = deepcopy(arguments)

            instance = self._real_instance_for_call(
                function_name,
                normalized.get("toolkit"),
            )
            if instance is None:
                result = {"error": f"No real tool instance found for {function_name}"}
                error_count += 1
            else:
                try:
                    method = getattr(instance, function_name)
                    result = method(**deepcopy(arguments))
                    if result is None:
                        result = {"success": True}
                except Exception as exc:
                    logger.warning(
                        "Fallback real execution failed for %s.%s: %s",
                        normalized.get("toolkit") or "unknown",
                        function_name,
                        exc,
                    )
                    result = {"error": str(exc)}
                    error_count += 1

            normalized["result"] = deepcopy(result)
            executed.append(normalized)

        return executed, error_count

    def _is_read_only_function(self, function_name: str) -> bool:
        state_access = self._state_access_index.get(function_name)
        if state_access == "read":
            return True
        if state_access == "write":
            return False
        return function_name in {
            "ls",
            "pwd",
            "find",
            "cat",
            "wc",
            "grep",
            "diff",
            "head",
            "tail",
            "sort",
            "mean",
            "calculate",
            "convert",
            "get",
            "check",
            "list",
            "view",
            "display",
            "search",
            "estimate",
            "posting_get_login_status",
            "view_messages_sent",
            "view_messages_received",
            "get_user_id",
            "list_users",
        }

    def _is_state_changing_function(self, function_name: str) -> bool:
        return bool(self._state_effects_index.get(function_name))

    def _state_changing_duplicate_signature(
        self,
        call: Dict[str, Any],
    ) -> Optional[Tuple[str, str, str]]:
        function_name = self._call_function_name(call)
        if not function_name or not self._is_state_changing_function(function_name):
            return None
        normalized = normalize_bfcl_multi_turn_tool_call(
            call,
            rename=True,
            preserve_original=False,
        )
        toolkit = str(normalized.get("toolkit") or "")
        arguments = call.get("arguments", normalized.get("arguments", {})) or {}
        flat_arguments = self._canonicalize_duplicate_arg_value(
            self._flat_arguments(arguments)
        )
        try:
            argument_signature = json.dumps(
                flat_arguments,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except TypeError:
            argument_signature = repr(sorted(flat_arguments.items()))
        return toolkit, function_name, argument_signature

    @staticmethod
    def _question_requests_read_only_derived_result(question: str) -> bool:
        q = (question or "").lower()
        return any(
            marker in q
            for marker in (
                "sort",
                "sorted",
                "alphabet",
                "compare",
                "difference",
                "diff",
                "grep",
                "search",
                "count",
                "how many",
                "line",
                "word",
                "character",
                "display",
                "show",
                "read",
                "tail",
                "head",
                "average",
                "mean",
            )
        )

    @staticmethod
    def _question_requests_persistence(question: str) -> bool:
        q = (question or "").lower()
        return any(
            marker in q
            for marker in (
                "save",
                "write",
                "store",
                "persist",
                "overwrite",
                "update",
                "replace",
                "modify",
                "record",
                "jot down",
                "put ",
                "into the file",
                "in the file",
            )
        )

    def _should_skip_plan_call_for_current_question(
        self,
        call: Dict[str, Any],
        current_question: str,
    ) -> bool:
        function_name = self._call_function_name(call)
        if function_name != "echo":
            return False
        if not self._question_requests_read_only_derived_result(current_question):
            return False
        return not self._question_requests_persistence(current_question)

    def _successful_plan_calls(
        self,
        best_attempt: Dict[str, Any],
        current_question: str = "",
    ) -> List[Dict[str, Any]]:
        plan_calls: List[Dict[str, Any]] = []
        seen_state_change_signatures = set()
        for call in best_attempt.get("tool_calls", []) or []:
            if not isinstance(call, dict):
                continue
            if self._should_skip_plan_call_for_current_question(call, current_question):
                continue
            if self._extract_error_text(call.get("result")):
                continue
            normalized = normalize_bfcl_multi_turn_tool_call(
                call,
                rename=True,
                preserve_original=True,
            )
            function_name = self._call_function_name(normalized)
            if not function_name:
                continue
            normalized["function"] = function_name
            normalized["arguments"] = deepcopy(call.get("arguments", normalized.get("arguments", {})) or {})
            duplicate_signature = self._state_changing_duplicate_signature(normalized)
            if duplicate_signature and duplicate_signature in seen_state_change_signatures:
                logger.info(
                    "Task %s: dropping duplicate state-changing plan call %s",
                    self._task.id,
                    function_name,
                )
                continue
            if duplicate_signature:
                seen_state_change_signatures.add(duplicate_signature)
            plan_calls.append(normalized)
            if len(plan_calls) >= 8:
                break
        return plan_calls

    @classmethod
    def _clean_real_call_against_plan(
        cls,
        real_call: Dict[str, Any],
        plan_call: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], int]:
        cleaned = deepcopy(real_call)
        real_args_original = cleaned.get("arguments", {})
        real_args = cls._flat_arguments(real_args_original)
        allowed = set(cls._flat_arguments(plan_call.get("arguments", {})).keys())
        if not allowed:
            return cleaned, 0

        stripped = 0
        filtered: Dict[str, Any] = {}
        plan_args = cls._flat_arguments(plan_call.get("arguments", {}))
        for key, value in real_args.items():
            if key in allowed:
                plan_value = plan_args.get(key)
                if isinstance(value, dict) and isinstance(plan_value, dict):
                    filtered_value, nested_stripped = cls._filter_nested_dict_against_plan(
                        value,
                        plan_value,
                    )
                    filtered[key] = filtered_value
                    stripped += nested_stripped
                else:
                    filtered[key] = value
            else:
                stripped += 1

        cleaned["arguments"] = cls._with_flat_arguments(real_args_original, filtered)
        return cleaned, stripped

    @classmethod
    def _filter_nested_dict_against_plan(
        cls,
        real_value: Dict[str, Any],
        plan_value: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], int]:
        filtered: Dict[str, Any] = {}
        stripped = 0
        for key, value in real_value.items():
            if key not in plan_value:
                stripped += 1
                continue
            expected = plan_value[key]
            if isinstance(value, dict) and isinstance(expected, dict):
                nested_value, nested_stripped = cls._filter_nested_dict_against_plan(
                    value,
                    expected,
                )
                filtered[key] = nested_value
                stripped += nested_stripped
            else:
                filtered[key] = value
        return filtered, stripped

    @classmethod
    def _plan_arguments_match(cls, real_call: Dict[str, Any], plan_call: Dict[str, Any]) -> bool:
        real_args = cls._flat_arguments(real_call.get("arguments", {}))
        plan_args = cls._flat_arguments(plan_call.get("arguments", {}))
        for key, expected in plan_args.items():
            if key not in real_args:
                return False
            actual = real_args[key]
            if actual != expected:
                return False
        return True

    def _align_real_calls_to_plan(
        self,
        real_tool_calls: List[Dict[str, Any]],
        plan_calls: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not real_tool_calls or not plan_calls:
            return real_tool_calls, {}

        projected: List[Dict[str, Any]] = []
        plan_idx = 0
        matched = 0
        dropped_read_only = 0
        stripped_args = 0
        appended_missing_read_only = 0
        kept_unplanned_state_changes = 0

        for real_call in real_tool_calls:
            real_function = self._call_function_name(real_call)
            if plan_idx < len(plan_calls):
                plan_call = plan_calls[plan_idx]
                if real_function == self._call_function_name(plan_call):
                    cleaned_call, stripped = self._clean_real_call_against_plan(
                        real_call,
                        plan_call,
                    )
                    if self._plan_arguments_match(cleaned_call, plan_call):
                        projected.append(cleaned_call)
                        stripped_args += stripped
                        plan_idx += 1
                        matched += 1
                        continue

            if self._is_read_only_function(real_function):
                dropped_read_only += 1
                continue

            projected.append(real_call)
            kept_unplanned_state_changes += 1

        while plan_idx < len(plan_calls):
            missing_plan_call = deepcopy(plan_calls[plan_idx])
            missing_function = self._call_function_name(missing_plan_call)
            if not self._is_read_only_function(missing_function):
                break
            projected.append(missing_plan_call)
            appended_missing_read_only += 1
            plan_idx += 1

        if matched == 0:
            return real_tool_calls, {}

        metadata = {
            "real_plan_matched_calls": matched,
            "real_plan_expected_calls": len(plan_calls),
            "real_plan_dropped_read_only_extras": dropped_read_only,
            "real_plan_appended_missing_read_only": appended_missing_read_only,
            "real_plan_stripped_unplanned_args": stripped_args,
            "real_plan_kept_unplanned_state_changes": kept_unplanned_state_changes,
            "real_plan_missing_calls": max(0, len(plan_calls) - plan_idx),
            "real_plan_next_index": plan_idx,
            "real_plan_missing_suffix_functions": [
                self._call_function_name(call) for call in plan_calls[plan_idx:]
            ],
        }
        return projected, metadata

    def _is_declared_precondition_remedy_call(self, call: Dict[str, Any]) -> bool:
        function_name = self._call_function_name(call)
        if not function_name:
            return False

        call_args = self._flat_arguments(call.get("arguments", {}))
        for entries in self._precondition_index.values():
            for entry in entries:
                remedy = entry.get("remedy")
                if not isinstance(remedy, dict):
                    continue
                if self._call_matches_remedy(function_name, call_args, remedy):
                    return True
        return False

    @staticmethod
    def _call_matches_remedy(
        function_name: str,
        call_args: Dict[str, Any],
        remedy: Dict[str, Any],
    ) -> bool:
        return remedy.get("name") == function_name and dict(remedy.get("arguments") or {}) == call_args

    def _preconditions_satisfied_by_calls_or_state(
        self,
        call: Dict[str, Any],
        prior_calls: List[Dict[str, Any]],
    ) -> bool:
        function_name = self._call_function_name(call)
        if not function_name:
            return False
        preconditions = self._precondition_index.get(function_name)
        if not preconditions:
            return False

        for entry in preconditions:
            if self._precondition_met_by_state(
                self._real_current_config,
                entry.get("toolkit", ""),
                entry.get("state_path", ""),
                entry.get("required_value"),
            ):
                continue

            remedy = entry.get("remedy")
            if not isinstance(remedy, dict):
                return False
            remedy_seen = False
            for prior_call in prior_calls:
                prior_function = self._call_function_name(prior_call)
                prior_args = self._flat_arguments(prior_call.get("arguments", {}))
                if self._call_matches_remedy(prior_function, prior_args, remedy):
                    remedy_seen = True
                    break
            if not remedy_seen:
                return False
        return True

    def _is_safe_partial_suffix_call(
        self,
        call: Dict[str, Any],
        prior_calls: List[Dict[str, Any]],
    ) -> bool:
        function_name = self._call_function_name(call)
        if not function_name:
            return False
        if self._is_read_only_function(function_name):
            return True
        if self._preconditions_satisfied_by_calls_or_state(call, prior_calls):
            return True
        return self._is_declared_precondition_remedy_call(call)

    def _repair_missing_plan_suffix(
        self,
        real_tool_calls: List[Dict[str, Any]],
        plan_calls: List[Dict[str, Any]],
        alignment_meta: Dict[str, Any],
        *,
        outcome_score: float,
        turn_idx: Optional[int],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not alignment_meta or outcome_score < 1.0:
            return real_tool_calls, {}

        missing_count = int(alignment_meta.get("real_plan_missing_calls", 0) or 0)
        if missing_count <= 0:
            return real_tool_calls, {}

        repair_meta: Dict[str, Any] = {}

        def skip(reason: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
            repair_meta["real_partial_plan_repair_skipped"] = reason
            return real_tool_calls, repair_meta

        if int(alignment_meta.get("real_plan_matched_calls", 0) or 0) <= 0:
            return skip("no_matched_prefix")
        if int(alignment_meta.get("real_plan_kept_unplanned_state_changes", 0) or 0) > 0:
            return skip("unplanned_state_changing_extra")
        if int(alignment_meta.get("real_plan_stripped_unplanned_args", 0) or 0) > 0:
            return skip("stripped_unplanned_args")
        if int(alignment_meta.get("real_plan_appended_missing_read_only", 0) or 0) > 0:
            return skip("appended_read_only_before_suffix")

        next_idx = int(alignment_meta.get("real_plan_next_index", len(plan_calls)) or len(plan_calls))
        if next_idx < 0 or next_idx >= len(plan_calls):
            return skip("invalid_missing_suffix")

        missing_suffix = [deepcopy(call) for call in plan_calls[next_idx:]]
        if len(missing_suffix) != missing_count:
            return skip("non_contiguous_missing_suffix")
        if not any(
            not self._is_read_only_function(self._call_function_name(call))
            for call in missing_suffix
        ):
            return skip("read_only_suffix_only")
        safe_prior_calls = list(real_tool_calls)
        for missing_call in missing_suffix:
            if not self._is_safe_partial_suffix_call(missing_call, safe_prior_calls):
                return skip("unsafe_suffix_arguments")
            safe_prior_calls.append(missing_call)

        repaired_calls, repair_errors = self._execute_plan_calls_on_real_tools(missing_suffix)
        repair_meta.update(
            {
                "real_partial_plan_repair": True,
                "real_partial_plan_repair_calls": len(repaired_calls),
                "real_partial_plan_repair_errors": repair_errors,
                "real_partial_plan_repair_functions": [
                    self._call_function_name(call) for call in repaired_calls
                ],
            }
        )
        if not repaired_calls:
            repair_meta["real_partial_plan_repair_skipped"] = "no_calls_executed"
            return real_tool_calls, repair_meta

        logger.info(
            "Task %s turn %s: repaired missing real-plan suffix with %d call(s): %s",
            self._task.id,
            "na" if turn_idx is None else turn_idx,
            len(repaired_calls),
            ", ".join(repair_meta["real_partial_plan_repair_functions"]),
        )
        return real_tool_calls + repaired_calls, repair_meta

    def _append_real_history(
        self,
        user_message: str,
        tool_calls: List[Dict[str, Any]],
        assistant_text: str,
    ) -> None:
        """Append one turn's results to real conversation history."""
        self._real_history_items.append({"role": "user", "content": user_message})
        for tc in tool_calls:
            normalized_call = self._normalize_real_tool_call(tc)
            fn = normalized_call.get("function") or normalized_call.get("name", "")
            args = normalized_call.get("arguments", {})
            result = deepcopy(tc.get("result"))
            entry: Dict[str, Any] = {
                "role": "tool_call",
                "function": fn,
                "arguments": deepcopy(args),
            }
            if normalized_call.get("toolkit"):
                entry["toolkit"] = normalized_call["toolkit"]
            if result is not None:
                entry["result"] = result
            self._real_history_items.append(entry)
        cleaned_assistant = strip_reasoning(assistant_text)
        if cleaned_assistant:
            self._real_history_items.append({"role": "assistant", "content": cleaned_assistant})

    def _normalize_real_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize real-stage tool calls to bare names plus explicit toolkit."""
        normalized = normalize_bfcl_multi_turn_tool_call(
            deepcopy(tool_call),
            rename=True,
            preserve_original=True,
        )
        function_name = str(normalized.get("function") or normalized.get("name") or "").strip()
        if function_name and not normalized.get("toolkit"):
            toolkit_name = self._real_toolkit_by_name.get(function_name)
            if toolkit_name:
                normalized["toolkit"] = toolkit_name
        return normalized

    @staticmethod
    def _compact_args(
        arguments: Any,
        max_items: int = 4,
        full_value_keys: Optional[set[str]] = None,
    ) -> str:
        args = arguments
        if isinstance(arguments, dict) and "requestBody" in arguments:
            rb = arguments["requestBody"]
            if isinstance(rb, dict):
                args = rb
        if not isinstance(args, dict):
            return ""
        items = []
        full_value_keys = full_value_keys or set()
        for idx, (k, v) in enumerate(args.items()):
            if idx >= max_items:
                items.append("...")
                break
            if isinstance(v, str) and len(v) > 60 and k not in full_value_keys:
                v = v[:57] + "..."
            items.append(f"{k}={repr(v)}")
        return ", ".join(items)

    @staticmethod
    def _scalar_appears_in(value: Any, container: Any) -> bool:
        """Recursively test whether ``value`` appears as a scalar anywhere inside ``container``.

        Used by the ICL argument classifier to decide whether a plan argument
        came from a prior tool call's result. Treats ``bool`` as a distinct type
        from ``int`` so ``True`` does not match ``1``.
        """
        if container is None:
            return False
        if isinstance(container, bool):
            return isinstance(value, bool) and value is container
        if isinstance(container, (str, int, float)):
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                return False
            return value == container or str(value) == str(container)
        if isinstance(container, list):
            return any(
                GATSSolver._scalar_appears_in(value, item) for item in container
            )
        if isinstance(container, dict):
            return any(
                GATSSolver._scalar_appears_in(value, v) for v in container.values()
            )
        return False

    @staticmethod
    def _classify_argument_source(
        value: Any,
        question_corpus: str,
        prior_tool_results: List[Any],
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Classify a plan argument as one of four source kinds.

        Rules (in priority order):

        1. Non-scalar or boolean → mock_derived (can't confidently classify).
        2. Scalar appears verbatim in the user's message(s) → user_literal.
        3. Scalar appears anywhere inside ``initial_state`` → initial_state.
           This identifies values already present in the conversation's starting
           config so they can be preserved exactly when used as arguments.
        4. Scalar appears inside any prior tool call's result → from_prior_result.
        5. Otherwise → mock_derived.
        """
        if value is None or isinstance(value, bool):
            return "mock_derived"
        if not isinstance(value, (str, int, float)):
            return "mock_derived"

        value_str = str(value).strip()
        if not value_str:
            return "mock_derived"

        if question_corpus:
            if isinstance(value, str) and len(value_str) >= 2:
                if value_str in question_corpus:
                    return "user_literal"
            else:
                pattern = rf"(?<![0-9.\-]){re.escape(value_str)}(?![0-9.])"
                if re.search(pattern, question_corpus):
                    return "user_literal"

        if isinstance(initial_state, dict) and initial_state:
            if GATSSolver._scalar_appears_in(value, initial_state):
                return "initial_state"

        for result in prior_tool_results:
            if GATSSolver._scalar_appears_in(value, result):
                return "from_prior_result"

        return "mock_derived"

    @staticmethod
    def _precondition_met_by_state(
        state: Optional[Dict[str, Any]],
        toolkit: str,
        state_path: str,
        required_value: Any,
    ) -> bool:
        """Return True if the given initial/current state already satisfies the
        precondition. The state_path is relative to the toolkit's top-level
        entry (e.g. ``brakePedalStatus`` looks up
        ``state[toolkit][brakePedalStatus]``)."""
        if not isinstance(state, dict) or not toolkit:
            return False
        root = state.get(toolkit)
        if not isinstance(root, dict):
            return False
        parts = [p for p in state_path.split("/") if p]
        current: Any = root
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]
        return current == required_value

    def _augment_plan_with_preconditions(
        self,
        tool_calls: List[Dict[str, Any]],
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Auto-insert declared precondition remedies into the plan.

        For every call in the plan whose OpenAPI schema declares a
        ``preconditions`` entry (see ``data/bfcl/openapi/multi_turn/*.json`` under
        ``info.x-default-state.tools.<name>.preconditions``), we check:

        - Is the required state already satisfied by ``initial_state``?
        - Does an earlier plan call invoke the declared ``remedy.name``?

        If neither, we insert the remedy call immediately before the dependent
        call. Generic (no toolkit branching); unknown tools are untouched.
        """
        index = self._precondition_index
        if not index:
            return tool_calls

        augmented: List[Dict[str, Any]] = []
        satisfied: set = set()

        for call in tool_calls:
            if not isinstance(call, dict):
                augmented.append(call)
                continue
            call_name = str(call.get("name") or call.get("function") or "").strip()
            if call_name in index:
                for pc in index[call_name]:
                    remedy_name = pc["remedy"]["name"]
                    if remedy_name in satisfied:
                        continue
                    if any(
                        isinstance(prev, dict)
                        and str(prev.get("name") or prev.get("function") or "").strip() == remedy_name
                        for prev in augmented
                    ):
                        satisfied.add(remedy_name)
                        continue
                    if self._precondition_met_by_state(
                        initial_state, pc["toolkit"], pc["state_path"], pc["required_value"],
                    ):
                        satisfied.add(remedy_name)
                        continue
                    remedy_call = {
                        "name": remedy_name,
                        "arguments": dict(pc["remedy"]["arguments"]),
                        "toolkit": pc["toolkit"],
                        "result": {},  # no mock result; will be produced by real agent
                        "_precondition_injected": True,
                    }
                    logger.info(
                        "[ICL] Injected precondition remedy %s before %s "
                        "(rationale: %s)",
                        remedy_name, call_name, pc.get("rationale", ""),
                    )
                    augmented.append(remedy_call)
                    satisfied.add(remedy_name)
            if call_name:
                satisfied.add(call_name)
            augmented.append(call)
        return augmented

    @staticmethod
    def _iter_scalar_values(value: Any) -> List[Any]:
        if value is None or isinstance(value, (str, int, float, bool)):
            return [value]
        if isinstance(value, list):
            values: List[Any] = []
            for item in value:
                values.extend(GATSSolver._iter_scalar_values(item))
            return values
        if isinstance(value, dict):
            values = []
            for item in value.values():
                values.extend(GATSSolver._iter_scalar_values(item))
            return values
        return []

    def _argument_guidance_bucket(
        self,
        value: Any,
        question_corpus: str,
        prior_tool_results: List[Any],
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        scalar_values = self._iter_scalar_values(value)
        if not scalar_values:
            return "rebind"

        sources = {
            self._classify_argument_source(
                scalar,
                question_corpus,
                prior_tool_results,
                initial_state=initial_state,
            )
            for scalar in scalar_values
        }
        if sources and sources <= {"user_literal", "initial_state"}:
            return "preserve"
        return "rebind"

    def _append_argument_guidance_items(
        self,
        step_number: int,
        args: Any,
        question_corpus: str,
        prior_tool_results: List[Any],
        initial_state: Optional[Dict[str, Any]],
        preserve_items: List[str],
        rebind_items: List[str],
    ) -> set[str]:
        if isinstance(args, dict) and "requestBody" in args:
            rb = args["requestBody"]
            if isinstance(rb, dict):
                args = rb
        if not isinstance(args, dict):
            return set()
        preserve_keys: set[str] = set()
        for idx, (k, v) in enumerate(args.items()):
            if idx >= 6:
                break
            bucket = self._argument_guidance_bucket(
                v,
                question_corpus,
                prior_tool_results,
                initial_state=initial_state,
            )
            if bucket == "preserve":
                value_text = self._compact_args(
                    {k: v},
                    max_items=1,
                    full_value_keys={k},
                )
                item = f"- step {step_number} {value_text}"
                preserve_items.append(item)
                preserve_keys.add(k)
            else:
                value_text = self._compact_args({k: v}, max_items=1)
                item = f"- step {step_number} {value_text}"
                rebind_items.append(item)
        return preserve_keys

    @staticmethod
    def _compact_result(result: Any) -> str:
        if result is None:
            return "(none)"
        s = str(result)
        if len(s) > 120:
            return s[:117] + "..."
        return s

    @staticmethod
    def _extract_error_text(result: Any) -> Optional[str]:
        if isinstance(result, dict):
            if "error" in result and result["error"]:
                return str(result["error"])
            detail = result.get("detail")
            if isinstance(detail, dict) and detail.get("error_message"):
                return str(detail["error_message"])
            if isinstance(detail, str) and detail:
                return detail
            if result.get("success") is False:
                return str(result.get("message") or "success=false")
        if isinstance(result, str) and "error" in result.lower():
            return result
        return None

    @staticmethod
    def _extract_unresolved_issues(
        feedback: Dict[str, Any], tool_calls: List[Dict[str, Any]]
    ) -> List[str]:
        issues = []
        failed_items = feedback.get("failed_items", []) if isinstance(feedback, dict) else []
        for item in failed_items:
            if isinstance(item, dict):
                desc = item.get("description")
                if desc:
                    issues.append(str(desc))
        for call in tool_calls:
            result = call.get("result")
            err = GATSSolver._extract_error_text(result)
            if err:
                issues.append(err)
        seen = set()
        deduped = []
        for issue in issues:
            if issue not in seen:
                seen.add(issue)
                deduped.append(issue)
        return deduped

    def process_turn(self, question: str) -> GATSTurn:
        self._solver.ensure_task_initialized()
        if self._enable_real and not self._real_state_seeded:
            self._real_current_config = self._solver.current_config
            self._real_state_seeded = True
        if self._enable_real and self._real_history_items:
            self._solver.set_history(self._real_history_items)
            self._solver.set_initial_config(self._real_current_config)

        outcome: TurnOutcome = self._solver.process(question)

        if not self._enable_real:
            return _outcome_to_gats_turn(outcome, question)

        best_attempt = self._extract_best_attempt(outcome)
        plan_calls: List[Dict[str, Any]] = []
        if outcome.score >= 1.0:
            plan_calls = self._successful_plan_calls(
                best_attempt,
                current_question=question,
            )
            icl_text = self._build_best_attempt_icl(
                best_attempt, current_question=question
            )
            logger.info(
                "Task %s turn %d: SimSolver score=%.2f, building ICL plan for real agent",
                self._task.id, outcome.turn_idx, outcome.score,
            )
        else:
            icl_text = ""
            logger.info(
                "Task %s turn %d: SimSolver score=%.2f < 1.0, skipping ICL (vanilla fallback)",
                self._task.id, outcome.turn_idx, outcome.score,
            )

        real_result = self._execute_real_turn(question, icl_text, turn_idx=outcome.turn_idx)
        real_tool_calls = real_result.get("tool_calls", [])
        assistant_text = real_result.get("assistant_text", "")

        mock_calls = best_attempt.get("tool_calls", [])
        if not real_tool_calls and mock_calls:
            logger.warning(
                "Real agent made no tool calls for %s turn %d; "
                "falling back to executing SimSolver best attempt calls on real tools",
                self._task.id, outcome.turn_idx,
            )
            real_tool_calls, _ = self._execute_plan_calls_on_real_tools(mock_calls)
            assistant_text = ""

        if plan_calls and real_tool_calls:
            aligned_tool_calls, alignment_meta = self._align_real_calls_to_plan(
                real_tool_calls,
                plan_calls,
            )
            if alignment_meta:
                real_tool_calls = aligned_tool_calls
                logger.info(
                    "Task %s turn %d: aligned real calls to plan "
                    "(matched=%s/%s dropped_read_only=%s appended_read_only=%s stripped_args=%s missing=%s)",
                    self._task.id,
                    outcome.turn_idx,
                    alignment_meta.get("real_plan_matched_calls", 0),
                    alignment_meta.get("real_plan_expected_calls", 0),
                    alignment_meta.get("real_plan_dropped_read_only_extras", 0),
                    alignment_meta.get("real_plan_appended_missing_read_only", 0),
                    alignment_meta.get("real_plan_stripped_unplanned_args", 0),
                    alignment_meta.get("real_plan_missing_calls", 0),
                )
                repaired_tool_calls, repair_meta = self._repair_missing_plan_suffix(
                    real_tool_calls,
                    plan_calls,
                    alignment_meta,
                    outcome_score=outcome.score,
                    turn_idx=outcome.turn_idx,
                )
                if repair_meta:
                    if repair_meta.get("real_partial_plan_repair"):
                        real_tool_calls = repaired_tool_calls
                        assistant_text = ""

        self._append_real_history(question, real_tool_calls, assistant_text)

        if real_tool_calls:
            try:
                mock_client = self._solver.mock_client
                if not self._real_session_id:
                    self._real_session_id = mock_client.create_session()
                    mock_client.set_session_state(
                        self._real_session_id,
                        self._real_current_config,
                    )
                    logger.info(
                        "Task %s: created persistent real session %s",
                        self._task.id, self._real_session_id,
                    )
                updated_state = mock_client.sync_state_from_real_results(
                    base_state=self._real_current_config,
                    tool_calls=real_tool_calls,
                    session_id=self._real_session_id,
                )
                if isinstance(updated_state, dict) and updated_state:
                    self._real_current_config = updated_state
                    logger.info(
                        "Task %s turn %d: synced real state (%d top-level keys)",
                        self._task.id, outcome.turn_idx, len(updated_state),
                    )
            except Exception as e:
                logger.warning(
                    "Task %s turn %d: failed to sync real state: %s",
                    self._task.id, outcome.turn_idx, e,
                )

        logger.info(
            "Task %s turn %d: real agent made %d tool calls",
            self._task.id, outcome.turn_idx, len(real_tool_calls),
        )

        return _outcome_to_gats_turn(outcome, question, real_tool_calls=real_tool_calls)

    def get_events(self) -> List[SimEvent]:
        """Return all accumulated SimEvents (deep copy)."""
        return self._solver.get_events()

    @property
    def current_config(self) -> Dict[str, Any]:
        return self._solver.current_config

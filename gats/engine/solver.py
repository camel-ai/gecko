import copy
import importlib
import inspect
import json
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from benchmarks.base.test_case import TestCase
from inference.core.sim_solver import SimSolver, TurnOutcome, SimEvent

from gats.core.task import GATSTask, GATSAttempt, GATSTurn

logger = logging.getLogger(__name__)

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

# func_doc filename for each class (used for OpenAI tool schemas)
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
    _PROJECT_ROOT, "data", "bfcl_v4", "multi_turn_func_doc"
)


# ---------------------------------------------------------------------------
# Schema helpers (for wrapping real tools as CAMEL FunctionTools)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        real_tool_calls=real_tool_calls or [],
    )


# ---------------------------------------------------------------------------
# GATSSolver
# ---------------------------------------------------------------------------


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
        collect_gecko_usage: bool = True,
        debug: bool = False,
        verbose: bool = False,
        enable_real_execution: bool = False,
        multi_agent_prompt: Optional[str] = None,
    ):
        self._task = task
        self._model = model
        test_case = _build_test_case(task)

        # Per-task agent prompt takes precedence over config-level prompt.
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
            collect_mock_server_usage=collect_gecko_usage,
            fetch_attempt_state=task.metadata.get("type") != "single_turn",
            enable_debug=debug,
            verbose_debug=verbose,
        )

        # --- Real-tool execution state (multi-turn only) ---
        self._enable_real = enable_real_execution
        self._multi_agent_prompt = multi_agent_prompt or agent_prompt
        self._agent_timeout = agent_timeout
        self._agent_max_iterations = agent_max_iterations
        self._real_task_agent = None
        self._real_tools = []
        self._real_tool_instances: Dict[str, Any] = {}
        self._real_history_items: List[Dict[str, Any]] = []
        self._real_current_config: Dict[str, Any] = deepcopy(
            task.initial_config or {}
        )
        self._real_session_id: Optional[str] = None

        if self._enable_real:
            self._init_real_execution(task)

    # ------------------------------------------------------------------
    # Real tool initialization
    # ------------------------------------------------------------------

    def _init_real_execution(self, task: GATSTask) -> None:
        """Initialize real Python class instances and wrap them as CAMEL tools."""
        involved_classes = task.metadata.get("involved_classes", [])
        if not involved_classes:
            logger.warning("No involved_classes for task %s; real execution disabled", task.id)
            self._enable_real = False
            return

        # Ensure func_source_code is on sys.path
        if _FUNC_SOURCE_DIR not in sys.path:
            sys.path.insert(0, _FUNC_SOURCE_DIR)
        self._patch_bfcl_dependencies()

        # 1. Instantiate real Python classes with initial config
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

        # 2. Load func_docs and wrap as CAMEL FunctionTools
        self._real_tools = self._build_real_tools(involved_classes)

        # 3. Create the real task agent
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

        for class_name in involved_classes:
            instance = self._real_tool_instances.get(class_name)
            if not instance:
                continue

            # Load func_doc for schema
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

            # Discover public methods
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
                tools.append(tool)
                seen_names.add(method_name)

        return tools

    def _create_real_task_agent(self):
        """Create the real task agent (ChatAgent with real tools)."""
        from inference.agents.chat_agent import ChatAgent

        agent = ChatAgent(
            model_name=self._model,
            timeout=self._agent_timeout or 120,
            system_message=self._multi_agent_prompt,
            max_iteration=self._agent_max_iterations,
            agent_role="real_task_agent",
        )
        agent.set_tools(self._real_tools)
        return agent

    # ------------------------------------------------------------------
    # ICL plan construction (from old multi_turn_executor._build_best_attempt_icl)
    # ------------------------------------------------------------------

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

    def _build_best_attempt_icl(self, best_attempt: Dict[str, Any]) -> str:
        """Build ICL text from SimSolver best attempt.

        Ported from FuncCallDataGen/inference/multi_turn_executor.py.
        """
        if not best_attempt:
            return ""

        score = float(best_attempt.get("score", 0.0) or 0.0)
        tool_calls = best_attempt.get("tool_calls", []) or []

        if score >= 1.0:
            # Successful plan: list successful calls as executable plan
            successful_calls = ""
            successful_count = 0
            for call in tool_calls[:8]:
                if not isinstance(call, dict):
                    continue
                fn = str(call.get("function", "") or call.get("name", "") or "")
                args = call.get("arguments", {})
                signature = f"{fn}({self._compact_args(args)})"
                result = call.get("result")
                err = self._extract_error_text(result)
                if not err:
                    successful_count += 1
                    successful_calls += f"{successful_count}. {signature}\n"

            successful_section = successful_calls.rstrip() if successful_calls else "<none>"

            icl_text = f"""Executable plan:
{successful_section}

Guidance:
1) The executable plan could solve the given task. Follow it strictly.
2) Update arguments in the plan using real tool results from your tool calls, since the plan may contain outdated values. Do not change the tool call sequence or add new calls that are not in the plan. Do not change the data types of the arguments (e.g., dict, float, int); only update the values as needed.
3) Keep each string argument exactly as written in the tool result or the plan. Do not add or remove any quotes, escapes, whitespace, and line breaks.
"""
            return "\n\n".join(
                section for section in icl_text.split("\n\n") if section.strip()
            )

        # Failed plan: show what went wrong
        tool_calls_block = "- tool calls:\n" if tool_calls else "- tool calls: none"
        if tool_calls:
            for idx, call in enumerate(tool_calls[:8], start=1):
                fn = call.get("function", "") or call.get("name", "")
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

    # ------------------------------------------------------------------
    # Real task agent execution
    # ------------------------------------------------------------------

    def _build_task_agent_prompt(self, turn_question: str, icl_text: str) -> str:
        """Build the user prompt for the real task agent."""
        parts = []

        # Include real conversation history for context
        if self._real_history_items:
            parts.append("=== Previous Conversation History ===")
            for item in self._real_history_items:
                role = item.get("role", "")
                if role == "user":
                    parts.append(f"User: {item.get('content', '')}")
                elif role == "assistant":
                    parts.append(f"Assistant: {item.get('content', '')}")
                elif role == "tool_call":
                    fn = item.get("function") or item.get("name", "unknown")
                    args = item.get("arguments") or item.get("args", {})
                    args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                    parts.append(f"Tool Call: {fn}")
                    parts.append(f"Arguments: {args_str}")
                    if "result" in item:
                        result = item["result"]
                        result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                        parts.append(f"Result: {result_str}")
            parts.append("")

        parts.append(f"Current task:\n{turn_question}")

        if icl_text:
            parts.append("")
            parts.append(icl_text)

        return "\n".join(parts)

    def _execute_real_turn(
        self, turn_question: str, icl_text: str
    ) -> Dict[str, Any]:
        """Run the real task agent for one turn.

        Returns dict with 'tool_calls' and 'assistant_text'.
        """
        prompt = self._build_task_agent_prompt(turn_question, icl_text)

        response = self._real_task_agent.generate_response(
            prompt, context={"temperature": 0.1}
        )

        tool_calls = []
        assistant_text = ""
        if response:
            tool_calls = response.tool_calls or []
            assistant_text = response.raw_response or ""

        return {
            "tool_calls": tool_calls,
            "assistant_text": assistant_text,
        }

    def _append_real_history(
        self,
        user_message: str,
        tool_calls: List[Dict[str, Any]],
        assistant_text: str,
    ) -> None:
        """Append one turn's results to real conversation history."""
        self._real_history_items.append({"role": "user", "content": user_message})
        for tc in tool_calls:
            fn = tc.get("function") or tc.get("name", "")
            args = tc.get("arguments", {})
            result = tc.get("result")
            entry: Dict[str, Any] = {
                "role": "tool_call",
                "function": fn,
                "arguments": args,
            }
            if result is not None:
                entry["result"] = result
            self._real_history_items.append(entry)
        if assistant_text:
            self._real_history_items.append({"role": "assistant", "content": assistant_text})

    # ------------------------------------------------------------------
    # Utility methods (ported from multi_turn_executor)
    # ------------------------------------------------------------------

    @staticmethod
    def _compact_args(arguments: Any) -> str:
        args = arguments
        if isinstance(arguments, dict) and "requestBody" in arguments:
            rb = arguments["requestBody"]
            if isinstance(rb, dict):
                args = rb
        if not isinstance(args, dict):
            return ""
        items = []
        for idx, (k, v) in enumerate(args.items()):
            if idx >= 4:
                items.append("...")
                break
            if isinstance(v, str) and len(v) > 60:
                v = v[:57] + "..."
            items.append(f"{k}={repr(v)}")
        return ", ".join(items)

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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_turn(self, question: str) -> GATSTurn:
        if self._enable_real and self._real_history_items:
            self._solver.set_history(self._real_history_items)
            self._solver.set_initial_config(self._real_current_config)

        # Stage 1: SimSolver (mock tools) → best attempt
        outcome: TurnOutcome = self._solver.process(question)

        if not self._enable_real:
            # Single-turn or no real execution → return mock result directly
            return _outcome_to_gats_turn(outcome, question)

        # Stage 2: Extract best attempt → build ICL plan
        # If all attempts failed (score == 0), skip ICL to avoid misleading
        # the real agent with incorrect instructions — degrade to vanilla.
        best_attempt = self._extract_best_attempt(outcome)
        if outcome.score >= 1.0:
            icl_text = self._build_best_attempt_icl(best_attempt)
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

        # Stage 3: Real task agent follows the ICL plan with real tools
        real_result = self._execute_real_turn(question, icl_text)
        real_tool_calls = real_result.get("tool_calls", [])
        assistant_text = real_result.get("assistant_text", "")

        # Fallback: if real agent made no calls but SimSolver did, use SimSolver's
        mock_calls = best_attempt.get("tool_calls", [])
        if not real_tool_calls and mock_calls:
            logger.warning(
                "Real agent made no tool calls for %s turn %d; "
                "falling back to SimSolver best attempt calls",
                self._task.id, outcome.turn_idx,
            )
            real_tool_calls = mock_calls

        # Update real conversation history
        self._append_real_history(question, real_tool_calls, assistant_text)

        # Stage 4: Sync real tool results → persistent Gecko session → updated state
        # One session is reused across all turns so the state-model has full context.
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

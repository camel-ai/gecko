#!/usr/bin/env python3
"""BFCL multi-turn executor with SimSolver-guided real execution.

Flow per turn:
1) SimSolver runs on mock environment and selects best attempt.
2) Build ICL from the best attempt only.
3) Task agent executes in real environment with wrapped real tools.
4) Real wrapper syncs each tool result to Gecko.
5) Next-turn state uses real synced session state.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import hashlib
import json
import logging
import os
import tempfile
import sys
import time
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from benchmarks.bfcl.data.loader import TestCase
from inference.core import SimSolver, TurnOutcome
from inference.real_tools import SessionContext, wrap_real_tool
from utils.test_case_adapter import TestCaseAdapter

from .agents.chat_agent import ChatAgent
from .client_engine import InferenceConfig, InferenceResult, MockServerClient, TurnResult

logger = logging.getLogger(__name__)


TOOL_DOMAIN_CONFIG: Dict[str, Dict[str, Any]] = {
    "GorillaFileSystem": {
        "module": "benchmarks.bfcl.multi_turn.func_source_code.gorilla_file_system",
        "class": "GorillaFileSystem",
        "schema": "GorillaFileSystem",
        "prefixes": ["gorillafilesystem"],
        "aliases": ["GorillaFileSystem"],
    },
    "TwitterAPI": {
        "module": "benchmarks.bfcl.multi_turn.func_source_code.posting_api",
        "class": "TwitterAPI",
        "schema": "TwitterAPI",
        "prefixes": ["twitterapi", "postingapi"],
        "aliases": ["TwitterAPI", "PostingAPI"],
    },
    "MathAPI": {
        "module": "benchmarks.bfcl.multi_turn.func_source_code.math_api",
        "class": "MathAPI",
        "schema": "MathAPI",
        "prefixes": ["mathapi"],
        "aliases": ["MathAPI"],
    },
    "MessageAPI": {
        "module": "benchmarks.bfcl.multi_turn.func_source_code.message_api",
        "class": "MessageAPI",
        "schema": "MessageAPI",
        "prefixes": ["messageapi"],
        "aliases": ["MessageAPI"],
    },
    "TicketAPI": {
        "module": "benchmarks.bfcl.multi_turn.func_source_code.ticket_api",
        "class": "TicketAPI",
        "schema": "TicketAPI",
        "prefixes": ["ticketapi"],
        "aliases": ["TicketAPI"],
    },
    "TradingBot": {
        "module": "benchmarks.bfcl.multi_turn.func_source_code.trading_bot",
        "class": "TradingBot",
        "schema": "TradingBot",
        "prefixes": ["tradingbot"],
        "aliases": ["TradingBot"],
    },
    "TravelAPI": {
        "module": "benchmarks.bfcl.multi_turn.func_source_code.travel_booking",
        "class": "TravelAPI",
        "schema": "TravelAPI",
        "prefixes": ["travelapi", "travelbooking"],
        "aliases": ["TravelAPI", "BookingAPI", "TravelBooking"],
    },
    "VehicleControlAPI": {
        "module": "benchmarks.bfcl.multi_turn.func_source_code.vehicle_control",
        "class": "VehicleControlAPI",
        "schema": "VehicleControlAPI",
        "prefixes": ["vehiclecontrolapi", "vehiclecontrol"],
        "aliases": ["VehicleControlAPI", "VehicleControl"],
    },
}

ALIAS_TO_CANONICAL: Dict[str, str] = {}
for _canonical, _cfg in TOOL_DOMAIN_CONFIG.items():
    for _alias in _cfg.get("aliases", []):
        ALIAS_TO_CANONICAL[_alias] = _canonical

TOOL_NAME_PREFIXES: List[str] = [
    "gorillafilesystem_",
    "postingapi_",
    "twitterapi_",
    "messageapi_",
    "ticketapi_",
    "mathapi_",
    "tradingbot_",
    "travelapi_",
    "travelbooking_",
    "vehiclecontrolapi_",
    "vehiclecontrol_",
]


class MultiTurnExecutor:
    """Multi-turn executor using SimSolver for planning and task agent for real calls."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.mock_client = MockServerClient(config.mock_server_url)
        try:
            self.mock_client.timeout = float(config.timeout)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid config.timeout=%s for mock client timeout; keep default %.1fs",
                config.timeout,
                self.mock_client.timeout,
            )
        self.test_case: Optional[TestCase] = None

        self.real_tool_instances: Dict[str, Any] = {}
        self.real_tools: List[Any] = []
        self._task_agent: Optional[ChatAgent] = None

        self._sim_solver: Optional[SimSolver] = None
        self._real_history_items: List[Dict[str, Any]] = []
        self._completed_tasks: List[str] = []
        self._trace_compact = self._resolve_trace_compact()
        self._real_session_id: str = ""
        self._real_session_seeded: bool = False
        self._real_sync_context: Optional[SessionContext] = None
        self._excluded_operations: set[str] = set()
        self._active_openapi_base_dir: Path = Path("data/bfcl_v4/openapi/multi_turn")
        self._real_tool_by_name: Dict[str, Any] = {}
        self._real_tool_ops_index: Dict[str, List[str]] = {}
        self._real_tool_argument_schemas: Dict[str, Dict[str, Any]] = {}

    def _trace_print(self, message: str) -> None:
        if bool(getattr(self.config, "enable_debug", False)):
            print(message)

    def execute_multi_turn(self, test_case: TestCase) -> InferenceResult:
        """Execute full BFCL multi-turn flow."""
        self.test_case = test_case
        self._real_session_id = ""
        self._real_session_seeded = False
        self._real_sync_context = None
        self._task_agent = None
        self._excluded_operations = set()
        self._active_openapi_base_dir = Path("data/bfcl_v4/openapi/multi_turn")
        self._real_tool_by_name = {}
        self._real_tool_ops_index = {}

        if hasattr(test_case, "content") and test_case.content:
            bfcl_test_case = test_case.content
        else:
            bfcl_test_case = test_case

        test_id = TestCaseAdapter.get_id(test_case)
        start_time = time.time()

        initial_config = self._extract_initial_config(test_case, bfcl_test_case)
        current_config = copy.deepcopy(initial_config)
        config_history = [copy.deepcopy(current_config)]

        self._excluded_operations = self._extract_excluded_operations(test_case, bfcl_test_case)
        openapi_paths, self._active_openapi_base_dir = self._get_openapi_paths_for_test(
            test_case,
            excluded_operations=self._excluded_operations,
            test_id=test_id,
        )
        base_checklist_items = None
        checklist_system_prompt = getattr(self.config, "checklist_system_prompt", None)
        if isinstance(getattr(self.config, "custom_config", None), dict):
            base_checklist_items = self.config.custom_config.get("base_checklist_items")
            if checklist_system_prompt is None:
                checklist_system_prompt = self.config.custom_config.get("checklist_system_prompt")

        self._sim_solver = SimSolver(
            test_case=test_case,
            model_name=self.config.model_name,
            max_retries=self.config.max_retries,
            agent_timeout=getattr(self.config, "timeout", 60),
            mock_server_url=self.config.mock_server_url,
            override_openapi_server=True,
            agent_max_iteration=getattr(self.config, "agent_max_iteration", None),
            agent_summarize_threshold=getattr(self.config, "agent_summarize_threshold", None),
            enable_evaluation=self.config.max_retries > 0,
            enable_checklist=getattr(self.config, "enable_checklist", True),
            openapi_tool_paths=openapi_paths,
            agent_persistence_mode=False,
            agent_system_prompt=getattr(self.config, "agent_system_prompt", None),
            judge_system_prompt=getattr(self.config, "judge_system_prompt", None),
            base_checklist_items=base_checklist_items,
            checklist_system_prompt=checklist_system_prompt,
            include_agent_response_in_judge=False,
            enable_tool_result_folding=getattr(self.config, "enable_tool_result_folding", True),
        )

        self._real_session_id, self._real_session_seeded = self._init_real_execution_session(
            initial_state=current_config
        )
        self._real_sync_context = SessionContext(
            session_id=self._real_session_id,
            mock_server_url=self.config.mock_server_url,
            task_id=f"{test_id}:real",
            buffer_updates=True,
        )
        self.real_tool_instances = self._initialize_real_tool_instances(test_case, current_config)
        self.real_tools = self._build_wrapped_real_tools(
            test_case,
            expected_session_id=self._real_session_id,
            session_context=self._real_sync_context,
            openapi_base_dir=self._active_openapi_base_dir,
        )
        logger.info("Initialized %d wrapped real tools", len(self.real_tools))
        self._task_agent = self._create_task_agent()
        self._task_agent.set_tools(self.real_tools)

        questions = TestCaseAdapter.get_questions(bfcl_test_case)
        total_turns = len(questions)
        max_turns = self._resolve_max_turn_limit()
        executed_questions = questions[:max_turns] if max_turns is not None else questions
        executed_turns = len(executed_questions)

        trace_started_at = time.time()
        process_trace: Dict[str, Any] = {
            "header": {
                "test_id": test_id,
                "model": self.config.model_name,
                "task_evaluator_model": getattr(self.config, "task_evaluator_model_name", None),
                "max_retries": self.config.max_retries,
                "enable_real_execution": True,
                "max_turns_requested": max_turns,
                "total_turns": total_turns,
                "executed_turns": executed_turns,
                "trace_compact": self._trace_compact,
                "excluded_operations": sorted(list(self._excluded_operations)),
                "started_at": datetime.fromtimestamp(trace_started_at).isoformat(),
            },
            "events": [],
        }

        self._append_trace_event(
            process_trace,
            trace_started_at,
            event="run_config",
            label="config",
            data=self._trace_payload("run_config", copy.deepcopy(process_trace["header"])),
        )
        self._append_trace_event(
            process_trace,
            trace_started_at,
            event="initial_config",
            label="initial_config",
            data=self._trace_payload("initial_config", copy.deepcopy(initial_config)),
        )
        if max_turns is not None and max_turns < total_turns:
            self._append_trace_event(
                process_trace,
                trace_started_at,
                event="turn_limit",
                label="turn_limit",
                data=self._trace_payload("turn_limit", {
                    "max_turns_requested": max_turns,
                    "total_turns": total_turns,
                    "executed_turns": executed_turns,
                }),
            )

        turns: List[TurnResult] = []

        logger.info(
            "Starting BFCL multi-turn run: %s (%d/%d turns)",
            test_id,
            executed_turns,
            total_turns,
        )

        for turn_idx, question_turn in enumerate(executed_questions):
            turn_started_at = time.time()
            turn_question = self._extract_turn_text(question_turn)

            self._append_trace_event(
                process_trace,
                trace_started_at,
                event="task",
                label=f"task{turn_idx}",
                data=self._trace_payload("task", {"turn_index": turn_idx, "task": turn_question}),
            )

            if self._sim_solver is None:
                raise RuntimeError("SimSolver not initialized")

            # Ensure SimSolver sees the latest real state + history.
            self._sim_solver.set_initial_config(current_config)
            self._sim_solver.set_history(self._real_history_items)
            sim_outcome = self._sim_solver.process(turn_question)

            best_attempt, checklist, all_attempts = self._extract_attempt_data(sim_outcome)
            icl_text = self._build_best_attempt_icl(best_attempt)

            for attempt in all_attempts:
                attempt_idx = int(attempt.get("attempt", 0) or 0)
                attempt_tool_calls = self._normalize_tool_calls(attempt.get("tool_calls", []))
                attempt_judge = copy.deepcopy(attempt.get("judgment", {}))
                attempt_judge.setdefault("checklist", copy.deepcopy(checklist))
                self._append_trace_event(
                    process_trace,
                    trace_started_at,
                    event="attempt",
                    label=f"turn{turn_idx}.attempt{attempt_idx}",
                    data=self._trace_payload("attempt", {
                        "turn_index": turn_idx,
                        "attempt_index": attempt_idx,
                        "tool_calls": attempt_tool_calls,
                        "judge": attempt_judge,
                        "mock_config": copy.deepcopy(attempt.get("mock_config", {})),
                        "agent_response": attempt.get("agent_response", ""),
                    }),
                )

            real_turn = self._execute_task_agent_turn(
                test_id=test_id,
                turn_idx=turn_idx,
                turn_question=turn_question,
                current_config=current_config,
                icl_text=icl_text,
                session_id=self._real_session_id,
                session_seeded=self._real_session_seeded,
            )

            raw_actual_tool_calls = self._normalize_tool_calls(real_turn.get("tool_calls", []))
            fallback_used = False
            if not raw_actual_tool_calls and self._normalize_tool_calls(best_attempt.get("tool_calls", [])):
                fallback_turn = self._execute_best_attempt_fallback_real_calls(
                    test_id=test_id,
                    turn_idx=turn_idx,
                    best_attempt=best_attempt,
                    session_id=self._real_session_id,
                    current_config=current_config,
                )
                raw_actual_tool_calls = self._normalize_tool_calls(fallback_turn.get("tool_calls", []))
                if raw_actual_tool_calls:
                    fallback_used = True
                    if isinstance(fallback_turn.get("synced_state"), dict) and fallback_turn.get("synced_state"):
                        real_turn["synced_state"] = fallback_turn["synced_state"]
                    if isinstance(fallback_turn.get("session_state"), dict) and fallback_turn.get("session_state"):
                        real_turn["session_state"] = fallback_turn["session_state"]
                    if not real_turn.get("assistant_text"):
                        real_turn["assistant_text"] = fallback_turn.get("assistant_text", "")
                    self._append_trace_event(
                        process_trace,
                        trace_started_at,
                        event="real_execution_fallback",
                        label=f"turn{turn_idx}.fallback_best_attempt",
                        data=self._trace_payload(
                            "real_execution",
                            {
                                "turn_index": turn_idx,
                                "reason": "task_agent_empty_tool_calls",
                                "tool_calls": copy.deepcopy(raw_actual_tool_calls),
                            },
                        ),
                    )
            actual_tool_calls = self._normalize_real_tool_call_arguments(raw_actual_tool_calls)
            synced_state = real_turn.get("synced_state")
            task_llm_cost = float(real_turn.get("task_agent_cost", 0.0) or 0.0)
            assistant_text = real_turn.get("assistant_text", "")

            self._append_trace_event(
                process_trace,
                trace_started_at,
                event="real_execution",
                label=f"turn{turn_idx}.real_execution",
                data=self._trace_payload("real_execution", {
                    "turn_index": turn_idx,
                    "tool_calls": copy.deepcopy(actual_tool_calls),
                    "raw_tool_calls": copy.deepcopy(raw_actual_tool_calls),
                    "fallback_used": fallback_used,
                    "session_id": real_turn.get("session_id", ""),
                    "session_state": copy.deepcopy(real_turn.get("session_state", {})),
                    "synced_state": copy.deepcopy(synced_state) if isinstance(synced_state, dict) else synced_state,
                }),
            )
            self._append_trace_event(
                process_trace,
                trace_started_at,
                event="agent_response",
                label=f"turn{turn_idx}.agent_response",
                data=self._trace_payload("agent_response", {
                    "turn_index": turn_idx,
                    "response": assistant_text,
                }),
            )

            self._append_real_history(
                user_message=turn_question,
                tool_calls=actual_tool_calls,
                assistant_text=assistant_text,
            )

            # Real-first success rule:
            # - normal turns require real tool calls
            # - allow no-call success only for no-action turns (empty checklist)
            turn_success = bool(actual_tool_calls) or (sim_outcome.score >= 1.0 and not checklist)
            turn_result = TurnResult(
                turn_idx=turn_idx,
                success=turn_success,
                mock_tool_calls=actual_tool_calls,
                real_tool_calls=actual_tool_calls,
                execution_time=time.time() - turn_started_at,
                judge_score=float(sim_outcome.score),
                mock_config=copy.deepcopy(sim_outcome.final_config),
                calibrated_config=copy.deepcopy(synced_state) if isinstance(synced_state, dict) else None,
                llm_costs={"task_agent": task_llm_cost},
                checklist=checklist,
                all_attempts=all_attempts,
                task_agent_response=assistant_text,
            )
            turns.append(turn_result)

            formatted_calls = self._format_tool_calls_for_output(actual_tool_calls)
            self._trace_print(f"{test_id}[{turn_idx + 1}/{executed_turns}]: {formatted_calls}")

            if isinstance(synced_state, dict) and synced_state:
                current_config = copy.deepcopy(synced_state)
            elif isinstance(real_turn.get("session_state"), dict) and real_turn.get("session_state"):
                current_config = copy.deepcopy(real_turn["session_state"])
            else:
                logger.warning("Turn %d did not return synced state; keeping previous config", turn_idx)

            config_history.append(copy.deepcopy(current_config))
            self._append_trace_event(
                process_trace,
                trace_started_at,
                event="task_state",
                label=f"task_state_{turn_idx + 1}",
                data=self._trace_payload("task_state", {
                    "turn_index": turn_idx,
                    "state": copy.deepcopy(current_config),
                }),
            )
            self._completed_tasks.append(turn_question)

        overall_success = all(turn.success for turn in turns)
        final_score = sum(turn.judge_score for turn in turns) / len(turns) if turns else 0.0

        total_llm_costs: Dict[str, float] = {}
        for turn in turns:
            for name, val in (turn.llm_costs or {}).items():
                total_llm_costs[name] = total_llm_costs.get(name, 0.0) + float(val or 0.0)

        finished_at = time.time()
        process_trace["header"]["finished_at"] = datetime.fromtimestamp(finished_at).isoformat()
        process_trace["header"]["total_time"] = round(finished_at - start_time, 6)
        self._append_trace_event(
            process_trace,
            trace_started_at,
            event="run_end",
            label="run_end",
            data=self._trace_payload("run_end", {
                "success": overall_success,
                "final_score": final_score,
                "executed_turns": executed_turns,
                "total_turns": total_turns,
            }),
        )

        return InferenceResult(
            test_id=test_id,
            success=overall_success,
            turns=turns,
            total_time=time.time() - start_time,
            judge_score=final_score,
            config_history=config_history,
            total_llm_costs=total_llm_costs,
            total_turns=total_turns,
            executed_turns=executed_turns,
            max_turns_applied=max_turns,
            process_trace=process_trace,
        )

    def _resolve_max_turn_limit(self) -> Optional[int]:
        """Resolve max-turn limit from config."""
        max_turns = getattr(self.config, "max_turns", None)
        if max_turns is None and hasattr(self.config, "custom_config"):
            custom = getattr(self.config, "custom_config", {}) or {}
            max_turns = custom.get("max_turns")

        if max_turns is None:
            return None

        try:
            max_turns_int = int(max_turns)
        except (TypeError, ValueError):
            logger.warning("Invalid max_turns=%r, ignoring turn limit", max_turns)
            return None

        if max_turns_int <= 0:
            logger.warning("Non-positive max_turns=%r, ignoring turn limit", max_turns)
            return None
        return max_turns_int

    def _resolve_trace_compact(self) -> bool:
        """Resolve trace compact mode from config."""
        value = getattr(self.config, "trace_compact", None)
        if value is None and hasattr(self.config, "custom_config"):
            custom = getattr(self.config, "custom_config", {}) or {}
            value = custom.get("trace_compact", False)
        return bool(value)

    def _trace_payload(self, event: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optionally compact heavy state fields in process trace."""
        if not self._trace_compact:
            return data

        compact = copy.deepcopy(data)
        if event == "initial_config":
            return {"state_summary": self._summarize_state(data)}

        if event == "attempt" and isinstance(compact.get("mock_config"), dict):
            compact["mock_config_summary"] = self._summarize_state(compact["mock_config"])
            compact.pop("mock_config", None)

        if event == "real_execution":
            if isinstance(compact.get("session_state"), dict):
                compact["session_state_summary"] = self._summarize_state(compact["session_state"])
                compact.pop("session_state", None)
            if isinstance(compact.get("synced_state"), dict):
                compact["synced_state_summary"] = self._summarize_state(compact["synced_state"])
                compact.pop("synced_state", None)

        if event == "task_state" and isinstance(compact.get("state"), dict):
            compact["state_summary"] = self._summarize_state(compact["state"])
            compact.pop("state", None)

        return compact

    @staticmethod
    def _summarize_state(state: Dict[str, Any]) -> Dict[str, Any]:
        """Create deterministic summary for large state dicts."""
        try:
            raw = json.dumps(state, sort_keys=True, ensure_ascii=False, default=str)
            digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            return {
                "sha256": digest,
                "bytes": len(raw.encode("utf-8")),
                "top_keys": sorted(list(state.keys()))[:20],
            }
        except Exception:
            return {
                "sha256": "",
                "bytes": 0,
                "top_keys": sorted(list(state.keys()))[:20],
            }

    @staticmethod
    def _append_trace_event(
        process_trace: Dict[str, Any],
        trace_started_at: float,
        *,
        event: str,
        label: str,
        data: Dict[str, Any],
    ) -> None:
        ts = time.time()
        process_trace.setdefault("events", []).append(
            {
                "timestamp": ts,
                "timestamp_iso": datetime.fromtimestamp(ts).isoformat(),
                "elapsed_seconds": round(ts - trace_started_at, 6),
                "event": event,
                "label": label,
                "data": data,
            }
        )

    def _execute_task_agent_turn(
        self,
        *,
        test_id: str,
        turn_idx: int,
        turn_question: str,
        current_config: Dict[str, Any],
        icl_text: str,
        session_id: str,
        session_seeded: bool,
    ) -> Dict[str, Any]:
        """Run the real task agent for one turn and return executed calls + synced state."""
        if not self.real_tools:
            raise RuntimeError("Real tools are not initialized")

        request_timeout = float(getattr(self.config, "timeout", 60) or 60)
        if not session_id:
            raise RuntimeError("Real execution session is not initialized")

        prompt = self._build_task_agent_prompt(turn_question, icl_text)

        agent = self._task_agent
        if agent is None:
            agent = self._create_task_agent()
            agent.set_tools(self.real_tools)
            self._task_agent = agent

        context = self._real_sync_context
        if context is None:
            raise RuntimeError("Real sync context is not initialized")
        if context.session_id != session_id:
            raise RuntimeError(
                f"Real sync context session mismatch: expected={session_id}, actual={context.session_id}"
            )
        context.task_id = f"{test_id}:turn{turn_idx}"
        response = agent.generate_response(prompt, context={"temperature": 0.1})

        sync_flush_error: Optional[str] = None
        flushed_state: Optional[Dict[str, Any]] = None
        try:
            flushed_state = context.maybe_flush_buffer()
        except Exception as e:
            if "UNTESTED_TASK:" in str(e):
                raise
            sync_flush_error = str(e)
            logger.warning(
                "Buffered real sync flush failed (test=%s turn=%d session=%s): %s",
                test_id,
                turn_idx,
                session_id,
                e,
            )

        tool_calls = self._normalize_tool_calls(response.tool_calls if response else [])
        if not tool_calls:
            tool_calls = self._extract_tool_calls_from_real_sync_history(session_id)

        if not tool_calls:
            session_state = copy.deepcopy(current_config)
            synced_state: Dict[str, Any] = copy.deepcopy(current_config)
        elif isinstance(flushed_state, dict) and flushed_state:
            session_state = copy.deepcopy(flushed_state)
            synced_state = copy.deepcopy(flushed_state)
        else:
            session_state = self.mock_client.get_session_state(
                session_id,
                timeout=request_timeout,
                retries=0,
                backoff_sec=0.5,
            )
            synced_state = session_state if isinstance(session_state, dict) else {}
            if sync_flush_error is not None and not synced_state:
                logger.warning(
                    "Sync flush failed and no session state available (test=%s turn=%d session=%s)",
                    test_id,
                    turn_idx,
                    session_id,
                )

        if not session_seeded and tool_calls:
            logger.warning(
                "Real session was not seeded successfully; synced state may be incomplete "
                "(test=%s turn=%d session=%s)",
                test_id,
                turn_idx,
                session_id,
            )

        return {
            "tool_calls": tool_calls,
            "synced_state": synced_state,
            "session_state": session_state,
            "session_id": session_id,
            "assistant_text": response.raw_response if response else "",
            "task_agent_cost": (response.metadata or {}).get("llm_cost", 0.0) if response else 0.0,
        }

    def _create_task_agent(self) -> ChatAgent:
        """Create one task agent instance to be reused across turns of the same test."""
        return ChatAgent(
            model_name=self.config.model_name,
            timeout=getattr(self.config, "timeout", 60),
            system_message=getattr(self.config, "agent_system_prompt", None),
            max_iteration=getattr(self.config, "agent_max_iteration", 10),
            summarize_threshold=getattr(self.config, "agent_summarize_threshold", None),
            agent_role="task_agent",
        )

    def _execute_best_attempt_fallback_real_calls(
        self,
        *,
        test_id: str,
        turn_idx: int,
        best_attempt: Dict[str, Any],
        session_id: str,
        current_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute best-attempt calls on wrapped real tools when task agent emits no calls."""
        context = self._real_sync_context
        if context is None:
            return {
                "tool_calls": [],
                "synced_state": copy.deepcopy(current_config),
                "session_state": copy.deepcopy(current_config),
                "session_id": session_id,
                "assistant_text": "",
            }
        if context.session_id != session_id:
            raise RuntimeError(
                f"Real sync context session mismatch during fallback: expected={session_id}, actual={context.session_id}"
            )

        context.task_id = f"{test_id}:turn{turn_idx}:fallback"
        attempt_calls = self._normalize_tool_calls(best_attempt.get("tool_calls", []))
        executed: List[Dict[str, Any]] = []

        for call in attempt_calls:
            op = self._canonical_tool_name(str(call.get("function", "") or ""))
            tool_name = self._resolve_real_tool_name_for_operation(op)
            if not tool_name:
                continue
            tool = self._real_tool_by_name.get(tool_name)
            if tool is None:
                continue

            args = self._unwrap_request_body(call.get("arguments", {}))
            if not isinstance(args, dict):
                args = {}

            try:
                result = tool(**args)
            except Exception as e:
                result = {"error": str(e)}

            executed.append({"function": tool_name, "arguments": args, "result": result})

        synced_state: Dict[str, Any] = copy.deepcopy(current_config)
        session_state: Dict[str, Any] = copy.deepcopy(current_config)
        if executed:
            try:
                flushed_state = context.maybe_flush_buffer()
                if isinstance(flushed_state, dict) and flushed_state:
                    synced_state = copy.deepcopy(flushed_state)
                    session_state = copy.deepcopy(flushed_state)
                else:
                    request_timeout = float(getattr(self.config, "timeout", 60) or 60)
                    fetched_state = self.mock_client.get_session_state(
                        session_id,
                        timeout=request_timeout,
                        retries=0,
                        backoff_sec=0.5,
                    )
                    if isinstance(fetched_state, dict) and fetched_state:
                        synced_state = copy.deepcopy(fetched_state)
                        session_state = copy.deepcopy(fetched_state)
            except Exception as e:
                if "UNTESTED_TASK:" in str(e):
                    raise
                logger.warning(
                    "Fallback sync failed (test=%s turn=%d session=%s): %s",
                    test_id,
                    turn_idx,
                    session_id,
                    e,
                )

        return {
            "tool_calls": executed,
            "synced_state": synced_state,
            "session_state": session_state,
            "session_id": session_id,
            "assistant_text": "",
        }

    def _init_real_execution_session(self, initial_state: Dict[str, Any]) -> Tuple[str, bool]:
        """Create and seed one Gecko session for the full multi-turn real execution."""
        request_timeout = float(getattr(self.config, "timeout", 60) or 60)
        session_id = self.mock_client.create_session(
            self.test_case,
            timeout=request_timeout,
            retries=0,
            backoff_sec=0.5,
        )
        seeded = self.mock_client.set_session_state(
            session_id,
            copy.deepcopy(initial_state),
            bootstrap_mode="auto",
            timeout=request_timeout,
            retries=0,
            backoff_sec=0.5,
        )
        if not seeded:
            logger.warning(
                "Failed to seed real execution session (session=%s, timeout=%.1fs)",
                session_id,
                request_timeout,
            )
        else:
            logger.info(
                "Initialized shared real execution session: %s",
                session_id,
            )
        return session_id, seeded

    def _build_task_agent_prompt(self, turn_question: str, icl_text: str) -> str:
        """Build the user prompt for the real task agent."""
        icl_section = f"""{icl_text}
""" if icl_text else ""

        prompt = f"""Current task:
{turn_question}

{icl_section}"""
        return "\n\n".join(section for section in prompt.split("\n\n") if section.strip())

    def _build_best_attempt_icl(self, best_attempt: Dict[str, Any]) -> str:
        """Build ICL text from SimSolver best attempt only."""
        if not best_attempt:
            return ""

        score = float(best_attempt.get("score", 0.0) or 0.0)
        success = score >= 1.0
        tool_calls = best_attempt.get("tool_calls", []) or []
        feedback = best_attempt.get("feedback", {}) if isinstance(best_attempt.get("feedback", {}), dict) else {}
        if success:
            forbidden_calls = ""
            successful_calls = ""
            forbidden_count = 0
            successful_count = 0

            for call in tool_calls[:8]:
                if not isinstance(call, dict):
                    continue
                fn = str(call.get("function", "") or "")
                args = call.get("arguments", {})
                signature = f"{fn}({self._compact_args(args)})"
                result = call.get("result")
                err = self._extract_error_text(result)
                if err:
                    forbidden_count += 1
                    forbidden_calls += f"{forbidden_count}. {signature}\n"
                else:
                    successful_count += 1
                    successful_calls += f"{successful_count}. {signature}\n"

            successful_section = (
                successful_calls.rstrip() if successful_calls else "<none>"
            )

            icl_text = f"""Executable plan:
{successful_section}

Guidance:
1) The executable plan could solve the given task. Follow it strictly.
2) Update arguments in the plan using real tool results from your tool calls, since the plan may contain outdated values. But do not change the tool call sequence or add new calls that are not in the plan.
3) Keep each string argument exactly as written in the tool result or the plan. Do not add or remove any quotes, escapes, whitespace, and line breaks.
"""
            return "\n\n".join(section for section in icl_text.split("\n\n") if section.strip())

        tool_calls_block = "- tool calls:\n" if tool_calls else "- tool calls: none"
        if tool_calls:
            for idx, call in enumerate(tool_calls[:8], start=1):
                fn = call.get("function", "")
                args = call.get("arguments", {})
                result = call.get("result")
                call_status = "invalid" if self._extract_error_text(result) else "valid"
                tool_calls_block += (
                    f"  {idx}. [{call_status}] {fn}({self._compact_args(args)})\n"
                    f"     result: {self._compact_result(result)}\n"
                )
            tool_calls_block = tool_calls_block.rstrip()

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
        return "\n\n".join(section for section in icl_text.split("\n\n") if section.strip())

    def _extract_unresolved_issues(
        self,
        feedback: Dict[str, Any],
        tool_calls: List[Dict[str, Any]],
    ) -> List[str]:
        issues: List[str] = []

        failed_items = feedback.get("failed_items", []) if isinstance(feedback, dict) else []
        for item in failed_items:
            if isinstance(item, dict):
                desc = item.get("description")
                if desc:
                    issues.append(str(desc))

        critical = feedback.get("critical_responses", []) if isinstance(feedback, dict) else []
        for msg in critical:
            if msg:
                issues.append(str(msg))

        for call in tool_calls:
            result = call.get("result")
            err = self._extract_error_text(result)
            if err:
                issues.append(err)

        # preserve order while deduping
        seen = set()
        deduped: List[str] = []
        for issue in issues:
            if issue not in seen:
                seen.add(issue)
                deduped.append(issue)

        return deduped

    @staticmethod
    def _extract_error_text(result: Any) -> Optional[str]:
        if isinstance(result, dict):
            if "error" in result and result["error"]:
                return str(result["error"])
            detail = result.get("detail")
            if isinstance(detail, dict) and detail.get("error_message"):
                return str(detail.get("error_message"))
            if isinstance(detail, str) and detail:
                return detail
            if result.get("success") is False:
                return str(result.get("message") or "Tool returned success=false")
        if isinstance(result, str) and "error" in result.lower():
            return result
        return None

    @staticmethod
    def _compact_args(arguments: Any) -> str:
        args = arguments
        if isinstance(arguments, dict) and "requestBody" in arguments and isinstance(arguments["requestBody"], dict):
            args = arguments["requestBody"]
        if not isinstance(args, dict):
            return ""

        items: List[str] = []
        for idx, (k, v) in enumerate(args.items()):
            if idx >= 4:
                items.append("...")
                break
            if isinstance(v, str):
                val = v.replace("\n", "\\n")
                items.append(f"{k}='{val}'")
            else:
                items.append(f"{k}={v}")
        return ", ".join(items)

    @staticmethod
    def _compact_result(result: Any, max_len: int = 320) -> str:
        """Compact tool result for ICL prompt readability."""
        try:
            if isinstance(result, str):
                text = result
            else:
                text = json.dumps(result, ensure_ascii=False, default=str)
        except Exception:
            text = str(result)

        text = text.replace("\n", "\\n")
        if len(text) > max_len:
            return text[: max_len - 3] + "..."
        return text

    def _filter_error_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split tool calls into non-error calls and filtered error calls."""
        kept: List[Dict[str, Any]] = []
        filtered: List[Dict[str, Any]] = []

        for call in tool_calls or []:
            if not isinstance(call, dict):
                continue
            err = self._extract_error_text(call.get("result"))
            if err:
                filtered.append(
                    {
                        "function": call.get("function", ""),
                        "arguments": call.get("arguments", {}) if isinstance(call.get("arguments"), dict) else {},
                        "error": err,
                    }
                )
                continue
            kept.append(call)

        return kept, filtered

    def _extract_attempt_data(self, outcome: TurnOutcome) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract best attempt/checklist/all attempts from SimSolver events."""
        attempts: Dict[int, Dict[str, Any]] = {}
        checklist: List[Dict[str, Any]] = []

        for ev in outcome.events:
            if ev.type == "checklist":
                raw = ev.data.get("checklist", []) if isinstance(ev.data, dict) else []
                checklist = raw if isinstance(raw, list) else []
                continue

            if ev.attempt is None:
                continue

            idx = int(ev.attempt)
            rec = attempts.setdefault(
                idx,
                {
                    "attempt": idx,
                    "score": 0.0,
                    "feedback": {},
                    "tool_calls": [],
                    "mock_config": {},
                    "agent_response": "",
                    "tools_count": 0,
                    "judgment": {"score": 0.0, "passed": False, "feedback": {}},
                },
            )

            if ev.type == "tool_calls":
                rec["tool_calls"] = ev.data.get("tool_calls", []) if isinstance(ev.data, dict) else []
                rec["tools_count"] = (
                    int(ev.data.get("tools_count", len(rec["tool_calls"])))
                    if isinstance(ev.data, dict)
                    else len(rec["tool_calls"])
                )
            elif ev.type == "judge":
                rec["score"] = float(ev.data.get("score", 0.0)) if isinstance(ev.data, dict) else 0.0
                rec["feedback"] = ev.data.get("feedback", {}) if isinstance(ev.data, dict) else {}
                rec["judgment"] = {
                    "score": rec["score"],
                    "passed": rec["score"] >= 1.0,
                    "feedback": copy.deepcopy(rec["feedback"]),
                }
            elif ev.type == "attempt_config":
                state = ev.data.get("state", {}) if isinstance(ev.data, dict) else {}
                rec["mock_config"] = state if isinstance(state, dict) else {}
            elif ev.type == "agent_response":
                rec["agent_response"] = ev.data.get("response", "") if isinstance(ev.data, dict) else ""

        all_attempts = [attempts[k] for k in sorted(attempts.keys())]
        best_attempt = attempts.get(outcome.best_attempt, {})

        if best_attempt and "score" not in best_attempt:
            best_attempt["score"] = float(outcome.score)
        elif best_attempt:
            best_attempt["score"] = float(best_attempt.get("score", outcome.score) or outcome.score)
            best_attempt.setdefault(
                "judgment",
                {
                    "score": best_attempt["score"],
                    "passed": best_attempt["score"] >= 1.0,
                    "feedback": copy.deepcopy(best_attempt.get("feedback", {})),
                },
            )

        return best_attempt, checklist, all_attempts

    def _append_real_history(
        self,
        *,
        user_message: str,
        tool_calls: List[Dict[str, Any]],
        assistant_text: str,
    ) -> None:
        """Append executed real-turn events to shared history for next-turn SimSolver context."""
        self._real_history_items.append({"role": "user", "content": user_message})

        for call in tool_calls or []:
            fn = call.get("function", "")
            args = call.get("arguments", {})
            self._real_history_items.append({"role": "tool_call", "function": fn, "arguments": args})
            if "result" in call:
                self._real_history_items.append(
                    {
                        "role": "tool_result",
                        "function": fn,
                        "result": call.get("result"),
                    }
                )

        if assistant_text:
            self._real_history_items.append({"role": "assistant", "content": assistant_text})

    def _extract_tool_calls_from_real_sync_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Fallback extraction from /get-session-history when agent response has no tool_calls."""
        try:
            resp = self.mock_client.session.get(
                f"{self.config.mock_server_url.rstrip('/')}/get-session-history",
                headers={"X-Session-ID": session_id},
                timeout=60,
            )
            resp.raise_for_status()
            history = (resp.json() or {}).get("history", [])
        except Exception as e:
            logger.warning("Failed to fetch session history for %s: %s", session_id, e)
            return []

        out: List[Dict[str, Any]] = []
        for item in history:
            req = (item or {}).get("request", {})
            if req.get("path") != "/update-state-from-real":
                continue

            body = req.get("body", {})
            if not isinstance(body, dict):
                continue

            if isinstance(body.get("tool_call"), dict):
                tc = body["tool_call"]
                name = tc.get("name") or tc.get("function")
                if name:
                    out.append(
                        {
                            "function": name,
                            "arguments": tc.get("arguments", {}),
                            "result": tc.get("result"),
                        }
                    )

            if isinstance(body.get("tool_calls"), list):
                for tc in body["tool_calls"]:
                    if not isinstance(tc, dict):
                        continue
                    name = tc.get("name") or tc.get("function")
                    if not name:
                        continue
                    out.append(
                        {
                            "function": name,
                            "arguments": tc.get("arguments", {}),
                            "result": tc.get("result"),
                        }
                    )

        return out

    @staticmethod
    def _normalize_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for call in tool_calls or []:
            if not isinstance(call, dict):
                continue

            fn = call.get("function") or call.get("name") or ""
            if not fn:
                continue

            args = call.get("arguments", {})
            if not isinstance(args, dict):
                args = {}

            normalized.append(
                {
                    "function": fn,
                    "arguments": args,
                    "result": call.get("result"),
                }
            )
        return normalized

    def _normalize_real_tool_call_arguments(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Normalize recorded real-tool arguments to the tool schema shape."""
        normalized_calls = self._normalize_tool_calls(tool_calls)
        if not normalized_calls:
            return normalized_calls

        aligned: List[Dict[str, Any]] = []
        for call in normalized_calls:
            out_call = copy.deepcopy(call)
            call_args = call.get("arguments", {})
            real_tool_name = self._resolve_real_tool_name_for_operation(str(call.get("function", "")))
            request_schema = self._real_tool_argument_schemas.get(real_tool_name or "", {})
            out_call["arguments"] = MockServerClient.normalize_arguments_to_schema(
                call_args,
                request_schema,
            )
            aligned.append(out_call)

        return aligned

    @staticmethod
    def _unwrap_request_body(arguments: Any) -> Any:
        if isinstance(arguments, dict) and "requestBody" in arguments and isinstance(arguments["requestBody"], dict):
            return arguments["requestBody"]
        return arguments

    @staticmethod
    def _canonical_tool_name(name: str) -> str:
        lowered = (name or "").strip().lower()
        for prefix in TOOL_NAME_PREFIXES:
            if lowered.startswith(prefix):
                return lowered[len(prefix):]
        return lowered

    def _build_wrapped_real_tools(
        self,
        test_case: TestCase,
        *,
        expected_session_id: Optional[str] = None,
        session_context: Optional[SessionContext] = None,
        openapi_base_dir: Optional[Path] = None,
    ) -> List[Any]:
        """Build wrapped real tools for task-agent execution using prefixed BFCL names."""
        if not self.real_tool_instances:
            return []

        tool_defs = self._get_tool_domains_for_test(test_case)
        tools: List[Any] = []
        self._real_tool_by_name = {}
        self._real_tool_ops_index = {}
        self._real_tool_argument_schemas = {}
        base_dir = openapi_base_dir or Path("data/bfcl_v4/openapi/multi_turn")

        for canonical_name, domain_cfg in tool_defs:
            schema_file = base_dir / f"{domain_cfg['schema']}.json"
            if not schema_file.exists():
                logger.warning("OpenAPI spec not found for %s: %s", canonical_name, schema_file)
                continue

            with schema_file.open("r", encoding="utf-8") as f:
                spec = json.load(f)

            for path, item in (spec.get("paths") or {}).items():
                post = item.get("post", {}) if isinstance(item, dict) else {}
                fn_base = path.lstrip("/")
                if not fn_base:
                    continue
                if self._is_excluded_operation(fn_base):
                    continue

                for prefix in domain_cfg.get("prefixes", []):
                    tool_name = f"{prefix}_{fn_base}"
                    request_schema = (
                        post.get("requestBody", {})
                        .get("content", {})
                        .get("application/json", {})
                        .get("schema", {})
                    )
                    if not isinstance(request_schema, dict):
                        request_schema = {"type": "object", "properties": {}}
                    else:
                        request_schema = copy.deepcopy(request_schema)

                    # OpenAI strict mode requires root required keys to match properties keys exactly.
                    request_schema["type"] = "object"
                    properties = request_schema.get("properties")
                    if not isinstance(properties, dict):
                        properties = {}
                    request_schema["properties"] = properties
                    request_schema["required"] = list(properties.keys())
                    if "additionalProperties" not in request_schema:
                        request_schema["additionalProperties"] = False
                    openai_tool_schema = {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": post.get("description") or post.get("summary") or f"Real tool: {tool_name}",
                            "strict": False,
                            "parameters": request_schema,
                        },
                    }

                    def _dispatch(_tool_name=tool_name, **kwargs):
                        return self.mock_client._execute_single_real_tool(
                            self.real_tool_instances,
                            _tool_name,
                            kwargs,
                        )

                    _dispatch.__name__ = tool_name
                    _dispatch.__doc__ = post.get("description") or post.get("summary") or f"Real tool: {tool_name}"

                    wrapped = wrap_real_tool(
                        _dispatch,
                        function_name=tool_name,
                        sync_config=True,
                        openai_tool_schema=openai_tool_schema,
                        argument_normalizer=lambda kwargs, _schema=copy.deepcopy(request_schema): MockServerClient.normalize_arguments_to_schema(
                            kwargs,
                            _schema,
                        ),
                        expected_session_id=expected_session_id,
                        strict_sync=True,
                        session_context=session_context,
                    )
                    tools.append(wrapped)
                    self._real_tool_by_name[tool_name] = wrapped
                    self._real_tool_argument_schemas[tool_name] = copy.deepcopy(request_schema)
                    self._real_tool_ops_index.setdefault(self._canonical_tool_name(fn_base), []).append(tool_name)
                    self._real_tool_ops_index.setdefault(self._canonical_tool_name(tool_name), []).append(tool_name)

        logger.info("Built %d wrapped real tools", len(tools))
        return tools

    def _initialize_real_tool_instances(
        self,
        test_case: TestCase,
        initial_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Initialize BFCL real tool instances once and keep them shared across turns."""
        self._patch_bfcl_dependencies()

        source_path = self._resolve_bfcl_func_source_path()
        if str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))

        domains = self._get_tool_domains_for_test(test_case)
        instances: Dict[str, Any] = {}

        for canonical_name, domain_cfg in domains:
            try:
                module = importlib.import_module(domain_cfg["module"])
                cls = getattr(module, domain_cfg["class"])
                instance = cls()

                scenario = self._find_scenario_for_domain(
                    canonical_name=canonical_name,
                    initial_config=initial_config,
                    aliases=domain_cfg.get("aliases", []),
                )
                if hasattr(instance, "_load_scenario") and isinstance(scenario, dict):
                    instance._load_scenario(copy.deepcopy(scenario))

                instances[canonical_name] = instance
                logger.info("Initialized real instance: %s", canonical_name)
            except Exception as e:
                logger.error("Failed to initialize %s: %s", canonical_name, e, exc_info=True)

        return instances

    @staticmethod
    def _resolve_bfcl_func_source_path() -> Path:
        """Resolve BFCL python tool source directory from current supported locations."""
        candidates = [Path("benchmarks/bfcl/multi_turn/func_source_code").resolve()]
        for path in candidates:
            if path.exists():
                return path
        # Return preferred path for clearer error message downstream.
        return candidates[0]

    @staticmethod
    def _find_scenario_for_domain(
        *,
        canonical_name: str,
        initial_config: Dict[str, Any],
        aliases: List[str],
    ) -> Dict[str, Any]:
        if not isinstance(initial_config, dict):
            return {}

        for key in [canonical_name] + list(aliases or []):
            if key in initial_config and isinstance(initial_config[key], dict):
                return initial_config[key]
        return {}

    @staticmethod
    def _patch_bfcl_dependencies() -> None:
        """Patch long_context import aliases expected by BFCL source modules."""
        aliases = ["benchmarks.bfcl.multi_turn.func_source_code.long_context"]
        if all(alias in sys.modules for alias in aliases) and "long_context" in sys.modules:
            return

        source_dir = MultiTurnExecutor._resolve_bfcl_func_source_path()
        long_context_path = (source_dir / "long_context.py").resolve()
        if not long_context_path.exists():
            logger.warning("long_context.py not found at %s", long_context_path)
            return

        spec = importlib.util.spec_from_file_location("long_context", str(long_context_path))
        if spec is None or spec.loader is None:
            logger.warning("Failed to load long_context module spec")
            return

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Top-level import style: from long_context import ...
        sys.modules.setdefault("long_context", module)

        # Namespaced import style used by BFCL source modules.
        for alias in aliases:
            package_parts = alias.split(".")[:-1]
            for i in range(1, len(package_parts) + 1):
                name = ".".join(package_parts[:i])
                if name in sys.modules:
                    continue
                try:
                    importlib.import_module(name)
                    continue
                except Exception:
                    pass

                # Fallback: create a package-like module only when import fails.
                pkg_module = types.ModuleType(name)
                pkg_module.__path__ = []  # Mark as package.
                sys.modules[name] = pkg_module

            sys.modules[alias] = module

    def _ensure_bfcl_real_api_schema(self) -> None:
        """Ensure BFCL RealAPI schema file exists before wrapping real tools."""
        target = Path("data/bfcl_v4/openapi/multi_turn/real/BFCLMultiTurnRealAPI.json")
        if target.exists():
            return

        script_path = Path("scripts/generate_bfcl_real_api_schema.py")
        if not script_path.exists():
            raise FileNotFoundError(
                "BFCL real schema missing and generator script not found: "
                f"{target}"
            )

        spec = importlib.util.spec_from_file_location("generate_bfcl_real_api_schema", str(script_path.resolve()))
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to import BFCL real schema generator")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "generate_bfcl_real_api_schema"):
            raise RuntimeError("Schema generator missing generate_bfcl_real_api_schema")

        module.generate_bfcl_real_api_schema(Path.cwd())
        if not target.exists():
            raise RuntimeError(f"Failed to generate BFCL real schema: {target}")

    def _get_tool_domains_for_test(self, test_case: TestCase) -> List[Tuple[str, Dict[str, Any]]]:
        """Resolve canonical tool domains needed by this test case."""
        metadata = getattr(test_case, "metadata", {}) or {}
        involved_classes = metadata.get("involved_classes", []) or []

        if not involved_classes:
            initial_cfg = metadata.get("initial_config", {}) or {}
            involved_classes = list(initial_cfg.keys())

        domains: List[Tuple[str, Dict[str, Any]]] = []
        seen = set()
        for cls_name in involved_classes:
            canonical = ALIAS_TO_CANONICAL.get(cls_name)
            if not canonical:
                continue
            if canonical in seen:
                continue
            seen.add(canonical)
            domains.append((canonical, TOOL_DOMAIN_CONFIG[canonical]))

        return domains

    def _get_openapi_paths_for_test(
        self,
        test_case: TestCase,
        *,
        excluded_operations: Optional[set[str]] = None,
        test_id: Optional[str] = None,
    ) -> Tuple[List[str], Path]:
        """Get OpenAPI paths for SimSolver mock execution."""
        domains = self._get_tool_domains_for_test(test_case)
        base_dir = Path("data/bfcl_v4/openapi/multi_turn")

        paths: List[str] = []
        excluded = {self._canonical_tool_name(op) for op in (excluded_operations or set()) if op}

        if not excluded:
            for _, domain_cfg in domains:
                path = base_dir / f"{domain_cfg['schema']}.json"
                if path.exists() and str(path) not in paths:
                    paths.append(str(path))
            if not paths:
                raise ValueError("No BFCL multi-turn OpenAPI specs found for test case")
            return paths, base_dir

        temp_root = Path(tempfile.mkdtemp(prefix="bfcl_openapi_filtered_"))
        for _, domain_cfg in domains:
            src_path = base_dir / f"{domain_cfg['schema']}.json"
            if not src_path.exists():
                continue
            with src_path.open("r", encoding="utf-8") as f:
                spec = json.load(f)

            filtered_paths: Dict[str, Any] = {}
            for path_key, path_item in (spec.get("paths") or {}).items():
                post = path_item.get("post", {}) if isinstance(path_item, dict) else {}
                operation_id = post.get("operationId") or path_key.lstrip("/")
                if self._canonical_tool_name(str(operation_id)) in excluded:
                    continue
                filtered_paths[path_key] = path_item
            spec["paths"] = filtered_paths

            dst_path = temp_root / f"{domain_cfg['schema']}.json"
            with dst_path.open("w", encoding="utf-8") as f:
                json.dump(spec, f, ensure_ascii=False, indent=2)
            paths.append(str(dst_path))

        if not paths:
            raise ValueError("No BFCL multi-turn OpenAPI specs found for test case after excluded_function filtering")

        logger.info(
            "Prepared filtered OpenAPI specs for test=%s, excluded_operations=%s",
            test_id or "",
            sorted(list(excluded)),
        )
        return paths, temp_root

    def _extract_excluded_operations(self, test_case: TestCase, bfcl_test_case: Any) -> set[str]:
        excluded: set[str] = set()

        raw_data: Dict[str, Any] = {}
        if hasattr(test_case, "metadata") and isinstance(test_case.metadata, dict):
            raw_data = test_case.metadata.get("raw_data", {}) or {}
        if not raw_data and hasattr(bfcl_test_case, "metadata") and isinstance(bfcl_test_case.metadata, dict):
            raw_data = bfcl_test_case.metadata.get("raw_data", {}) or {}

        candidate = raw_data.get("excluded_function")
        if candidate is None and isinstance(raw_data.get("content"), dict):
            candidate = raw_data["content"].get("excluded_function")
        if candidate is None and hasattr(bfcl_test_case, "excluded_function"):
            candidate = getattr(bfcl_test_case, "excluded_function")

        if isinstance(candidate, str):
            candidate = [candidate]

        if isinstance(candidate, list):
            for item in candidate:
                if item is None:
                    continue
                excluded.add(self._canonical_tool_name(str(item)))
        return excluded

    def _is_excluded_operation(self, operation: str) -> bool:
        if not self._excluded_operations:
            return False
        return self._canonical_tool_name(operation) in self._excluded_operations

    def _resolve_real_tool_name_for_operation(self, operation: str) -> Optional[str]:
        op = self._canonical_tool_name(operation)
        if op in self._real_tool_by_name:
            return op
        candidates = self._real_tool_ops_index.get(op, [])
        return candidates[0] if candidates else None

    @staticmethod
    def _extract_initial_config(test_case: TestCase, bfcl_test_case: TestCase) -> Dict[str, Any]:
        if hasattr(test_case, "metadata") and isinstance(test_case.metadata, dict):
            cfg = test_case.metadata.get("initial_config", {})
            if isinstance(cfg, dict):
                return copy.deepcopy(cfg)

        if hasattr(bfcl_test_case, "metadata") and isinstance(bfcl_test_case.metadata, dict):
            cfg = bfcl_test_case.metadata.get("initial_config", {})
            if isinstance(cfg, dict):
                return copy.deepcopy(cfg)

        return {}

    @staticmethod
    def _extract_turn_text(question_turn: Any) -> str:
        if isinstance(question_turn, list) and question_turn:
            first = question_turn[0]
            if isinstance(first, dict) and first.get("content"):
                return str(first["content"])
        return str(question_turn)

    def _format_tool_calls_for_output(self, tool_calls: List[Dict[str, Any]]) -> List[str]:
        """Format tool calls in BFCL-friendly function-call style."""
        formatted: List[str] = []

        for call in tool_calls:
            if isinstance(call, str):
                formatted.append(call)
                continue

            fn = call.get("function", "")
            args = call.get("arguments", {})
            if not fn:
                continue

            for prefix in TOOL_NAME_PREFIXES:
                if fn.startswith(prefix):
                    fn = fn[len(prefix):]
                    break

            if isinstance(args, dict) and "requestBody" in args and isinstance(args["requestBody"], dict):
                args = args["requestBody"]

            if isinstance(args, dict) and args:
                pieces = []
                for k, v in args.items():
                    if isinstance(v, str):
                        escaped = v.replace("\n", "\\n")
                        pieces.append(f"{k}='{escaped}'")
                    else:
                        pieces.append(f"{k}={v}")
                formatted.append(f"{fn}({','.join(pieces)})")
            else:
                formatted.append(f"{fn}()")

        return formatted

    def format_result_for_bfcl(self, result: InferenceResult) -> Dict[str, Any]:
        """Format result to BFCL output shape."""
        turns_tool_calls: List[List[str]] = []
        for turn in result.turns:
            turns_tool_calls.append(self._format_tool_calls_for_output(turn.mock_tool_calls))

        return {
            "id": result.test_id,
            "result": turns_tool_calls,
            "input_token_count": 0,
            "output_token_count": 0,
            "latency": result.total_time,
            "reasoning_content": "",
        }

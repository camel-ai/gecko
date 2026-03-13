import logging
from copy import deepcopy
from typing import Any, Dict, List

from benchmarks.base.test_case import TestCase
from inference.core.sim_solver import SimSolver, TurnOutcome, SimEvent

from gats.core.config import GATSConfig
from gats.core.task import GATSTask, GATSAttempt, GATSTurn

logger = logging.getLogger(__name__)


def _build_test_case(task: GATSTask) -> TestCase:
    """Create a minimal TestCase from a GATSTask for SimSolver compatibility."""
    return TestCase(
        id=task.id,
        metadata={
            "initial_config": deepcopy(task.initial_config) if task.initial_config else {},
            **task.metadata,
        },
    )


def _outcome_to_gats_turn(outcome: TurnOutcome, question: str) -> GATSTurn:
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

    return GATSTurn(
        index=outcome.turn_idx,
        question=question,
        best_attempt=outcome.best_attempt,
        score=outcome.score,
        attempts=gats_attempts,
        checklist=checklist,
        config_after=deepcopy(outcome.final_config),
        execution_time=outcome.execution_time,
    )


class GATSSolver:
    """Wraps the existing SimSolver with GATS types.

    One GATSSolver is created per task. Call ``process_turn(question)``
    for each turn; SimSolver maintains internal state across turns.
    """

    def __init__(self, task: GATSTask, config: GATSConfig):
        test_case = _build_test_case(task)

        # Per-task agent prompt takes precedence over config-level prompt.
        effective_agent_prompt = (
            task.agent_prompt
            if task.agent_prompt is not None
            else config.agent_prompt
        )

        self._solver = SimSolver(
            test_case=test_case,
            initial_config=task.initial_config,
            model_name=config.model,
            max_retries=config.max_retries,
            agent_timeout=config.agent_timeout,
            mock_server_url=config.gecko_url,
            override_openapi_server=config.override_openapi_servers,
            agent_max_iteration=config.agent_max_iterations,
            enable_evaluation=(config.max_retries > 0),
            enable_checklist=config.enable_checklist,
            agent_system_prompt=effective_agent_prompt,
            judge_system_prompt=config.judge_prompt,
            openapi_tool_paths=task.tool_schemas,
            agent_persistence_mode=config.agent_persistence,
            enable_tool_filtering=config.enable_tool_filtering,
            base_checklist_items=config.base_checklist_items,
            checklist_system_prompt=config.checklist_prompt,
            include_agent_response_in_judge=config.include_agent_response_in_judge,
            enable_tool_result_folding=config.enable_tool_result_folding,
            collect_mock_server_usage=config.collect_gecko_usage,
            enable_debug=config.debug,
            verbose_debug=config.verbose,
        )

    def process_turn(self, question: str) -> GATSTurn:
        """Process one turn and return a GATSTurn."""
        outcome: TurnOutcome = self._solver.process(question)
        return _outcome_to_gats_turn(outcome, question)

    def get_events(self) -> List[SimEvent]:
        """Return all accumulated SimEvents (deep copy)."""
        return self._solver.get_events()

    @property
    def current_config(self) -> Dict[str, Any]:
        return self._solver.current_config

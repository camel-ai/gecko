#!/usr/bin/env python3
"""
SimSolver - Core simulation solver for mock server based execution
Handles multi-turn conversations with retry mechanisms and state management
"""

import json
import logging
import time
from typing import Callable, Dict, List, Any, Optional, Tuple, TYPE_CHECKING, Literal
from dataclasses import dataclass, field
from copy import deepcopy

from benchmarks.base.test_case import TestCase
from inference.client_engine import MockServerClient, TaskInfraError
from inference.core.agent_executor import AgentExecutor
from inference.core.eval_coordinator import EvalCoordinator
from inference.core.message_builder import MessageBuilder
from inference.core.tool_loader import ToolLoader
from inference.utils.provider_errors import is_retryable_provider_error
from inference.evaluation.task_feedback import TaskFeedback
from utils.conversation import render_conversation
from utils.reasoning import strip_reasoning
from utils.conversation_memory import ConversationMemoryStore
from utils.test_case_adapter import TestCaseAdapter
from inference.utils.log_context import (
    TaskContextFilter,
    bind_log_context,
    clear_task_context,
    set_task_context,
)

if TYPE_CHECKING:
    from inference.real_tools import ToolRegistry

logger = logging.getLogger(__name__)

_camel_logger = logging.getLogger("camel")
_filter_installed = any(isinstance(f, TaskContextFilter) for f in _camel_logger.filters)
if not _filter_installed:
    _camel_logger.addFilter(TaskContextFilter())


@dataclass
class SimEvent:
    """A single execution event in SimSolver.

    This is the primary external data model, replacing the older split between
    AttemptData/ConversationTurn/MultiTurnExample.
    """

    type: Literal[
        "history_set",
        "initial_config_set",
        "calibration",
        "turn_start",
        "checklist",
        "attempt_start",
        "agent_prompt",
        "agent_response",
        "tool_calls",
        "inherited_evidence",
        "attempt_config",
        "judge",
        "attempt_end",
        "turn_end",
    ]
    ts: float
    turn_idx: int
    attempt: Optional[int] = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnOutcome:
    """Summary for a single processed user turn, with full event detail."""

    turn_idx: int
    user_message: str
    best_attempt: int
    score: float
    final_tool_calls: List[Dict[str, Any]]
    final_config: Dict[str, Any]
    execution_time: float
    events: List[SimEvent]


@dataclass
class _AttemptState:
    """Internal attempt summary used for retry selection."""

    attempt: int
    tool_calls: List[Dict[str, Any]]
    score: float
    feedback: Dict[str, Any]
    session_id: str
    mock_config: Dict[str, Any]
    execution_time: float
    agent_response: str = ""
    tools_count: int = 0
    agent_prompt: str = ""
    inherited_tool_calls: List[Dict[str, Any]] = field(default_factory=list)


class SimSolver:
    """
    Core simulation solver for mock server based execution
    
    Key responsibilities:
    - Manage mock server sessions
    - Process turns with retry logic
    - Maintain conversation state
    - Collect attempt data for examples
    - Support config sync from real execution
    """
    
    def __init__(self,
                 test_case: TestCase,
                 initial_config: Optional[Dict[str, Any]] = None,
                 model_name: str = "gpt-4.1-mini",
                 max_retries: int = 2,
                 agent_timeout: Optional[int] = None,
                 mock_server_url: str = "http://localhost:8000",
                 override_openapi_server: bool = True,
                 agent_max_iteration: Optional[int] = None,
                 agent_summarize_threshold: Optional[int] = None,
                 enable_evaluation: bool = True,
                 enable_checklist: bool = True,
                 agent_system_prompt: Optional[str] = None,
                 judge_system_prompt: Optional[str] = None,
                 triage_judge_prompt: Optional[str] = None,
                 tool_registry: Optional['ToolRegistry'] = None,
                 openapi_tool_paths: Optional[List[str]] = None,
                 agent_persistence_mode: bool = False,
                 base_checklist_items: Optional[List[str]] = None,
                 checklist_system_prompt: Optional[str] = None,
                 include_agent_response_in_judge: bool = True,
                 enable_tool_result_folding: bool = True,
                 append_base_checklist_to_generated: bool = True,
                 fetch_attempt_state: bool = True,
                 enable_readonly_evidence_branch: bool = False,
                 read_only_sync_policy: Optional[str] = None,
                 readonly_evidence_result_projector: Optional[Callable[[Any, int], Any]] = None,
                 readonly_evidence_result_max_chars: int = 1200,
                 enable_debug: bool = False,
                 verbose_debug: bool = False,
):
        """
        Initialize SimSolver

        Args:
            test_case: Test case containing tools, initial config, etc.
            initial_config: Explicit initial config to use (overrides test_case.metadata['initial_config']).
            model_name: Model to use for inference and evaluation
            max_retries: Maximum retry attempts per turn
            mock_server_url: Mock server endpoint
            override_openapi_server: Whether to override OpenAPI servers with mock_server_url
            agent_max_iteration: Max iterations per agent step (None uses ChatAgent default)
            agent_summarize_threshold: CAMEL summarization threshold (None disables)
            enable_evaluation: Whether to use checklist/judge evaluation
            enable_checklist: Whether to generate dynamic checklist. Dynamic checklist
                generation also requires checklist_system_prompt; without an explicit
                prompt it is disabled.
            agent_system_prompt: Custom system prompt for task agent (optional)
            judge_system_prompt: Custom system prompt for judge LLM (optional)
            tool_registry: ToolRegistry with hybrid real/mock tools (optional, takes precedence over openapi_tool_paths)
            openapi_tool_paths: List of OpenAPI spec file paths to load tools from (optional, used if tool_registry not provided)
            agent_persistence_mode: Enable agent persistence using CAMEL clone (default: False)
                - True: Clone agents with memory for retries and across turns
                - False: Create new agents for each attempt (current behavior)
            base_checklist_items: Base checklist items to append to generated items (optional)
                - If None, uses default standard checks (backward compatible)
                - If [], no base items are added
                - If list of strings, uses custom base items
            checklist_system_prompt: Custom system prompt dedicated to checklist generation
            include_agent_response_in_judge: Whether to pass assistant text response into judge prompt
            enable_tool_result_folding: Whether to fold large tool results into TOOL_RESULT_REF
                for judge input (default: True)
            append_base_checklist_to_generated: Whether generated dynamic checklists
                should be extended with base_checklist_items. Fixed-checklist mode
                always uses base_checklist_items directly.
            fetch_attempt_state: Whether to fetch session state from Gecko after each attempt.
                Single-turn BFCL evaluation only needs judged tool calls, so this can be
                disabled to avoid hard dependency on /get-session-state.
            enable_readonly_evidence_branch: Whether retries may inherit successful
                read-only prefix evidence from prior attempts in the same turn. The
                inherited evidence is only used in prompts/judging; no state-changing
                effects are inherited and sessions still start from the turn base state.
                Defaults to False so BFCL behavior is unchanged.
            read_only_sync_policy: Optional policy forwarded when real-tool buffers
                flush to Gecko. Leave unset for default BFCL/mock semantics; tau2
                hybrid uses an explicit observed-state policy.
            readonly_evidence_result_projector: Optional callable used to compact
                inherited read-only results before rendering them to retry prompts.
        """
        self.test_case = test_case
        self._initial_config = deepcopy(initial_config) if initial_config is not None else None
        self.model_name = model_name
        self.max_retries = max_retries
        self.agent_timeout = agent_timeout
        self.enable_checklist = bool(enable_checklist and checklist_system_prompt)
        self.target_score = 1.0
        self.mock_server_url = mock_server_url
        self.override_openapi_server = override_openapi_server
        self.agent_max_iteration = agent_max_iteration if agent_max_iteration is not None else 10
        self.agent_summarize_threshold = agent_summarize_threshold
        self.agent_system_prompt = agent_system_prompt
        self.judge_system_prompt = judge_system_prompt
        self.triage_judge_prompt = triage_judge_prompt
        self.tool_registry = tool_registry
        self.openapi_tool_paths = openapi_tool_paths
        self.agent_persistence_mode = agent_persistence_mode
        self.base_checklist_items = base_checklist_items
        self.checklist_system_prompt = checklist_system_prompt
        self.include_agent_response_in_judge = include_agent_response_in_judge
        self.enable_tool_result_folding = bool(enable_tool_result_folding)
        self.append_base_checklist_to_generated = bool(append_base_checklist_to_generated)
        self.fetch_attempt_state = bool(fetch_attempt_state)
        self.enable_readonly_evidence_branch = bool(enable_readonly_evidence_branch)
        self.read_only_sync_policy = (read_only_sync_policy or "").strip() or None
        self.readonly_evidence_result_projector = readonly_evidence_result_projector
        self.readonly_evidence_result_max_chars = int(readonly_evidence_result_max_chars or 1200)
        self.enable_debug = bool(enable_debug)
        self.verbose_debug = bool(verbose_debug)

        raw_task_id = TestCaseAdapter.get_id(self.test_case) or "unknown_task"
        self._cache_task_id = "".join(
            ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(raw_task_id)
        )

        if tool_registry is None and openapi_tool_paths is None:
            raise ValueError("Must provide either tool_registry or openapi_tool_paths")
        
        self.mock_client = MockServerClient(mock_server_url)
        if self.agent_timeout is not None:
            try:
                self.mock_client.timeout = float(self.agent_timeout)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid agent_timeout=%s for mock client timeout; keep default %.1fs",
                    self.agent_timeout,
                    self.mock_client.timeout,
                )
        
        if enable_evaluation and max_retries > 0:
            _evaluator = TaskFeedback(
                model_name=self.model_name,
                system_prompt=judge_system_prompt,
                base_checklist_items=[],
                checklist_system_prompt=checklist_system_prompt,
                timeout=self.agent_timeout,
            )
        else:
            _evaluator = None
        self._eval = EvalCoordinator(
            evaluator=_evaluator,
            enable_checklist=self.enable_checklist,
            enable_tool_result_folding=enable_tool_result_folding,
            include_agent_response_in_judge=include_agent_response_in_judge,
            base_checklist_items=base_checklist_items,
            append_base_checklist_to_generated=self.append_base_checklist_to_generated,
        )
            
        self._custom_tools = None
        self._openapi_toolkit = None  # Store toolkit instance for session management
        self._tool_definitions: List[Dict[str, Any]] = []  # Simple tool definitions for judge/checklist prompts
        self._tool_catalog: List[Dict[str, str]] = []  # Lightweight name+description list

        if tool_registry is not None:
            self._custom_tools = tool_registry.get_all_tools()
            logger.info(f"SimSolver using ToolRegistry: "
                       f"{len(tool_registry.get_real_tools())} real tools, "
                       f"{len(tool_registry.get_mock_tools())} mock tools")
        elif openapi_tool_paths:
            override_url = self.mock_server_url if self.override_openapi_server else None
            self._custom_tools, self._openapi_toolkit = ToolLoader.load_from_openapi(
                openapi_tool_paths, override_server_url=override_url, tool_registry=self.tool_registry
            )

        try:
            self._tool_definitions = ToolLoader.build_simple_definitions(self._custom_tools)
            self._tool_catalog = ToolLoader.build_tool_catalog(self._tool_definitions)
            self._tool_metadata_by_name = self._build_tool_metadata_by_name(self._tool_definitions)
            if self._tool_definitions:
                logger.info(f"[SIMSOLVER] Built {len(self._tool_definitions)} simple tool definitions for judge/checklist")
        except Exception as e:
            logger.warning(f"[SIMSOLVER] Failed to build tool definitions: {e}")
            self._tool_metadata_by_name = {}

        self._initialize_state()

        self._agent_exec = AgentExecutor(
            model_name=self.model_name,
            agent_timeout=self.agent_timeout,
            agent_system_prompt=self.agent_system_prompt,
            agent_max_iteration=self.agent_max_iteration,
            agent_summarize_threshold=self.agent_summarize_threshold,
            tools=self._custom_tools or [],
        )

        self._current_agent = None

        self._turn_base_agent = None  # Base agent for current turn
        self._best_agent = None  # Best agent from all attempts
        self._attempt_agents = []  # List of (agent, score) tuples for current turn
    
        logger.info(f"SimSolver initialized for test {TestCaseAdapter.get_id(test_case)}")

    @staticmethod
    def _build_tool_metadata_by_name(tool_definitions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        metadata: Dict[str, Dict[str, Any]] = {}
        for definition in tool_definitions or []:
            if not isinstance(definition, dict):
                continue
            name = str(definition.get("name") or "").strip()
            if not name:
                continue
            metadata[name] = {
                "state_access": definition.get("state_access"),
                "state_effects": definition.get("state_effects") or [],
            }
        return metadata

    def _initialize_state(self):
        """Initialize internal state"""
        config_from_case = {}
        if hasattr(self.test_case, "metadata") and isinstance(self.test_case.metadata, dict):
            config_from_case = self.test_case.metadata.get("initial_config", {}) or {}

        base_config = self._initial_config if self._initial_config is not None else config_from_case
        if base_config is None:
            base_config = {}

        self._current_config: Dict[str, Any] = deepcopy(base_config)
        self._config_history: List[Dict[str, Any]] = [deepcopy(self._current_config)]
        self._task_state_initialized = False
        self._bootstrapped_initial_config: Dict[str, Any] = deepcopy(self._current_config)

        self._history_items: List[Dict[str, Any]] = []

        self._memory_store = ConversationMemoryStore()

        self._events: List[SimEvent] = []

        self._turn_count = 0
        self._total_attempts = 0

        self._latest_checklist: List[Dict[str, Any]] = []

    def set_history(self, history: Any) -> None:
        """Set conversation history used for agent prompt and judge context.

        Accepted inputs:
        - list[dict]: already in render_conversation-compatible schema
        - dict: may contain 'structured_conversation' or 'messages'
        """
        items: List[Dict[str, Any]] = []

        if history is None:
            items = []
        elif isinstance(history, dict):
            structured = history.get("structured_conversation")
            messages = history.get("messages")
            if isinstance(structured, list):
                items = [i for i in structured if isinstance(i, dict)]
            elif isinstance(messages, list):
                items = [i for i in messages if isinstance(i, dict)]
        elif isinstance(history, list):
            items = [i for i in history if isinstance(i, dict)]
        else:
            raise TypeError(f"Unsupported history type: {type(history)}")

        self._history_items = deepcopy(items)
        self._events.append(
            SimEvent(
                type="history_set",
                ts=time.time(),
                turn_idx=max(self._turn_count, 0),
                data={"items": len(self._history_items)},
            )
        )

    def set_initial_config(self, initial_config: Optional[Dict[str, Any]]) -> None:
        """Set the initial config used for subsequent attempts/turns."""
        cfg = deepcopy(initial_config) if initial_config is not None else {}
        self._current_config = cfg
        if not self._task_state_initialized:
            self._bootstrapped_initial_config = deepcopy(cfg)
        self._config_history.append(deepcopy(cfg))
        self._events.append(
            SimEvent(
                type="initial_config_set",
                ts=time.time(),
                turn_idx=max(self._turn_count, 0),
                data={"keys": len(cfg) if isinstance(cfg, dict) else 0},
            )
        )

    def get_events(self) -> List[SimEvent]:
        """Return a deep copy of all accumulated events."""
        return deepcopy(self._events)

    def ensure_task_initialized(self) -> None:
        """Initialize task state once via Gecko bootstrap, then reuse it."""
        if self._task_state_initialized:
            return

        base_config = deepcopy(self._current_config) if isinstance(self._current_config, dict) else {}
        involved_classes = TestCaseAdapter.get_involved_classes(self.test_case) or []
        if not involved_classes and isinstance(base_config, dict):
            involved_classes = [
                key for key, value in base_config.items()
                if key != "runtime_state" and isinstance(value, dict)
            ]

        init_session_id = self.mock_client.create_session(self.test_case)
        initialized_state = self.mock_client.init_session_state(
            init_session_id,
            base_config,
            involved_classes=involved_classes or None,
        )
        if isinstance(initialized_state, dict):
            self._current_config = deepcopy(initialized_state)
        else:
            self._current_config = deepcopy(base_config)

        self._bootstrapped_initial_config = deepcopy(self._current_config)
        if self._config_history:
            self._config_history[0] = deepcopy(self._current_config)
        else:
            self._config_history.append(deepcopy(self._current_config))
        self._task_state_initialized = True
        logger.info(
            "Initialized task state once for %s (%d top-level keys)",
            TestCaseAdapter.get_id(self.test_case),
            len(self._current_config) if isinstance(self._current_config, dict) else 0,
        )

    def process(self, user_message: str) -> TurnOutcome:
        """Process a single turn with retry logic.

        - Each attempt runs in a fresh mock-server session.
        - Retries within a turn always start from the same turn-start config.
        - After the turn, SimSolver updates its internal config to the best attempt's config.
        """
        self.ensure_task_initialized()

        turn_start = time.time()
        turn_idx = self._turn_count
        self._turn_count += 1

        logger.info(f"Processing turn {turn_idx}: {user_message[:100]}...")

        turn_base_config = deepcopy(self._current_config) if isinstance(self._current_config, dict) else {}

        turn_events: List[SimEvent] = []
        turn_events.append(
            SimEvent(
                type="turn_start",
                ts=time.time(),
                turn_idx=turn_idx,
                data={
                    "user_message_preview": user_message[:200],
                    "max_retries": self.max_retries,
                    "history_items": len(self._history_items),
                    "config_keys": len(turn_base_config),
                },
            )
        )

        if self.agent_persistence_mode:
            if self._best_agent:
                self._turn_base_agent = self._best_agent
                logger.info(f"Using best agent from previous turn as base for turn {turn_idx}")
            elif not self._turn_base_agent:
                logger.info(f"First turn {turn_idx}, will create initial agent")
            self._attempt_agents = []

        checklist: List[Dict[str, Any]] = []

        if self._eval.should_generate and self.max_retries > 0:
            checklist = self._eval.generate_checklist(
                user_message, self._history_items,
                self._tool_catalog, self.agent_system_prompt,
            )
        turn_events.append(
            SimEvent(
                type="checklist",
                ts=time.time(),
                turn_idx=turn_idx,
                data={
                    "items": len(checklist),
                    "checklist": deepcopy(checklist),
                },
            )
        )

        attempts: List[_AttemptState] = []
        triage_clarified = False
        for attempt in range(self.max_retries + 1):
            strip_tools = False
            if triage_clarified:
                strip_tools = True

            attempt_state, attempt_events = self._execute_attempt(
                user_message=user_message,
                attempt=attempt,
                previous_attempts=attempts,
                checklist=checklist,
                base_config=turn_base_config,
                turn_idx=turn_idx,
                strip_tools=strip_tools,
            )
            attempts.append(attempt_state)
            self._total_attempts += 1
            turn_events.extend(attempt_events)

            feedback = attempt_state.feedback if isinstance(attempt_state.feedback, dict) else {}
            if feedback.get("triage_verdict") == "CLARIFY" and not triage_clarified:
                triage_clarified = True
                logger.info(
                    "[TRIAGE] CLARIFY on attempt %d — will do one toolless retry, "
                    "reason: %s", attempt, feedback.get("triage_reason", "")
                )
                continue

            if triage_clarified:
                logger.info("[TRIAGE] Toolless retry done, stopping.")
                break

            reached_target = attempt_state.score >= self.target_score

            if reached_target or attempt >= self.max_retries:
                break

            logger.info(
                f"Attempt {attempt} score {attempt_state.score:.2f} < {self.target_score}, retrying..."
            )

        if attempts and all(self._is_retryable_provider_failure(a) for a in attempts):
            raise TaskInfraError(
                "UNTESTED_TASK: all SimSolver attempts failed with retryable provider errors"
            )

        best_attempt_index, best_attempt = max(
            enumerate(attempts),
            key=lambda pair: self._attempt_selection_key(pair[0], pair[1]),
        )

        if not best_attempt.tool_calls and best_attempt.score < self.target_score:
            attempts_with_calls = [
                (i, a) for i, a in enumerate(attempts)
                if a.tool_calls and a.score >= best_attempt.score
            ]
            if attempts_with_calls:
                alt_idx, alt = max(
                    attempts_with_calls,
                    key=lambda pair: (pair[1].score, pair[0]),
                )
                logger.info(
                    f"Anti-abstention: overriding empty attempt {best_attempt_index} "
                    f"(score {best_attempt.score:.2f}) with attempt {alt_idx} "
                    f"(score {alt.score:.2f}, {len(alt.tool_calls)} calls)"
                )
                best_attempt_index, best_attempt = alt_idx, alt

        if (
            best_attempt.score < self.target_score
            and best_attempt_index != 0
            and attempts[0].score >= best_attempt.score
            and not self._is_agent_failure_attempt(attempts[0])
            and (attempts[0].tool_calls or not best_attempt.tool_calls)
        ):
            logger.info(
                f"Raw fallback: no attempt reached {self.target_score:.2f} "
                f"(best={best_attempt_index} score={best_attempt.score:.2f}), "
                f"reverting to attempt 0 "
                f"(score={attempts[0].score:.2f}, "
                f"{len(attempts[0].tool_calls)} calls)"
            )
            best_attempt_index, best_attempt = 0, attempts[0]

        logger.info(
            f"Selected attempt {best_attempt_index} with score {best_attempt.score:.2f} from {len(attempts)} attempts"
        )

        if self.agent_persistence_mode and self._attempt_agents:
            _, (self._best_agent, best_score) = max(
                enumerate(self._attempt_agents),
                key=lambda pair: (pair[1][1], pair[0]),
            )
            logger.info(f"Selected best agent with score {best_score:.2f} for next turn")

        self._apply_best_attempt_to_state(
            user_message=user_message,
            best_attempt=best_attempt,
        )

        turn_events.append(
            SimEvent(
                type="turn_end",
                ts=time.time(),
                turn_idx=turn_idx,
                data={
                    "best_attempt": best_attempt_index,
                    "score": best_attempt.score,
                    "tool_calls": len(best_attempt.tool_calls),
                    "config_keys": len(best_attempt.mock_config) if isinstance(best_attempt.mock_config, dict) else 0,
                },
            )
        )

        outcome = TurnOutcome(
            turn_idx=turn_idx,
            user_message=user_message,
            best_attempt=best_attempt_index,
            score=best_attempt.score,
            final_tool_calls=best_attempt.tool_calls,
            final_config=deepcopy(best_attempt.mock_config) if isinstance(best_attempt.mock_config, dict) else {},
            execution_time=time.time() - turn_start,
            events=turn_events,
        )

        self._events.extend(turn_events)

        return outcome

    @staticmethod
    def _attempt_failure_type(attempt_state: _AttemptState) -> str:
        feedback = attempt_state.feedback if isinstance(attempt_state.feedback, dict) else {}
        return str(feedback.get("failure_type") or "").strip()

    @staticmethod
    def _is_agent_failure_attempt(attempt_state: _AttemptState) -> bool:
        feedback = attempt_state.feedback if isinstance(attempt_state.feedback, dict) else {}
        return bool(feedback.get("agent_failure"))

    @classmethod
    def _is_retryable_provider_failure(cls, attempt_state: _AttemptState) -> bool:
        if not cls._is_agent_failure_attempt(attempt_state):
            return False
        failure_type = cls._attempt_failure_type(attempt_state)
        if failure_type == "provider_transient_error":
            return True
        return is_retryable_provider_error(attempt_state.agent_response)

    @classmethod
    def _attempt_selection_key(cls, idx: int, attempt_state: _AttemptState) -> Tuple[float, int, int, int]:
        return (
            float(attempt_state.score or 0.0),
            0 if cls._is_agent_failure_attempt(attempt_state) else 1,
            1 if attempt_state.tool_calls else 0,
            idx,
        )

    @property
    def current_config(self) -> Dict:
        """Current configuration state"""
        return deepcopy(self._current_config)

    @property
    def completed_tasks(self) -> List[str]:
        """List of completed task descriptions (for backward compatibility)"""
        return [item.get('content', '') for item in self._history_items if item.get('role') == 'user' and item.get('content')]

    @property
    def conversation_history(self) -> List[Dict[str, Any]]:
        """Full conversation history including tool calls and responses"""
        return deepcopy(self._history_items)

    @property
    def turn_count(self) -> int:
        """Number of processed turns"""
        return self._turn_count

    @property
    def total_attempts(self) -> int:
        """Total number of attempts across all turns"""
        return int(getattr(self, "_total_attempts", 0))

    def _run_triage_judge(
        self,
        user_message: str,
        tool_calls: List[Dict[str, Any]],
        turn_idx: Optional[int] = None,
        attempt: Optional[int] = None,
    ) -> Tuple[Optional[str], str]:
        """Run lightweight triage judge to check if clarification was needed.

        Returns ``("CLARIFY", reason)`` when the triage judge says the agent
        should have asked the user for more information, ``(None, "")``
        otherwise (proceed to Stage 2).
        """
        if not self.triage_judge_prompt:
            return None, ""

        import re as _re
        _msg_lower = user_message.lower()
        for tc in (tool_calls or []):
            fn = (tc.get("function") or tc.get("name") or "").lower()
            args = tc.get("arguments") or tc.get("args") or {}
            if isinstance(args, str):
                try:
                    import json as _json
                    args = _json.loads(args)
                except Exception:
                    args = {}
            if not isinstance(args, dict):
                continue
            if "request" in fn and "url" in args:
                url_val = str(args["url"])
                domain_match = _re.search(r'https?://([^/]+)', url_val)
                if domain_match:
                    domain = domain_match.group(1).lower()
                    if domain not in _msg_lower and url_val.lower() not in _msg_lower:
                        reason = (
                            f"URL fabrication: agent called {fn} with domain "
                            f"'{domain}' which the user never mentioned. "
                            f"The user should provide the actual URL."
                        )
                        logger.info(
                            "[TRIAGE] Deterministic CLARIFY: URL fabrication detected "
                            "(domain=%s not in user message)", domain
                        )
                        return "CLARIFY", reason

        import json_repair
        from utils.model_utils import create_model, sanitize_llm_json_text
        from camel.agents import ChatAgent as CamelChatAgent
        from camel.messages import BaseMessage

        tool_defs_lines: List[str] = []
        for td in (self._tool_definitions or []):
            name = td.get("name", "")
            desc = str(td.get("description", ""))
            params = td.get("parameters", {})
            props = params.get("properties", {}) if isinstance(params, dict) else {}
            req = params.get("required", []) if isinstance(params, dict) else []
            param_lines = []
            for pn, pv in (props.items() if isinstance(props, dict) else []):
                r = " (required)" if pn in req else " (optional)"
                pdesc = str(pv.get("description", "")) if isinstance(pv, dict) else ""
                ptype = str(pv.get("type", "")) if isinstance(pv, dict) else ""
                pdefault = pv.get("default") if isinstance(pv, dict) else None
                default_str = f", default={pdefault!r}" if pdefault is not None else ""
                param_lines.append(f"    - {pn}{r}: {ptype}{default_str} — {pdesc}")
            tool_defs_lines.append(f"Tool: {name}\nDescription: {desc}")
            if param_lines:
                tool_defs_lines.append("Parameters:\n" + "\n".join(param_lines))
            tool_defs_lines.append("")
        tool_defs_text = "\n".join(tool_defs_lines)

        calls_parts: List[str] = []
        for tc in (tool_calls or []):
            if isinstance(tc, dict):
                fn = tc.get("function", tc.get("name", ""))
                args = tc.get("arguments", tc.get("args", {}))
                calls_parts.append(f"{fn}({args})")
        calls_text = "\n".join(calls_parts) if calls_parts else "(no calls)"

        user_prompt = (
            f"USER MESSAGE:\n{user_message}\n\n"
            f"AVAILABLE TOOLS:\n{tool_defs_text}\n\n"
            f"AGENT'S TOOL CALLS:\n{calls_text}\n\n"
            "Should the agent have asked the user for clarification instead of making (or not making) these calls?"
        )

        try:
            model_kwargs = {
                "model_name": self.model_name,
                "temperature": 0.0,
                "max_tokens": 8192,
            }
            if self.agent_timeout is not None:
                model_kwargs["timeout"] = self.agent_timeout
            model = create_model(**model_kwargs)
            sys_msg = BaseMessage.make_assistant_message(
                role_name="TriageJudge", content=self.triage_judge_prompt,
            )
            agent = CamelChatAgent(system_message=sys_msg, model=model)
            with bind_log_context(
                agent_role="triage_judge",
                turn_idx=turn_idx,
                attempt=attempt,
            ):
                response = agent.step(
                    BaseMessage.make_user_message(role_name="User", content=user_prompt)
                )
            raw = str(response.msg.content).strip()
            cleaned = sanitize_llm_json_text(raw)
            parsed = json_repair.loads(cleaned)
            if isinstance(parsed, dict):
                verdict = str(parsed.get("verdict", "PROCEED")).upper()
                reason = str(parsed.get("reason", "")).strip()
            else:
                verdict = "PROCEED"
                reason = ""
            logger.info("[TRIAGE] verdict=%s reason=%s for message: %.80s", verdict, reason, user_message)
            if verdict == "CLARIFY":
                return "CLARIFY", reason or "Triage judge determined clarification is needed."
            return None, ""
        except Exception as e:
            logger.warning("[TRIAGE] Failed, proceeding to Stage 2: %s", e)
            return None, ""

    @staticmethod
    def _tool_call_name(tool_call: Dict[str, Any]) -> str:
        return str(
            tool_call.get("function")
            or tool_call.get("name")
            or tool_call.get("tool")
            or ""
        ).strip()

    @staticmethod
    def _tool_call_args(tool_call: Dict[str, Any]) -> Dict[str, Any]:
        args = tool_call.get("arguments") or tool_call.get("args") or {}
        return args if isinstance(args, dict) else {}

    @staticmethod
    def _tool_call_key(tool_call: Dict[str, Any]) -> Tuple[str, str]:
        name = SimSolver._tool_call_name(tool_call)
        args = SimSolver._tool_call_args(tool_call)
        try:
            rendered_args = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            rendered_args = str(args)
        return name, rendered_args

    def _is_safe_readonly_evidence_call(self, tool_call: Dict[str, Any]) -> bool:
        """Return True only for successful schema-declared read-only calls."""
        if not isinstance(tool_call, dict):
            return False
        name = self._tool_call_name(tool_call)
        if not name:
            return False

        metadata = getattr(self, "_tool_metadata_by_name", {}).get(name)
        if not isinstance(metadata, dict):
            return False
        if str(metadata.get("state_access") or "").lower() != "read":
            return False
        effects = metadata.get("state_effects") or []
        if isinstance(effects, list) and effects:
            return False
        if name == "transfer_to_human_agents":
            return False

        if "result" not in tool_call:
            return False
        if tool_call.get("error") or tool_call.get("exception"):
            return False
        result = tool_call.get("result")
        if isinstance(result, dict) and result.get("error"):
            return False
        return True

    def _project_readonly_evidence_result(self, result: Any) -> Any:
        projector = self.readonly_evidence_result_projector
        if callable(projector):
            try:
                return projector(result, self.readonly_evidence_result_max_chars)
            except Exception:
                logger.debug("Readonly evidence projector failed; falling back to bounded preview", exc_info=True)

        try:
            rendered = json.dumps(result, ensure_ascii=False, default=str)
        except Exception:
            rendered = str(result)
        if len(rendered) <= self.readonly_evidence_result_max_chars:
            return deepcopy(result)
        return rendered[: self.readonly_evidence_result_max_chars].rstrip() + "... [truncated]"

    def _compact_readonly_evidence_call(
        self,
        tool_call: Dict[str, Any],
        *,
        source_attempt: int,
    ) -> Dict[str, Any]:
        return {
            "function": self._tool_call_name(tool_call),
            "arguments": deepcopy(self._tool_call_args(tool_call)),
            "result": self._project_readonly_evidence_result(tool_call.get("result")),
            "inherited_readonly_evidence": True,
            "source_attempt": source_attempt,
        }

    def _safe_readonly_prefix(self, attempt_state: _AttemptState) -> List[Dict[str, Any]]:
        prefix: List[Dict[str, Any]] = []
        for tool_call in attempt_state.tool_calls or []:
            if not self._is_safe_readonly_evidence_call(tool_call):
                break
            prefix.append(
                self._compact_readonly_evidence_call(
                    tool_call,
                    source_attempt=attempt_state.attempt,
                )
            )
        return prefix

    def _build_inherited_readonly_evidence(
        self,
        previous_attempts: List[_AttemptState],
    ) -> List[Dict[str, Any]]:
        """Collect deduplicated safe read-only evidence from prior attempts."""
        if not self.enable_readonly_evidence_branch or not previous_attempts:
            return []

        inherited: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str]] = set()

        def add_call(call: Dict[str, Any]) -> None:
            key = self._tool_call_key(call)
            if not key[0] or key in seen:
                return
            seen.add(key)
            inherited.append(deepcopy(call))

        for attempt_state in previous_attempts:
            for call in getattr(attempt_state, "inherited_tool_calls", []) or []:
                add_call(call)
            for call in self._safe_readonly_prefix(attempt_state):
                add_call(call)

        return inherited

    def _execute_attempt(
        self,
        user_message: str,
        attempt: int,
        previous_attempts: List[_AttemptState],
        checklist: List[Dict[str, Any]],
        base_config: Dict[str, Any],
        turn_idx: int,
        strip_tools: bool = False,
    ) -> Tuple[_AttemptState, List[SimEvent]]:
        """Execute a single attempt within a turn and return (state, events)."""
        attempt_start = time.time()
        attempt_events: List[SimEvent] = []

        current_task_id = TestCaseAdapter.get_id(self.test_case)
        set_task_context(current_task_id)

        session_id = self._create_session_with_config(base_config)
        attempt_events.append(
            SimEvent(
                type="attempt_start",
                ts=time.time(),
                turn_idx=turn_idx,
                attempt=attempt,
                data={"session_id": session_id},
            )
        )

        if self.tool_registry and hasattr(self.tool_registry, "reset_with_fresh_clone"):
            try:
                refreshed = self.tool_registry.reset_with_fresh_clone()
                if refreshed:
                    self._custom_tools = self.tool_registry.get_all_tools()
                    self._agent_exec.tools = self._custom_tools
                    logger.info(f"[TOOLS] Refreshed real tools for attempt {attempt}")
            except Exception as refresh_exc:  # pragma: no cover
                logger.warning(f"[TOOLS] Failed to refresh real tools for attempt {attempt}: {refresh_exc}")

        if self.tool_registry:
            self.tool_registry.bind_session_context(
                session_id=session_id,
                mock_server_url=self.mock_server_url,
                task_id=TestCaseAdapter.get_id(self.test_case),
                buffer_updates=True,
                read_only_sync_policy=self.read_only_sync_policy,
            )
            self.tool_registry.set_session_id(session_id)
            self._custom_tools = self.tool_registry.get_all_tools()
            self._agent_exec.tools = self._custom_tools
            logger.debug(
                "Bound session context for attempt %s: session_id=%s",
                attempt,
                session_id,
            )
        else:
            logger.debug("No tool_registry, skipping session context setup")

        try:
            inherited_tool_calls = self._build_inherited_readonly_evidence(previous_attempts)
            if inherited_tool_calls:
                attempt_events.append(
                    SimEvent(
                        type="inherited_evidence",
                        ts=time.time(),
                        turn_idx=turn_idx,
                        attempt=attempt,
                        data={
                            "tool_calls": deepcopy(inherited_tool_calls),
                            "count": len(inherited_tool_calls),
                        },
                    )
                )

            enhanced_message = MessageBuilder.build_enhanced_message(
                user_message, attempt, previous_attempts, base_config,
                self._history_items,
                inherited_tool_calls=inherited_tool_calls,
            )
            attempt_events.append(
                SimEvent(
                    type="agent_prompt",
                    ts=time.time(),
                    turn_idx=turn_idx,
                    attempt=attempt,
                    data={"prompt": enhanced_message},
                )
            )

            agent = None
            if self.agent_persistence_mode:
                if self._turn_base_agent:
                    agent = AgentExecutor.clone_with_memory(self._turn_base_agent)
                    if agent:
                        logger.info(f"[PERSISTENCE] Cloned agent for attempt {attempt}")
                    else:
                        logger.warning(f"[PERSISTENCE] Clone failed, creating new agent")
                else:
                    logger.info(f"[PERSISTENCE] Creating first agent for attempt {attempt}")

            no_tools_available = not getattr(self._agent_exec, "tools", None)
            if strip_tools or no_tools_available:
                try:
                    from inference.agents.chat_agent import ChatAgent as _ChatAgent
                    toolless_agent = _ChatAgent(
                        model_name=self.model_name,
                        timeout=self.agent_timeout,
                        system_message=self.agent_system_prompt,
                        max_iteration=1,
                        agent_role="simsolver",
                    )
                    toolless_agent.set_log_context(turn_idx=turn_idx, attempt=attempt)
                    _resp = toolless_agent.generate_response(enhanced_message, {})
                    agent_response = getattr(_resp, "raw_response", "") or ""
                    tool_calls: List[Dict[str, Any]] = []
                    tools_count = 0
                    used_agent = toolless_agent
                    agent_metadata = {"agent_success": True}
                except Exception:
                    raise
            else:
                try:
                    result = self._agent_exec.execute(
                        enhanced_message,
                        session_id,
                        turn_count=turn_idx,
                        context=None,
                        existing_agent=agent,
                        attempt=attempt,
                    )
                except Exception:
                    raise
                if result is None:
                    raise RuntimeError("Agent invoke returned None")
                tool_calls, agent_response, tools_count, used_agent, agent_metadata = result


            if self.tool_registry:
                _ctx = self.tool_registry.get_session_context()
                _buffer_count = len(_ctx.get_buffer()) if _ctx else 0
                if _buffer_count > 0:
                    self._flush_real_tool_buffer(session_id)

            agent_metadata = agent_metadata or {}
            agent_success = bool(agent_metadata.get("agent_success", True))
            agent_error = str(agent_metadata.get("agent_error", "") or "").strip()
            if not agent_success:
                mock_config = deepcopy(base_config) if isinstance(base_config, dict) else {}
                failure_reason = agent_error or "Agent execution failed before completion"
                feedback = {
                    "agent_failure": True,
                    "failure_type": agent_metadata.get("failure_type", "agent_error"),
                    "judgment_results": [
                        {
                            "name": "agent_failure",
                            "description": "The previous attempt failed before completing the task.",
                            "reasoning": failure_reason,
                            "status": "failed",
                        }
                    ],
                    "failed_items": [
                        {
                            "name": "agent_failure",
                            "description": "The previous attempt failed before completing the task.",
                            "reasoning": failure_reason,
                            "status": "failed",
                        }
                    ],
                }
                score = 0.0
                attempt_events.append(
                    SimEvent(
                        type="attempt_config",
                        ts=time.time(),
                        turn_idx=turn_idx,
                        attempt=attempt,
                        data={"state": deepcopy(mock_config)},
                    )
                )
                attempt_events.append(
                    SimEvent(
                        type="judge",
                        ts=time.time(),
                        turn_idx=turn_idx,
                        attempt=attempt,
                        data={"score": score, "feedback": deepcopy(feedback)},
                    )
                )

                state = _AttemptState(
                    attempt=attempt,
                    tool_calls=[],
                    score=score,
                    feedback=feedback,
                    session_id=session_id,
                    mock_config=mock_config,
                    execution_time=time.time() - attempt_start,
                    agent_response=agent_error or agent_response,
                    tools_count=tools_count,
                    agent_prompt=enhanced_message,
                    inherited_tool_calls=deepcopy(inherited_tool_calls),
                )
                attempt_events.append(
                    SimEvent(
                        type="attempt_end",
                        ts=time.time(),
                        turn_idx=turn_idx,
                        attempt=attempt,
                        data={
                            "score": state.score,
                            "tool_calls": len(state.tool_calls),
                            "inherited_tool_calls": len(state.inherited_tool_calls),
                            "execution_time": state.execution_time,
                            "session_id": state.session_id,
                        },
                    )
                )
                return state, attempt_events

            if self.fetch_attempt_state:
                mock_config = self.mock_client.get_session_state(session_id)
            else:
                mock_config = deepcopy(base_config) if isinstance(base_config, dict) else {}
            attempt_events.append(
                SimEvent(
                    type="attempt_config",
                    ts=time.time(),
                    turn_idx=turn_idx,
                    attempt=attempt,
                    data={"state": deepcopy(mock_config) if isinstance(mock_config, dict) else {}},
                )
            )

            triage_verdict = None
            triage_reason = ""
            if self.triage_judge_prompt and self._eval.should_generate and checklist and tool_calls:
                triage_verdict, triage_reason = self._run_triage_judge(
                    user_message,
                    tool_calls,
                    turn_idx=turn_idx,
                    attempt=attempt,
                )

            if triage_verdict == "CLARIFY":
                tool_calls = []
                agent_response = ""
                tools_count = 0
                score = 0.0
                feedback = {
                    "triage_verdict": "CLARIFY",
                    "triage_reason": triage_reason,
                }

            attempt_events.append(
                SimEvent(
                    type="agent_response",
                    ts=time.time(),
                    turn_idx=turn_idx,
                    attempt=attempt,
                    data={"response": agent_response},
                )
            )
            attempt_events.append(
                SimEvent(
                    type="tool_calls",
                    ts=time.time(),
                    turn_idx=turn_idx,
                    attempt=attempt,
                    data={"tool_calls": deepcopy(tool_calls), "tools_count": tools_count},
                )
            )

            if triage_verdict == "CLARIFY":
                pass
            elif strip_tools:
                score = 0.0
                feedback = {"triage_verdict": "CLARIFY_RETRY"}
            elif self._eval.should_generate and checklist:
                effective_tool_calls = deepcopy(inherited_tool_calls) + deepcopy(tool_calls or [])
                score, feedback, updated_checklist = self._eval.evaluate_attempt(
                    checklist, mock_config, effective_tool_calls, agent_response,
                    history_items=self._history_items,
                    tool_definitions=self._tool_definitions,
                    memory_store=self._memory_store,
                    agent_system_prompt=self.agent_system_prompt,
                    attempt=attempt,
                    user_message=user_message,
                    policy_cache_scope=f"{TestCaseAdapter.get_id(self.test_case) or self._cache_task_id}:turn={turn_idx}",
                )
                if inherited_tool_calls and isinstance(feedback, dict):
                    feedback["inherited_readonly_evidence_count"] = len(inherited_tool_calls)
                if updated_checklist is not None:
                    self._latest_checklist = updated_checklist
            elif self._eval.should_generate and not checklist:
                score = 1.0
                feedback = {"skipped_judge": "empty_checklist_no_action"}
            else:
                score = 1.0 if tool_calls else 0.0
                feedback = {}
            attempt_events.append(
                SimEvent(
                    type="judge",
                    ts=time.time(),
                    turn_idx=turn_idx,
                    attempt=attempt,
                    data={"score": score, "feedback": deepcopy(feedback)},
                )
            )

            if self.agent_persistence_mode and used_agent:
                if not self._turn_base_agent:
                    self._turn_base_agent = used_agent
                    logger.info(f"[PERSISTENCE] Saved first agent as turn base")
                self._attempt_agents.append((used_agent, score))
                logger.info(f"[PERSISTENCE] Collected agent for attempt {attempt} with score {score:.2f}")

            state = _AttemptState(
                attempt=attempt,
                tool_calls=tool_calls,
                score=score,
                feedback=feedback,
                session_id=session_id,
                mock_config=mock_config if isinstance(mock_config, dict) else {},
                execution_time=time.time() - attempt_start,
                agent_response=agent_response,
                tools_count=tools_count,
                agent_prompt=enhanced_message,
                inherited_tool_calls=deepcopy(inherited_tool_calls),
            )
            attempt_events.append(
                SimEvent(
                    type="attempt_end",
                    ts=time.time(),
                    turn_idx=turn_idx,
                    attempt=attempt,
                    data={
                        "score": state.score,
                        "tool_calls": len(state.tool_calls),
                        "inherited_tool_calls": len(state.inherited_tool_calls),
                        "execution_time": state.execution_time,
                        "session_id": session_id,
                    },
                )
            )
            return state, attempt_events
        finally:
            if self.tool_registry:
                self.tool_registry.clear_session_context()
                logger.debug(f"Cleared bound session context for attempt {attempt}")
            clear_task_context()
    
    def _flush_real_tool_buffer(self, session_id: str) -> None:
        """Flush buffered real tool updates to mock server."""
        if self.tool_registry:
            self.tool_registry.flush_session_buffer()

    def _create_session_with_config(self, config: Dict[str, Any]) -> str:
        """Create mock server session with the provided config."""
        session_id = self.mock_client.create_session(self.test_case)

        if self._openapi_toolkit:
            self._openapi_toolkit.set_session_id(session_id)
            logger.info(f"[TOOLKIT] Set session ID {session_id} on OpenAPI toolkit")

        success = self.mock_client.set_session_state(
            session_id,
            config or {},
        )
        if not success:
            logger.warning(f"Failed to set config for session {session_id}")
        
        return session_id

    def get_latest_checklist(self) -> List[Dict]:
        """Get the latest generated checklist"""
        return self._latest_checklist if hasattr(self, '_latest_checklist') else []

    def _apply_best_attempt_to_state(self, user_message: str, best_attempt: _AttemptState) -> None:
        """Update conversation history + current config using the best attempt."""
        self._history_items.append({"role": "user", "content": user_message})

        for tool_call in best_attempt.tool_calls or []:
            self._history_items.append(
                {
                    "role": "tool_call",
                    "function": tool_call.get("function", "unknown"),
                    "arguments": tool_call.get("arguments", {}) or {},
                }
            )
            if "result" in tool_call:
                self._history_items.append(
                    {
                        "role": "tool_result",
                        "function": tool_call.get("function", "unknown"),
                        "result": tool_call.get("result"),
                    }
                )

        cleaned_assistant = strip_reasoning(best_attempt.agent_response or "")
        if cleaned_assistant:
            self._history_items.append({"role": "assistant", "content": cleaned_assistant})

        self._current_config = deepcopy(best_attempt.mock_config) if isinstance(best_attempt.mock_config, dict) else {}
        self._config_history.append(deepcopy(self._current_config))
    

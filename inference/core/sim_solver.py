#!/usr/bin/env python3
"""
SimSolver - Core simulation solver for mock server based execution
Handles multi-turn conversations with retry mechanisms and state management
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING, Literal
from dataclasses import dataclass, field
from copy import deepcopy

from benchmarks.base.test_case import TestCase
from inference.client_engine import MockServerClient
from utils.legacy.task_feedback import TaskFeedback
from utils.conversation import render_conversation
from utils.conversation_memory import ConversationMemoryStore
from utils.test_case_adapter import TestCaseAdapter
from inference.utils.log_context import TaskContextFilter, set_task_context, clear_task_context

if TYPE_CHECKING:
    from inference.real_tools import ToolRegistry

logger = logging.getLogger(__name__)

# Install task context filter on camel loggers (idempotent)
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
    llm_tokens: Dict[str, Dict[str, int]] = field(default_factory=dict)
    agent_prompt: str = ""


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
                 tool_registry: Optional['ToolRegistry'] = None,
                 openapi_tool_paths: Optional[List[str]] = None,
                 agent_persistence_mode: bool = False,
                 enable_tool_filtering: bool = False,
                 base_checklist_items: Optional[List[str]] = None,
                 checklist_system_prompt: Optional[str] = None,
                 include_agent_response_in_judge: bool = True,
                 enable_tool_result_folding: bool = True,
                 collect_mock_server_usage: bool = True,
                 enable_debug: bool = False,
                 verbose_debug: bool = False):
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
            enable_checklist: Whether to generate detailed checklist (if False, uses simple checklist)
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
            collect_mock_server_usage: Whether to fetch mock-server usage events
                from /get-session-llm-usage (default: True)
        """
        self.test_case = test_case
        self._initial_config = deepcopy(initial_config) if initial_config is not None else None
        self.model_name = model_name
        self.max_retries = max_retries
        self.agent_timeout = agent_timeout
        self.enable_checklist = enable_checklist
        self.target_score = 1.0
        self.mock_server_url = mock_server_url
        self.override_openapi_server = override_openapi_server
        self.agent_max_iteration = agent_max_iteration if agent_max_iteration is not None else 10
        self.agent_summarize_threshold = agent_summarize_threshold
        self.agent_system_prompt = agent_system_prompt
        self.judge_system_prompt = judge_system_prompt
        self.tool_registry = tool_registry
        self.openapi_tool_paths = openapi_tool_paths
        self.agent_persistence_mode = agent_persistence_mode
        self.enable_tool_filtering = bool(enable_tool_filtering)
        self.base_checklist_items = base_checklist_items
        self.checklist_system_prompt = checklist_system_prompt
        self.include_agent_response_in_judge = include_agent_response_in_judge
        self.enable_tool_result_folding = bool(enable_tool_result_folding)
        self.collect_mock_server_usage = bool(collect_mock_server_usage)
        self.enable_debug = bool(enable_debug)
        self.verbose_debug = bool(verbose_debug)
        self._trace_console_enabled = self.enable_debug or self.verbose_debug
        raw_task_id = TestCaseAdapter.get_id(self.test_case) or "unknown_task"
        self._trace_task_id = "".join(
            ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(raw_task_id)
        )

        # Validate that we have tools from at least one source
        if tool_registry is None and openapi_tool_paths is None:
            raise ValueError("Must provide either tool_registry or openapi_tool_paths")
        
        # Initialize mock client (align HTTP timeout with configured agent timeout)
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
        
        # Initialize evaluator if enabled
        if enable_evaluation and max_retries > 0:
            self.evaluator = TaskFeedback(
                model_name=self.model_name,
                system_prompt=judge_system_prompt,
                base_checklist_items=base_checklist_items,
                checklist_system_prompt=checklist_system_prompt
            )
        else:
            self.evaluator = None
            
        # Load custom tools if specified (do this before state init for tool context)
        self._custom_tools = None
        self._openapi_toolkit = None  # Store toolkit instance for session management
        self._tool_definitions: List[Dict[str, Any]] = []  # Simple tool definitions for judge/checklist prompts
        self._tool_catalog: List[Dict[str, str]] = []  # Lightweight name+description list

        # Priority: tool_registry > openapi_tool_paths
        if tool_registry is not None:
            # Use tools from registry (hybrid mode)
            self._custom_tools = tool_registry.get_all_tools()
            logger.info(f"SimSolver using ToolRegistry: "
                       f"{len(tool_registry.get_real_tools())} real tools, "
                       f"{len(tool_registry.get_mock_tools())} mock tools")
        elif openapi_tool_paths:
            # Legacy: load from OpenAPI
            self._custom_tools, self._openapi_toolkit = self._load_tools_from_openapi(openapi_tool_paths)

        # Build lightweight tool definitions for judge/checklist prompts (name + description only).
        try:
            self._tool_definitions = self._build_simple_tool_definitions(self._custom_tools)
            self._tool_catalog = self._build_tool_catalog(self._tool_definitions)
            if self._tool_definitions:
                logger.info(f"[SIMSOLVER] Built {len(self._tool_definitions)} simple tool definitions for judge/checklist")
        except Exception as e:
            logger.warning(f"[SIMSOLVER] Failed to build tool definitions: {e}")

        # State and trace management
        self._initialize_state()

        # Agent management
        self._current_agent = None

        # Agent persistence state (for persistence mode)
        self._turn_base_agent = None  # Base agent for current turn
        self._best_agent = None  # Best agent from all attempts
        self._attempt_agents = []  # List of (agent, score) tuples for current turn
    
        logger.info(f"SimSolver initialized for test {TestCaseAdapter.get_id(test_case)}")

    def _trace_print(self, message: str) -> None:
        if self._trace_console_enabled:
            print(message)

    def _initialize_state(self):
        """Initialize internal state"""
        # Config management: caller-provided initial_config > test_case.metadata['initial_config'] > {}
        config_from_case = {}
        if hasattr(self.test_case, "metadata") and isinstance(self.test_case.metadata, dict):
            config_from_case = self.test_case.metadata.get("initial_config", {}) or {}

        base_config = self._initial_config if self._initial_config is not None else config_from_case
        if base_config is None:
            base_config = {}

        self._current_config: Dict[str, Any] = deepcopy(base_config)
        self._config_history: List[Dict[str, Any]] = [deepcopy(self._current_config)]

        # Conversation history used for prompt composition and judge context.
        # Schema: list of dict events compatible with utils.conversation.render_conversation
        # (roles: user/assistant/tool_call/tool_result).
        self._history_items: List[Dict[str, Any]] = []

        # Conversation-scope memory store for large tool results (judge retrieval via get_memory).
        self._memory_store = ConversationMemoryStore()

        # Trace (primary external data): all events across all turns/attempts.
        self._events: List[SimEvent] = []

        # Turn/attempt tracking
        self._turn_count = 0
        self._total_attempts = 0

        # Store latest checklist (template or evaluated) for external access.
        self._latest_checklist: List[Dict[str, Any]] = []

    # ========== Core Interface ==========

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

    def process(self, user_message: str) -> TurnOutcome:
        """Process a single turn with retry logic.

        - Each attempt runs in a fresh mock-server session.
        - Retries within a turn always start from the same turn-start config.
        - After the turn, SimSolver updates its internal config to the best attempt's config.
        """
        turn_start = time.time()
        turn_idx = self._turn_count
        self._turn_count += 1

        logger.info(f"Processing turn {turn_idx}: {user_message[:100]}...")

        # Turn-local base config: retries must NOT inherit state from failed attempts.
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

        # Set up turn base agent for persistence mode
        if self.agent_persistence_mode:
            if self._best_agent:
                self._turn_base_agent = self._best_agent
                logger.info(f"Using best agent from previous turn as base for turn {turn_idx}")
            elif not self._turn_base_agent:
                logger.info(f"First turn {turn_idx}, will create initial agent")
            self._attempt_agents = []

        # Generate checklist once for all attempts.
        checklist: List[Dict[str, Any]] = []

        if self.evaluator and self._should_generate_checklist():
            checklist = self._generate_checklist(user_message)
        checklist_usage: Dict[str, Any] = {}
        try:
            if self.evaluator and hasattr(self.evaluator, "last_checklist_usage"):
                usage = getattr(self.evaluator, "last_checklist_usage") or {}
                if isinstance(usage, dict) and any(int(usage.get(k, 0) or 0) for k in ("input_tokens", "output_tokens", "total_tokens")):
                    checklist_usage = {
                        "component": "simsolver_checklist",
                        "model": getattr(self.evaluator, "model_name", self.model_name),
                        "input_tokens": int(usage.get("input_tokens", 0) or 0),
                        "output_tokens": int(usage.get("output_tokens", 0) or 0),
                        "total_tokens": int(usage.get("total_tokens", 0) or 0),
                    }
        except Exception:
            checklist_usage = {}
        turn_events.append(
            SimEvent(
                type="checklist",
                ts=time.time(),
                turn_idx=turn_idx,
                data={"items": len(checklist), "checklist": deepcopy(checklist), "llm_usage": deepcopy(checklist_usage)},
            )
        )

        attempts: List[_AttemptState] = []
        for attempt in range(self.max_retries + 1):
            attempt_state, attempt_events = self._execute_attempt(
                user_message=user_message,
                attempt=attempt,
                previous_attempts=attempts,
                checklist=checklist,
                base_config=turn_base_config,
                turn_idx=turn_idx,
            )
            attempts.append(attempt_state)
            self._total_attempts += 1
            turn_events.extend(attempt_events)

            if attempt_state.score >= self.target_score or attempt >= self.max_retries:
                break

            logger.info(
                f"Attempt {attempt} score {attempt_state.score:.2f} < {self.target_score}, retrying..."
            )

        best_attempt_index, best_attempt = max(
            enumerate(attempts),
            key=lambda pair: (pair[1].score, pair[0]),
        )
        logger.info(
            f"Selected attempt {best_attempt_index} with score {best_attempt.score:.2f} from {len(attempts)} attempts"
        )

        if self.agent_persistence_mode and self._attempt_agents:
            _, (self._best_agent, best_score) = max(
                enumerate(self._attempt_agents),
                key=lambda pair: (pair[1][1], pair[0]),
            )
            logger.info(f"Selected best agent with score {best_score:.2f} for next turn")

        # Update internal state for next turn (best attempt only).
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

    # ========== Properties ==========

    @property
    def current_config(self) -> Dict:
        """Current configuration state"""
        return deepcopy(self._current_config)

    @property
    def completed_tasks(self) -> List[str]:
        """List of completed task descriptions (for backward compatibility)"""
        # Extract only user messages for backward compatibility
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

    # ========== Internal Methods ==========

    def _clone_agent_with_memory(self, base_agent) -> Any:
        """Clone an agent with its conversation memory using CAMEL's clone method"""
        try:
            if hasattr(base_agent, '_camel_agent'):
                # Our wrapper agent - clone the CAMEL agent inside
                cloned_camel = base_agent._camel_agent.clone(with_memory=True)
            
                # Create a new wrapper with the cloned CAMEL agent
                from inference.agents.chat_agent import ChatAgent
                cloned_wrapper = ChatAgent(
                    model_name=base_agent.model_name,
                    system_message=getattr(base_agent, 'system_message', None),
                    timeout=getattr(base_agent, 'timeout', 60),
                    agent_role=getattr(base_agent, 'agent_role', 'simsolver'),
                )
            
                # Replace the CAMEL agent with our cloned one
                cloned_wrapper._camel_agent = cloned_camel
            
                logger.info(f"Successfully cloned agent with memory")
                return cloned_wrapper
            else:
                # Direct CAMEL agent
                cloned_agent = base_agent.clone(with_memory=True)
                logger.info("Successfully cloned CAMEL agent directly")
                return cloned_agent
        except Exception as e:
            logger.error(f"Failed to clone agent: {e}")
            return None

    def _execute_attempt(
        self,
        user_message: str,
        attempt: int,
        previous_attempts: List[_AttemptState],
        checklist: List[Dict[str, Any]],
        base_config: Dict[str, Any],
        turn_idx: int,
    ) -> Tuple[_AttemptState, List[SimEvent]]:
        """Execute a single attempt within a turn and return (state, events)."""
        attempt_start = time.time()
        attempt_events: List[SimEvent] = []

        # Set task context for downstream logging (e.g., camel warnings)
        current_task_id = TestCaseAdapter.get_id(self.test_case)
        set_task_context(current_task_id)

        # Create session with turn base config (retries must start from the same base).
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

        # Refresh real tools to a fresh clone of the tau2 toolkit for every attempt (real/hybrid modes).
        if self.tool_registry and hasattr(self.tool_registry, "reset_with_fresh_clone"):
            try:
                refreshed = self.tool_registry.reset_with_fresh_clone()
                if refreshed:
                    # Update custom tools so the agent sees the new wrappers.
                    self._custom_tools = self.tool_registry.get_all_tools()
                    logger.info(f"[TOOLS] Refreshed real tools for attempt {attempt}")
            except Exception as refresh_exc:  # pragma: no cover
                logger.warning(f"[TOOLS] Failed to refresh real tools for attempt {attempt}: {refresh_exc}")

        # Bind explicit session context for real tool wrappers (if using ToolRegistry)
        if self.tool_registry:
            self.tool_registry.bind_session_context(
                session_id=session_id,
                mock_server_url=self.mock_server_url,
                task_id=TestCaseAdapter.get_id(self.test_case),
                buffer_updates=True,
            )
            # Set session_id for mock tools via OpenAPIToolkit
            self.tool_registry.set_session_id(session_id)
            logger.debug(
                "Bound session context for attempt %s: session_id=%s",
                attempt,
                session_id,
            )
        else:
            logger.debug("No tool_registry, skipping session context setup")

        try:
            # Build enhanced message with context and retry info
            enhanced_message = self._build_enhanced_message(
                user_message, attempt, previous_attempts, base_config
            )
            try:
                import os
                prompt_len = len(enhanced_message)
                prompt_preview = enhanced_message[:1200]
                os.makedirs("debug_traces", exist_ok=True)
                prompt_path = os.path.join(
                    "debug_traces",
                    f"{self._trace_task_id}_simsolver_prompt_t{turn_idx}_a{attempt}.txt",
                )
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(enhanced_message)
                self._trace_print(
                    f"[PROMPT] turn={turn_idx} attempt={attempt} len={prompt_len} file={prompt_path}"
                )
                self._trace_print(f"[PROMPT PREVIEW]\n{prompt_preview}\n[END PROMPT PREVIEW]")
            except Exception as e:
                self._trace_print(f"[PROMPT] Failed to write preview: {e}")
            attempt_events.append(
                SimEvent(
                    type="agent_prompt",
                    ts=time.time(),
                    turn_idx=turn_idx,
                    attempt=attempt,
                    data={"prompt": enhanced_message},
                )
            )

            # DEBUG: Print agent user message

            # Create or clone agent based on persistence mode
            agent = None
            if self.agent_persistence_mode:
                if self._turn_base_agent:
                    # Clone from turn base agent (preserves conversation history)
                    agent = self._clone_agent_with_memory(self._turn_base_agent)
                    if agent:
                        logger.info(f"[PERSISTENCE] Cloned agent for attempt {attempt}")
                    else:
                        logger.warning(f"[PERSISTENCE] Clone failed, creating new agent")
                else:
                    # First agent ever - create and save as base
                    logger.info(f"[PERSISTENCE] Creating first agent for attempt {attempt}")

                # Execute with agent and get response details
            self._trace_print(f"[FLOW] Agent invoke turn={turn_idx} attempt={attempt}")
            try:
                result = self._execute_with_agent_ex(
                    enhanced_message,
                    session_id,
                    None,
                    agent,
                    attempt=attempt,
                    task_content=user_message,
                )
            except Exception as e:
                self._trace_print(f"[FLOW] Agent invoke failed: {e}")
                raise
            if result is None:
                self._trace_print("[FLOW] Agent invoke returned None")
                raise RuntimeError("Agent invoke returned None")
            tool_calls, agent_response, tools_count, used_agent, agent_metadata = result
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

            # Flush buffered real tool updates
            if self.tool_registry:
                _ctx = self.tool_registry.get_session_context()
                _buffer_count = len(_ctx.get_buffer()) if _ctx else 0
                if _buffer_count > 0:
                    _flush_t0 = datetime.now()
                    self._trace_print(
                        f"[TIME] SimSolver Flush Buffer START {_flush_t0.strftime('%H:%M:%S')} "
                        f"(count={_buffer_count}, session_id={session_id})"
                    )
                    self._flush_real_tool_buffer(session_id)
                    _flush_t1 = datetime.now()
                    self._trace_print(
                        f"[TIME] SimSolver Flush Buffer END   {_flush_t1.strftime('%H:%M:%S')} "
                        f"(elapsed={(_flush_t1 - _flush_t0).total_seconds():.3f}s, session_id={session_id})"
                    )

            agent_metadata = agent_metadata or {}
            agent_tokens = {
                'input_tokens': agent_metadata.get('input_tokens', 0),
                'output_tokens': agent_metadata.get('output_tokens', 0),
            }
            agent_tokens['total_tokens'] = agent_metadata.get(
                'total_tokens',
                (agent_tokens['input_tokens'] or 0) + (agent_tokens['output_tokens'] or 0)
            )
            attempt_tokens: Dict[str, Dict[str, int]] = {}
            if any(agent_tokens.values()):
                attempt_tokens['agent'] = agent_tokens

            llm_requests: List[Dict[str, Any]] = []
            if any(agent_tokens.values()):
                llm_requests.append(
                    {
                        "component": "simsolver_agent",
                        "model": self.model_name,
                        "input_tokens": int(agent_tokens.get("input_tokens", 0) or 0),
                        "output_tokens": int(agent_tokens.get("output_tokens", 0) or 0),
                        "total_tokens": int(agent_tokens.get("total_tokens", 0) or 0),
                    }
                )

            agent_success = bool(agent_metadata.get("agent_success", True))
            agent_error = str(agent_metadata.get("agent_error", "") or "").strip()
            if not agent_success:
                # Treat timed-out/failed agent attempts as invalid and avoid reading
                # potentially mutated session state from late background tool calls.
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
                    llm_tokens=attempt_tokens,
                    agent_prompt=enhanced_message,
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
                            "execution_time": state.execution_time,
                            "session_id": state.session_id,
                            "llm_tokens": deepcopy(state.llm_tokens),
                            "llm_requests": deepcopy(llm_requests),
                            "mock_server_llm_usage": {},
                        },
                    )
                )
                return state, attempt_events

            # Get updated config from mock server with timing
            _cfg_t0 = time.time()
            mock_config = self.mock_client.get_session_state(session_id)
            attempt_events.append(
                SimEvent(
                    type="attempt_config",
                    ts=time.time(),
                    turn_idx=turn_idx,
                    attempt=attempt,
                    data={"state": deepcopy(mock_config) if isinstance(mock_config, dict) else {}},
                )
            )
            _cfg_t1 = time.time()
            logger.info(f"[TIME] SimSolver Config Fetch END   {datetime.fromtimestamp(_cfg_t1).strftime('%H:%M:%S')} (elapsed={( _cfg_t1 - _cfg_t0 ): .3f}s, session_id={session_id})")

            # Evaluate if enabled.
            #
            # NOTE: TaskFeedback.generate_checklist() may intentionally return []
            # for "no actionable request" turns (e.g., gratitude/closing/small talk).
            # In that case we should skip judge AND treat the turn as successful,
            # otherwise SimSolver may retry unnecessarily with score=0.0.
            if self.evaluator and checklist:
                _judge_t0 = time.time()
                score, feedback = self._evaluate_attempt(
                    checklist,
                    mock_config,
                    tool_calls,
                    agent_response,
                    attempt=attempt,
                    user_message=user_message,
                )
                _judge_t1 = time.time()
                logger.info(
                    f"[TIME] SimSolver Judge Wrap END {datetime.fromtimestamp(_judge_t1).strftime('%H:%M:%S')} "
                    f"(elapsed={( _judge_t1 - _judge_t0 ): .3f}s, session_id={session_id})"
                )
            elif self.evaluator and not checklist:
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

            judge_tokens = {}
            if self.evaluator and hasattr(self.evaluator, 'last_judge_usage'):
                judge_usage = getattr(self.evaluator, 'last_judge_usage') or {}
                judge_tokens = {
                    'input_tokens': judge_usage.get('input_tokens', 0),
                    'output_tokens': judge_usage.get('output_tokens', 0),
                }
                judge_tokens['total_tokens'] = judge_usage.get(
                    'total_tokens',
                    (judge_tokens['input_tokens'] or 0) + (judge_tokens['output_tokens'] or 0)
                )
                if any(judge_tokens.values()):
                    attempt_tokens['judge'] = judge_tokens
                    llm_requests.append(
                        {
                            "component": "simsolver_judge",
                            "model": getattr(self.evaluator, "model_name", self.model_name),
                            "input_tokens": int(judge_tokens.get("input_tokens", 0) or 0),
                            "output_tokens": int(judge_tokens.get("output_tokens", 0) or 0),
                            "total_tokens": int(judge_tokens.get("total_tokens", 0) or 0),
                        }
                    )

            mock_server_llm_usage: Dict[str, Any] = {}
            if self.collect_mock_server_usage:
                try:
                    mock_server_llm_usage = self.mock_client.get_session_llm_usage(
                        session_id,
                        include_events=True,
                        limit=10000,
                        since_id=0,
                    )
                except Exception:
                    mock_server_llm_usage = {}

            # In persistence mode, track agent and score
            if self.agent_persistence_mode and used_agent:
                # Save first agent as turn base
                if not self._turn_base_agent:
                    self._turn_base_agent = used_agent
                    logger.info(f"[PERSISTENCE] Saved first agent as turn base")
                # Collect agent with score for later selection
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
                llm_tokens=attempt_tokens,
                agent_prompt=enhanced_message,
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
                        "execution_time": state.execution_time,
                        "session_id": state.session_id,
                        "llm_tokens": deepcopy(state.llm_tokens),
                        "llm_requests": deepcopy(llm_requests),
                        "mock_server_llm_usage": deepcopy(mock_server_llm_usage),
                    },
                )
            )
            return state, attempt_events
        finally:
            if self.tool_registry:
                self.tool_registry.clear_session_context()
                logger.debug(f"Cleared bound session context for attempt {attempt}")
            # Clear task context to avoid leakage across attempts/tasks
            clear_task_context()
    
    def _flush_real_tool_buffer(self, session_id: str) -> None:
        """Flush buffered real tool updates to mock server."""
        if self.tool_registry:
            self.tool_registry.flush_session_buffer()

    def _create_session_with_config(self, config: Dict[str, Any]) -> str:
        """Create mock server session with the provided config."""
        session_id = self.mock_client.create_session(self.test_case)

        # Set session ID on OpenAPI toolkit if we have custom tools
        if self._openapi_toolkit:
            self._openapi_toolkit.set_session_id(session_id)
            logger.info(f"[TOOLKIT DEBUG] Set session ID {session_id} on OpenAPI toolkit")

        # Always set config (allow empty dict {}).
        runtime_state = (config or {}).get("runtime_state") if isinstance(config, dict) else None
        bootstrap_mode = "skip" if isinstance(runtime_state, dict) and runtime_state else "auto"
        success = self.mock_client.set_session_state(
            session_id,
            config or {},
            bootstrap_mode=bootstrap_mode,
        )
        if not success:
            logger.warning(f"Failed to set config for session {session_id}")
        
        return session_id

    def _build_enhanced_message(
        self,
        user_message: str,
        attempt: int,
        previous_attempts: List[_AttemptState],
        current_state: Dict[str, Any],
    ) -> str:
        """Build the agent-facing message: Conversation History + [Current Task] (+ retry context).

        Checklist/judge will still only see raw user_message elsewhere.
        """
        parts = []

        # Include conversation history if available
        if self._history_items:
            parts.append("=== Previous Conversation History ===")
            history_view = render_conversation(
                self._history_items,
                max_items=30,
                include_tool_calls=True,
                include_results=True,
                truncate_assistant=None,
                truncate_result=None,
            )
            for item in history_view:
                role = item.get('role', '')
                if role == 'user':
                    parts.append(f"User: {item.get('content', '')}")
                elif role == 'assistant':
                    parts.append(f"Assistant: {item.get('content', '')}")
                elif role == 'tool_call':
                    # Support both 'function'/'arguments' and 'name'/'args' keys for backward compatibility
                    func_name = item.get('function') or item.get('name', 'unknown')
                    args = item.get('arguments') or item.get('args', {})
                    args_str = json.dumps(args, indent=2) if isinstance(args, dict) else str(args)
                    parts.append(f"Tool Call: {func_name}")
                    parts.append(f"Arguments: {args_str}")
                    # Handle merged format: tool_call includes result
                    if 'result' in item:
                        result = item.get('result')
                        result_str = json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)
                        # Truncate long results
                        # if len(result_str) > 500:
                        #     result_str = result_str[:500] + "... [truncated]"
                        parts.append(f"Result: {result_str}")
                elif role == 'tool_result':
                    # Legacy separate tool_result format (backward compatibility)
                    func_name = item.get('function') or item.get('name', 'unknown')
                    result = item.get('result', {})
                    result_str = json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)
                    # if len(result_str) > 500:
                    #     result_str = result_str[:500] + "... [truncated]"
                    parts.append(f"Tool Result ({func_name}): {result_str}")

            parts.append("")

        # Always provide authoritative state so the agent can ground path/state decisions.
        parts.append("=== Authoritative Current State (Turn-Start) ===")
        parts.append(self._render_authoritative_state(current_state))
        parts.append(
            "Important: this state is authoritative for this attempt. "
            "If history text conflicts with this state, trust this state."
        )
        parts.append("")

        # Always append the raw user message with a clear marker
        parts.append("[Current Task]")
        parts.append(user_message)

        # On retries, include compact retry guidance from last attempt
        if attempt > 0 and previous_attempts:
            parts.append("")
            parts.append(self._build_retry_context(attempt, previous_attempts))

        final_message = '\n'.join(parts)

        return final_message
    
    def _build_retry_context(self, attempt: int, previous_attempts: List[_AttemptState]) -> str:
        """Build retry context from all previous attempts, with last-attempt detail."""
        last_attempt = previous_attempts[-1]
        retry_memory = self._collect_retry_memory(previous_attempts)
        lines = ["Problematic solution from previous attempt:"]

        lines.append("- Tool calls:")
        if last_attempt.tool_calls:
            for i, tc in enumerate(last_attempt.tool_calls, 1):
                func = tc.get('function', 'unknown')
                args = tc.get('arguments', {}) or {}
                if isinstance(args, dict) and 'requestBody' in args and isinstance(args.get('requestBody'), dict):
                    args = args.get('requestBody') or {}
                result = tc.get('result', {})

                if args:
                    arg_items = list(args.items())
                    args_str = ', '.join(
                        f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}"
                        for k, v in arg_items[:3]
                    )
                    if len(arg_items) > 3:
                        args_str += ", ..."
                    call_repr = f"{func}({args_str})"
                else:
                    call_repr = f"{func}()"

                lines.append(f"  {i}. {call_repr}")

                if result:
                    result_str = str(result)
                    if len(result_str) > 150:
                        result_str = result_str[:150] + "... [truncated]"
                    lines.append(f"     result: {result_str}")
                else:
                    lines.append("     result: (no result)")
        else:
            lines.append("   (none)")

        lines.append("- Response:")
        response_text = (last_attempt.agent_response or "").strip()
        lines.append(f"   {response_text if response_text else '(empty)'}")

        lines.append("")
        lines.append("Judge findings (aggregated across previous attempts):")
        failed_items = retry_memory.get("failed_items", [])
        if failed_items:
            for idx, item in enumerate(failed_items[:5], start=1):
                desc = item.get("description", "Unknown requirement")
                reason = item.get("reasoning", "")
                from_attempt = item.get("attempt")
                suffix = f" (from attempt {from_attempt})" if from_attempt is not None else ""
                lines.append(f"  {idx}. {desc}{suffix}")
                if reason:
                    lines.append(f"     - reason: {reason}")
        else:
            lines.append("  1. No explicit failed checklist items were returned.")

        lines.append("")
        lines.append("Known invalid calls to avoid (aggregated):")
        invalid_calls = retry_memory.get("invalid_calls", [])
        if invalid_calls:
            for idx, item in enumerate(invalid_calls[:8], start=1):
                signature = item.get("signature", "<unknown_call>")
                error = item.get("error", "")
                from_attempt = item.get("attempt")
                suffix = f" (from attempt {from_attempt})" if from_attempt is not None else ""
                lines.append(f"  {idx}. {signature}{suffix}")
                if error:
                    lines.append(f"     - error: {error}")
        else:
            lines.append("  1. No invalid calls recorded.")

        lines.append("")
        lines.append("Retry needed:")
        lines.append("  1. This retry runs in a fresh session but starts from the same turn-start state shown above.")
        lines.append("  2. Previous turns' completed effects are included in the authoritative current state.")
        lines.append("  3. Resolve all unresolved judge findings in this retry.")
        lines.append("  4. Do not repeat known-invalid calls from prior attempts.")
        lines.append("")
        lines.append("Please 改正上述问题，解决当前的task。")
        return '\n'.join(lines)

    def _collect_retry_memory(self, previous_attempts: List[_AttemptState]) -> Dict[str, List[Dict[str, Any]]]:
        """Aggregate failed checklist items and invalid tool calls across attempts."""
        failed_items: List[Dict[str, Any]] = []
        invalid_calls: List[Dict[str, Any]] = []
        seen_failed: set[Tuple[str, str]] = set()
        seen_invalid: set[Tuple[str, str]] = set()

        for attempt_state in previous_attempts or []:
            feedback = attempt_state.feedback if isinstance(attempt_state.feedback, dict) else {}
            raw_failed = feedback.get("failed_items", [])
            if isinstance(raw_failed, list):
                for item in raw_failed:
                    if not isinstance(item, dict):
                        continue
                    desc = str(item.get("description", "") or "").strip()
                    reason = str(item.get("reasoning", "") or "").strip()
                    if not desc:
                        continue
                    key = (desc, reason)
                    if key in seen_failed:
                        continue
                    seen_failed.add(key)
                    failed_items.append(
                        {
                            "description": desc,
                            "reasoning": reason,
                            "attempt": attempt_state.attempt,
                        }
                    )

            for call in attempt_state.tool_calls or []:
                if not isinstance(call, dict):
                    continue
                err = self._extract_error_text(call.get("result"))
                if not err:
                    continue
                signature = self._format_tool_call_signature(call)
                key = (signature, err)
                if key in seen_invalid:
                    continue
                seen_invalid.add(key)
                invalid_calls.append(
                    {
                        "signature": signature,
                        "error": err,
                        "attempt": attempt_state.attempt,
                    }
                )

        return {"failed_items": failed_items, "invalid_calls": invalid_calls}

    def _format_tool_call_signature(self, call: Dict[str, Any]) -> str:
        """Format a compact, stable signature for a tool call."""
        func = str(call.get("function", "unknown"))
        args = call.get("arguments", {})
        if isinstance(args, dict) and "requestBody" in args and isinstance(args.get("requestBody"), dict):
            args = args.get("requestBody") or {}
        if not isinstance(args, dict) or not args:
            return f"{func}()"

        parts: List[str] = []
        for idx, (k, v) in enumerate(args.items()):
            if idx >= 4:
                parts.append("...")
                break
            if isinstance(v, str):
                parts.append(f"{k}='{v}'")
            else:
                parts.append(f"{k}={v}")
        return f"{func}({', '.join(parts)})"

    def _extract_error_text(self, result: Any) -> Optional[str]:
        """Best-effort extraction of tool-call error text."""
        if isinstance(result, dict):
            if result.get("error"):
                return str(result.get("error"))
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

    def _render_authoritative_state(self, current_state: Dict[str, Any], max_chars: int = 20000) -> str:
        """Render authoritative state in deterministic JSON with bounded size."""
        state = current_state if isinstance(current_state, dict) else {}
        try:
            text = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True)
        except Exception:
            text = str(state)

        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]
        return f"{truncated}\n... [truncated authoritative state]"
    
    def _execute_with_agent_ex(self, message: str, session_id: str,
                              context: Optional[Dict] = None,
                              existing_agent: Optional[Any] = None,
                              attempt: Optional[int] = None,
                              task_content: Optional[str] = None) -> Tuple[List[Dict], str, int, Any, Dict[str, Any]]:
        """Execute message with agent and return tool calls, response content, tools count, agent used, and metadata"""
        from inference.agents.chat_agent import ChatAgent

        # Use existing agent or create new one
        if existing_agent:
            agent = existing_agent
            logger.info("[AGENT DEBUG] Using provided agent")
        else:
            agent_timeout = self.agent_timeout if self.agent_timeout is not None else 60
            agent = ChatAgent(
                model_name=self.model_name,
                timeout=agent_timeout,
                system_message=self.agent_system_prompt,  # Use custom prompt if provided
                max_iteration=self.agent_max_iteration,
                summarize_threshold=self.agent_summarize_threshold,
                agent_role="simsolver",
            )
        if hasattr(agent, "_camel_agent") and agent._camel_agent is not None:
            agent._camel_agent.max_iteration = self.agent_max_iteration
            agent._camel_agent.summarize_threshold = self.agent_summarize_threshold
        try:
            import os
            system_prompt = getattr(agent, "system_message", None) or ""
            sys_len = len(system_prompt)
            sys_preview = system_prompt[:1200] if system_prompt else "<EMPTY>"
            os.makedirs("debug_traces", exist_ok=True)
            sys_path = os.path.join(
                "debug_traces",
                f"{self._trace_task_id}_simsolver_system_prompt_t{self._turn_count}_a{attempt if attempt is not None else 'na'}.txt",
            )
            with open(sys_path, "w", encoding="utf-8") as f:
                f.write(system_prompt)
            self._trace_print(f"[SYSTEM PROMPT] len={sys_len} file={sys_path}")
            self._trace_print(f"[SYSTEM PROMPT PREVIEW]\n{sys_preview}\n[END SYSTEM PROMPT PREVIEW]")
        except Exception as e:
            self._trace_print(f"[SYSTEM PROMPT] Failed to write preview: {e}")
        
        # Determine which tools to use (explicitly provided via ToolRegistry or OpenAPI).
        tools = self._custom_tools or []
        if not tools:
            raise RuntimeError("SimSolver has no tools configured (expected ToolRegistry or openapi_tool_paths).")

        tools = self._filter_tools_for_task(tools, task_content or message)

        tools_count = len(tools) if tools else 0
        logger.info(f"[AGENT DEBUG] Turn {self._turn_count}: Available tools count: {tools_count}")
        logger.info(f"[AGENT DEBUG] Session ID: {session_id}")
        logger.info(f"[AGENT DEBUG] Current config: {self._current_config}")

        # Print tool names for debugging
        if tools:
            tool_names = []
            for tool in tools:
                if hasattr(tool, '__name__'):
                    tool_names.append(tool.__name__)
                elif hasattr(tool, 'get_function_name'):
                    tool_names.append(tool.get_function_name())
                elif isinstance(tool, dict) and 'name' in tool:
                    tool_names.append(tool['name'])
            if tool_names:
                preview_names = tool_names[:20]
                self._trace_print(f"[TOOLS] count={tools_count} sample={preview_names}")
        else:
            logger.warning("[AGENT DEBUG] No tools available!")

        # Only set tools if agent doesn't already have them (e.g., not cloned)
        # Cloned agents already have tools from the clone operation
        need_tools = True
        if hasattr(agent, '_camel_agent') and hasattr(agent._camel_agent, 'tool_dict'):
            # Check if agent already has tools
            existing_tools_count = len(agent._camel_agent.tool_dict)
            if existing_tools_count > 0:
                logger.info(f"[AGENT DEBUG] Agent already has {existing_tools_count} tools (likely from cloning)")
                need_tools = False

        if need_tools and hasattr(agent, 'set_tools') and tools:
            agent.set_tools(tools)
            logger.info(f"[AGENT DEBUG] Tools set on agent successfully")

            # Verify tools were actually set
            if hasattr(agent, '_camel_agent') and hasattr(agent._camel_agent, 'tool_dict'):
                actual_tools_count = len(agent._camel_agent.tool_dict)
        elif need_tools:
            logger.warning(f"[AGENT DEBUG] Failed to set tools on agent")
        
        # Generate response
        try:
            if context is None:
                context = {}
            _at0 = datetime.now()
            attempt_label = attempt if attempt is not None else "?"
            self._trace_print(
                f"[TIME] SimSolver Agent LLM START {_at0.strftime('%H:%M:%S')} "
                f"(model={self.model_name}, turn={self._turn_count}, attempt={attempt_label})"
            )
            response = agent.generate_response(message, context)
            _at1 = datetime.now()
            usage = self._extract_usage_from_response(response)
            agent_success = bool(getattr(response, "success", True))
            agent_error = getattr(response, "error_message", None)
            self._trace_print(
                "[TIME] SimSolver Agent LLM END   "
                f"{_at1.strftime('%H:%M:%S')} (elapsed={( _at1 - _at0 ).total_seconds():.3f}s, "
                f"model={self.model_name}, turn={self._turn_count}, attempt={attempt_label}, "
                f"tokens_in={usage['input_tokens']}, tokens_out={usage['output_tokens']}, "
                f"tokens_total={usage['total_tokens']})"
            )
            logger.info("[AGENT DEBUG] Response generated")
            if not agent_success:
                logger.warning(
                    "Agent execution reported failure (turn=%s attempt=%s): %s",
                    self._turn_count,
                    attempt_label,
                    agent_error,
                )

            # Extract response content
            agent_response = ""
            if hasattr(response, 'raw_response'):
                agent_response = response.raw_response if response.raw_response else ""
                logger.info(f"[AGENT DEBUG] Response content: {agent_response[:200] if agent_response else 'None'}")
                resp_display = agent_response if agent_response else "<EMPTY>"
                self._trace_print(f"[SIMSOLVER AGENT RESPONSE] {resp_display}")

            # Extract tool calls
            tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []
            logger.info(f"[AGENT DEBUG] Tool calls count: {len(tool_calls)}")

            # DEBUG: Print raw tool calls structure
            import json
            logger.info(f"[AGENT DEBUG] Raw tool_calls type: {type(tool_calls)}")
            if tool_calls:
                logger.info(f"[AGENT DEBUG] Raw tool_calls content: {json.dumps(tool_calls, indent=2, default=str)}")

            # Log details for tool calls if present
            # Normalize tool calls to have function/arguments/results consistently
            tool_calls = [self._normalize_tool_call(tc) for tc in tool_calls if tc is not None]

            try:
                for i, tc in enumerate(tool_calls, 1):
                    logger.info(f"[AGENT DEBUG] ToolCall {i} type: {type(tc)}")
                    logger.info(f"[AGENT DEBUG] ToolCall {i} raw: {tc}")

                    func = tc.get('function', 'unknown') if isinstance(tc, dict) else str(tc)
                    args = tc.get('arguments', {}) if isinstance(tc, dict) else {}
                    res = tc.get('result', None) if isinstance(tc, dict) else None

                    logger.info(f"[AGENT DEBUG] ToolCall {i}: function={func}, args={args}")
                    logger.info(f"[AGENT DEBUG] ToolCall {i} args type: {type(args)}")

                    if res is not None:
                        res_preview = str(res)
                        if len(res_preview) > 300:
                            res_preview = res_preview[:300] + "..."
                        logger.info(f"[AGENT DEBUG] ToolCall {i} result: {res_preview}")
            except Exception as e:
                logger.warning(f"[AGENT DEBUG] Failed to log detailed tool calls: {e}")
                import traceback
                logger.warning(f"[AGENT DEBUG] Traceback: {traceback.format_exc()}")
            
            metadata = {}
            if hasattr(response, 'metadata') and isinstance(response.metadata, dict):
                metadata = response.metadata
            metadata["agent_success"] = agent_success
            if agent_error:
                metadata["agent_error"] = str(agent_error)
                if "failure_type" not in metadata:
                    metadata["failure_type"] = (
                        "timeout" if "timed out" in str(agent_error).lower() else "agent_error"
                    )

            return tool_calls, agent_response, tools_count, agent, metadata
        except Exception as e:
            logger.error(f"Failed to execute with agent: {e}")
            return [], str(e), tools_count, agent, {
                "agent_success": False,
                "agent_error": str(e),
                "failure_type": "exception",
            }

    def _normalize_tool_call(self, tc: Any) -> Dict[str, Any]:
        """Normalize tool call structure to ensure function name and arguments are present."""
        if not isinstance(tc, dict):
            # Try to extract attributes if tc is an object
            possible = {}
            for key in ['function', 'name', 'tool', 'tool_name']:
                val = getattr(tc, key, None)
                if val:
                    possible['function'] = val if not isinstance(val, dict) else val.get('name') or val.get('function')
                    break
            args = getattr(tc, 'arguments', None) or getattr(tc, 'args', None)
            if isinstance(args, str):
                try:
                    import json as _json
                    args = _json.loads(args)
                except Exception:
                    pass
            if args is None:
                args = {}
            if not possible:
                logger.warning(f"Unrecognized tool_call structure (non-dict): {tc}")
                return {'function': 'unknown', 'arguments': args, 'raw': tc}
            return {'function': possible.get('function', 'unknown'), 'arguments': args}

        normalized = dict(tc)

        # Flatten OpenAI-style function object if present
        func_field = normalized.get('function')
        if isinstance(func_field, dict):
            normalized['function'] = func_field.get('name', func_field.get('function', 'unknown'))
            if 'arguments' in func_field and not normalized.get('arguments'):
                normalized['arguments'] = func_field.get('arguments')
        elif not func_field and normalized.get('name'):
            normalized['function'] = normalized.get('name')

        # Ensure arguments is a dict (parse JSON if it's a string)
        args = normalized.get('arguments')
        if isinstance(args, str):
            try:
                import json as _json
                normalized['arguments'] = _json.loads(args)
            except Exception:
                pass
        elif args is None:
            normalized['arguments'] = {}

        if not normalized.get('function'):
            logger.warning(f"Missing function name in tool_call dict: {normalized}")
            normalized['function'] = 'unknown'

        return normalized
    
    def _execute_with_agent(self, message: str, session_id: str, 
                           context: Optional[Dict] = None,
                           attempt: Optional[int] = None) -> Tuple[List[Dict], str, int]:
        """Execute message with agent and return tool calls, response content, and tools count (legacy)"""
        # Call new method but return only first 3 values for backward compatibility
        tool_calls, agent_response, tools_count, _, _ = self._execute_with_agent_ex(
            message, session_id, context, None, attempt=attempt
        )
        return tool_calls, agent_response, tools_count

    def _extract_usage_from_response(self, response: Any) -> Dict[str, int]:
        """Extract token usage from agent response, best-effort."""
        usage_info: Dict[str, Any] = {}
        if hasattr(response, "metadata") and isinstance(response.metadata, dict):
            # Our AgentResponse stores token counts at the top-level of metadata.
            if any(k in response.metadata for k in ("input_tokens", "output_tokens", "total_tokens")):
                usage_info = dict(response.metadata)
            else:
                usage_info = response.metadata.get("usage", {}) or {}
        if not usage_info and hasattr(response, "info") and isinstance(response.info, dict):
            usage_info = response.info.get("usage", {}) or {}

        input_tokens = usage_info.get("prompt_tokens") or usage_info.get("input_tokens") or 0
        output_tokens = usage_info.get("completion_tokens") or usage_info.get("output_tokens") or 0
        total_tokens = usage_info.get("total_tokens")
        if total_tokens is None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)

        return {
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "total_tokens": int(total_tokens or 0),
        }
    
    def _generate_checklist(self, user_message: str) -> List[Dict]:
        """Generate checklist for task evaluation"""
        try:
            if self.enable_checklist:
                # Build previous_tasks as high-level user/assistant dialogue only
                # (no tool calls/results). Judge still receives full conversation
                # with tools via conversation_history.
                previous_tasks: List[str] = []
                try:
                    text_history = render_conversation(
                        self._history_items,
                        text_only=True,
                        include_tool_calls=False,
                        include_results=False,
                        truncate_assistant=None,
                    )
                    for item in text_history:
                        content = item.get('content')
                        if isinstance(content, str) and content.strip():
                            previous_tasks.append(content)
                except Exception as e:
                    logger.warning(f"[CHECKLIST] Failed to build previous_tasks from completed_tasks: {e}")
                    previous_tasks = []

                # Generate detailed checklist
                checklist = self.evaluator.generate_checklist(
                    task=user_message,
                    initial_config=None,  # Don't pass config to save tokens
                    previous_tasks=previous_tasks,
                    conversation_history=self._history_items,
                    tool_definitions=self._tool_catalog or None,
                    policy_text=self._extract_policy_text(),
                )
                # If the generator decides there's no actionable request, it can return an empty checklist.
                logger.info(f"Generated checklist with {len(checklist)} items")
                try:
                    for i, item in enumerate(checklist[:5], 1):
                        desc = item.get('description', str(item)) if isinstance(item, dict) else str(item)
                        logger.info(f"[CHECKLIST] Item {i}: {desc}")
                except Exception:
                    pass
            else:
                # Use deterministic fixed checklist when generation is disabled.
                # If caller provided base_checklist_items, treat them as the full
                # checklist to avoid any extra LLM checklist call in single-turn runs.
                if isinstance(self.base_checklist_items, list) and self.base_checklist_items:
                    checklist = [{"description": str(item)} for item in self.base_checklist_items if str(item).strip()]
                    logger.info(
                        "Using fixed checklist from base_checklist_items (checklist generation disabled): %d items",
                        len(checklist),
                    )
                else:
                    checklist = [{"description": f"Verify the operation was executed: {user_message[:100]}"}]
                    logger.info("Using simple checklist (checklist generation disabled)")
            
            # Store the latest checklist for external access
            self._latest_checklist = checklist
            return checklist
        except Exception as e:
            logger.error(f"Failed to generate checklist: {e}")
            self._latest_checklist = []
            return []

    def _evaluate_attempt(
        self,
        checklist: List[Dict],
        config: Dict,
        tool_calls: List[Dict],
        agent_response: str,
        attempt: int = None,
        user_message: Optional[str] = None,
    ) -> Tuple[float, Dict]:
        """Evaluate attempt using checklist

        Args:
            checklist: Checklist items to verify
            config: Current config state
            tool_calls: Tool calls made in this attempt
            agent_response: Agent's text response
            attempt: Attempt number for logging (optional)
        """
        try:
            rendered_history = render_conversation(
                self._history_items,
                include_tool_calls=True,
                include_results=True,
                truncate_assistant=None,
                truncate_result=None,
            )
            if self.enable_tool_result_folding:
                judge_tool_calls = self._fold_tool_calls_for_judge(tool_calls)
                judge_history = self._fold_history_for_judge(rendered_history)
                judge_memory_store: Optional[ConversationMemoryStore] = self._memory_store
            else:
                judge_tool_calls = tool_calls or []
                judge_history = rendered_history
                judge_memory_store = None

            policy_text = self._extract_policy_text()
            involved_tool_definitions = self._select_involved_tool_definitions(tool_calls)

            # Judge execution with complete tool_calls/history. Agent response is optional.
            judge_agent_response = agent_response if self.include_agent_response_in_judge else None
            judgment_results, critical, score = self.evaluator.judge_execution(
                checklist=checklist,
                current_config=config,
                tool_calls=judge_tool_calls,
                tool_definitions=involved_tool_definitions or None,
                agent_response=judge_agent_response,
                conversation_history=judge_history,
                attempt=attempt
                ,
                memory_store=judge_memory_store,
                policy_text=policy_text,
                user_request=user_message,
            )

            # Keep complete feedback for retry context.
            # Note: Storage may filter detailed fields in some adapters.
            feedback = {
                'judgment_results': judgment_results,
                'failed_items': [item for item in judgment_results if item.get('status') == 'failed']
            }
            try:
                logger.info(
                    f"[JUDGE] Score: {score:.2f}, Failed items: "
                    f"{len(feedback.get('failed_items', [])) if isinstance(feedback.get('failed_items', []), list) else 0}"
                )
                if isinstance(judgment_results, list):
                    for i, jr in enumerate(judgment_results[:5], 1):
                        status = jr.get('status', 'unknown')
                        desc = jr.get('description', '')
                        logger.info(f"[JUDGE] Item {i}: status={status}, desc={desc[:120]}")
            except Exception:
                pass

            # Store judgment_results as part of checklist info if available
            if judgment_results and isinstance(judgment_results, list):
                # judgment_results contains the evaluated checklist items
                self._latest_checklist = judgment_results

            return score, feedback

        except Exception as e:
            logger.error(f"Failed to evaluate attempt: {e}")
            failure_item = {
                "name": "judge_evaluation_failed",
                "description": "Judge evaluation failed; this attempt must be retried.",
                "reasoning": f"Judge exception: {str(e)}",
                "status": "failed",
            }
            feedback = {
                "judge_failure": True,
                "error": str(e),
                "judgment_results": [failure_item],
                "failed_items": [failure_item],
            }
            # Conservative fallback: never pass an attempt when judge execution fails.
            return 0.0, feedback

    def _fold_history_for_judge(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        folded: List[Dict[str, Any]] = []
        for item in history or []:
            if not isinstance(item, dict):
                continue
            new_item = dict(item)
            role = new_item.get("role")
            if role == "tool_result" and "result" in new_item:
                function_name = str(new_item.get("function") or new_item.get("name") or "unknown")
                new_item["result"] = self._memory_store.fold_result(
                    function_name=function_name,
                    result=new_item.get("result"),
                )
            if role == "tool_call" and "result" in new_item:
                function_name = str(new_item.get("function") or new_item.get("name") or "unknown")
                new_item["result"] = self._memory_store.fold_result(
                    function_name=function_name,
                    result=new_item.get("result"),
                )
            folded.append(new_item)
        return folded

    def _extract_policy_text(self) -> Optional[str]:
        """Extract policy text from the agent system prompt, if present."""
        if not self.agent_system_prompt:
            return None
        prompt = self.agent_system_prompt
        if "<policy>" in prompt and "</policy>" in prompt:
            try:
                return prompt.split("<policy>", 1)[1].split("</policy>", 1)[0].strip()
            except Exception:
                return prompt.strip()
        return None

    def _fold_tool_calls_for_judge(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        folded: List[Dict[str, Any]] = []
        for tc in tool_calls or []:
            if not isinstance(tc, dict):
                continue
            new_tc = dict(tc)
            if "result" in new_tc:
                function_name = str(new_tc.get("function") or "unknown")
                new_tc["result"] = self._memory_store.fold_result(
                    function_name=function_name,
                    result=new_tc.get("result"),
                )
            folded.append(new_tc)
        return folded
    
    def _should_generate_checklist(self) -> bool:
        """Determine if checklist should be generated"""
        return self.evaluator is not None and self.max_retries > 0
    
    def get_latest_checklist(self) -> List[Dict]:
        """Get the latest generated checklist"""
        return self._latest_checklist if hasattr(self, '_latest_checklist') else []

    def _build_simple_tool_definitions(self, tools: Optional[List[Any]]) -> List[Dict[str, Any]]:
        """
        Build tool definitions (name + description + full input schema) for judge/checklist.

        Includes argument schema when available (flattening requestBody where needed). Supports
        CAMEL FunctionTool/RealToolWrapper and
        tau2 Tool-like objects.
        """
        definitions: List[Dict[str, Any]] = []
        if not tools:
            return definitions

        def _extract_parameters(schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Extract a compact parameters schema from an OpenAI/CAMEL style schema."""
            if not isinstance(schema, dict):
                return None

            # OpenAI function-style schema
            params = schema.get("parameters") or schema.get("function", {}).get("parameters")
            if isinstance(params, dict):
                return params

            # OpenAPI-style requestBody
            request_body = (
                schema.get("requestBody")
                or schema.get("function", {}).get("requestBody")
            )
            if isinstance(request_body, dict):
                content = request_body.get("content", {})
                if isinstance(content, dict):
                    app_json = content.get("application/json", {}) or content.get("application/json; charset=utf-8", {})
                    if isinstance(app_json, dict):
                        body_schema = app_json.get("schema")
                        if isinstance(body_schema, dict):
                            return {
                                "type": "object",
                                "properties": body_schema.get("properties", {}),
                                "required": body_schema.get("required", []),
                                "description": body_schema.get("description", "request body"),
                            }
            return None

        for tool in tools:
            try:
                name: Optional[str] = None
                description: Optional[str] = None
                parameters: Optional[Dict[str, Any]] = None

                # CAMEL FunctionTool / RealToolWrapper: has openai_tool_schema
                schema = getattr(tool, "openai_tool_schema", None)
                if isinstance(schema, dict):
                    name = schema.get("name") or schema.get("function", {}).get("name")
                    description = schema.get("description") or schema.get("function", {}).get("description")
                    parameters = _extract_parameters(schema)

                # tau2 Tool objects: typically expose name/description attributes
                if not name and hasattr(tool, "name"):
                    name = getattr(tool, "name", None)
                if not description and hasattr(tool, "description"):
                    description = getattr(tool, "description", None)

                # Fallback: use function __name__ or repr if name still missing
                if not name:
                    name = getattr(getattr(tool, "func", None), "__name__", None) or getattr(tool, "__name__", None) or repr(tool)

                definitions.append(
                    {
                        "name": str(name),
                        "description": str(description) if description is not None else "",
                        **({"parameters": parameters} if parameters else {}),
                    }
                )
            except Exception as e:
                logger.debug(f"[SIMSOLVER] Skipped tool when building definitions: {e}")

        return definitions

    def _filter_tools_for_task(
        self, tools: List[Any], task_content: str
    ) -> List[Any]:
        """Remove definitely irrelevant tools when tool filtering is enabled."""
        if not self.enable_tool_filtering or not tools or not task_content:
            return tools

        from benchmarks.bfcl.tool_filtering import filter_definitely_irrelevant_tools

        filter_payload = self._build_simple_tool_definitions(tools)
        if not filter_payload:
            return tools

        try:
            irrelevant_tool_names = filter_definitely_irrelevant_tools(
                task=task_content,
                tools=filter_payload,
                model_name=self.model_name,
                timeout=min(self.agent_timeout or 60, 60),
            )
        except Exception as exc:
            logger.warning("Tool filtering failed; keeping all tools: %s", exc)
            return tools

        if not irrelevant_tool_names:
            logger.info("Tool filtering kept all %d tools for task", len(filter_payload))
            return tools

        irrelevant_set = set(irrelevant_tool_names)
        filtered_tools = []
        removed_names: List[str] = []
        for tool, tool_def in zip(tools, filter_payload):
            tool_name = str(tool_def.get("name", "") or "")
            if tool_name and tool_name in irrelevant_set:
                removed_names.append(tool_name)
                continue
            filtered_tools.append(tool)

        logger.info(
            "Tool filtering removed %d/%d tools: %s",
            len(removed_names),
            len(filter_payload),
            removed_names,
        )
        return filtered_tools

    @staticmethod
    def _build_tool_catalog(tool_definitions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Build lightweight tool catalog (name + description only)."""
        catalog: List[Dict[str, str]] = []
        for td in tool_definitions or []:
            if not isinstance(td, dict):
                continue
            name = td.get("name")
            if not isinstance(name, str) or not name:
                continue
            catalog.append(
                {
                    "name": name,
                    "description": str(td.get("description", "") or ""),
                }
            )
        return catalog

    @staticmethod
    def _tool_name_variants(raw_name: str) -> List[str]:
        """Create normalized name variants for matching tool calls to definitions."""
        name = str(raw_name or "").strip()
        if not name:
            return []
        base = name.split("/")[-1]
        variants = {name, base, name.lower(), base.lower(), base.replace(".", "_"), base.replace("_", ".")}
        return [v for v in variants if v]

    def _select_involved_tool_definitions(
        self, tool_calls: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Return detailed schemas only for tools involved in this attempt."""
        if not self._tool_definitions:
            return []

        used_variants: set[str] = set()
        for tc in tool_calls or []:
            if not isinstance(tc, dict):
                continue
            call_name = tc.get("function") or tc.get("name") or ""
            used_variants.update(self._tool_name_variants(str(call_name)))

        if not used_variants:
            return []

        selected: List[Dict[str, Any]] = []
        for td in self._tool_definitions:
            if not isinstance(td, dict):
                continue
            def_name = str(td.get("name") or "").strip()
            if not def_name:
                continue
            def_variants = set(self._tool_name_variants(def_name))
            if def_variants.intersection(used_variants):
                selected.append(td)
                continue
            def_lower = def_name.lower()
            if any(def_lower.endswith(v.lower()) or v.lower().endswith(def_lower) for v in used_variants):
                selected.append(td)

        return selected

    def _apply_best_attempt_to_state(self, user_message: str, best_attempt: _AttemptState) -> None:
        """Update conversation history + current config using the best attempt."""
        # Conversation history (best-attempt only)
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

        if best_attempt.agent_response:
            self._history_items.append({"role": "assistant", "content": best_attempt.agent_response})

        # Config update for next turn
        self._current_config = deepcopy(best_attempt.mock_config) if isinstance(best_attempt.mock_config, dict) else {}
        self._config_history.append(deepcopy(self._current_config))
    
    def _load_tools_from_openapi(self, openapi_paths: List[str]) -> Tuple[List[Any], Any]:
        """
        Load tools from OpenAPI specification files
        
        Args:
            openapi_paths: List of paths to OpenAPI spec files
            
        Returns:
            Tuple of (List of FunctionTool objects, OpenAPIToolkit instance)
        """
        import json
        from camel.toolkits import OpenAPIToolkit, FunctionTool

        all_tools = []
        openapi_toolkit = OpenAPIToolkit()

        # Set server URL (session ID will be set later when created)
        if self.override_openapi_server and self.mock_server_url:
            openapi_toolkit.set_override_server_url(self.mock_server_url)

        for path in openapi_paths:
            try:
                logger.info(f"Loading OpenAPI spec from {path}")
                with open(path, 'r') as f:
                    openapi_json = json.load(f)
                
                api_name = openapi_json.get("info", {}).get("title", "Unknown API")

                # Generate functions and schemas from OpenAPI spec
                toolkit = openapi_toolkit.generate_openapi_funcs(api_name, openapi_json)
                schemas = openapi_toolkit.openapi_spec_to_openai_schemas(api_name, openapi_json)
                
                # Create FunctionTool objects
                tools = [FunctionTool(func=func, openai_tool_schema=schema)
                        for func, schema in zip(toolkit, schemas)]

                # Fix requestBody parameter wrapping issue
                from utils.openapi_toolkit_fix import fix_openapi_tools
                tools = fix_openapi_tools(tools)

                # Wrap mock tools to flush explicitly bound real-tool buffer before execution.
                def create_flush_wrapper(func):
                    def wrapper(*args, **kwargs):
                        # Flush any pending real tool updates before executing mock tool.
                        if self.tool_registry:
                            self.tool_registry.flush_session_buffer()
                        return func(*args, **kwargs)
                    return wrapper

                for tool in tools:
                    # Wrap the underlying function
                    original_func = tool.func
                    tool.func = create_flush_wrapper(original_func)

                all_tools.extend(tools)
                
                logger.info(f"Loaded {len(tools)} tools from {api_name}")
                
            except Exception as e:
                logger.error(f"Failed to load OpenAPI spec from {path}: {e}")
                continue
        
        logger.info(f"Total {len(all_tools)} tools loaded from {len(openapi_paths)} OpenAPI specs")
        return all_tools, openapi_toolkit
    
    def dump_events_json(self, file_path: str) -> bool:
        """Dump the full event trace to a JSON file. Returns True on success."""
        import json

        try:
            payload = [
                {
                    "type": e.type,
                    "ts": e.ts,
                    "turn_idx": e.turn_idx,
                    "attempt": e.attempt,
                    "data": e.data,
                }
                for e in self.get_events()
            ]
            with open(file_path, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to dump events JSON: {e}")
            return False

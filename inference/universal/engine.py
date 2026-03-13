
import logging
import os
import time
from typing import Any, Dict, List, Optional

from benchmarks.bfcl.utils import derive_single_turn_schema_name
from benchmarks.base.benchmark import BaseBenchmark
from benchmarks.base.test_case import TestCase
from benchmarks.base.execution_result import ExecutionResult
from .config import InferenceConfig
from inference.evaluation import RuleBasedTaskEvaluator, LLMTaskEvaluator, TaskEvaluationResult
# Import both executors for compatibility
try:
    from inference.multi_turn_executor_refactored import MultiTurnExecutor as MultiTurnExecutorRefactored
    USE_REFACTORED_EXECUTOR = True
except ImportError:
    USE_REFACTORED_EXECUTOR = False
from inference.multi_turn_executor import MultiTurnExecutor
from utils.test_case_adapter import TestCaseAdapter

logger = logging.getLogger(__name__)


class UniversalInferenceEngine:
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.benchmark: Optional[BaseBenchmark] = None
        
        self._model = None
        self._mock_client = None
        self._real_executor = None
        
        if self.config.max_retries > 0:
            self._task_evaluator = self._init_task_evaluator()
        else:
            self._task_evaluator = None
            logger.info("Task evaluator disabled (max_retries=0)")
        
        logger.info(f"Universal Inference Engine initialized with model: {config.model_name}")

    def _trace_print(self, message: str) -> None:
        if bool(getattr(self.config, "enable_debug", False)):
            print(message)
    
    def _init_task_evaluator(self):
        evaluator_type = self.config.task_evaluator_type
        
        if evaluator_type == "rule_based":
            mode = self.config.rule_evaluator_mode
            evaluator = RuleBasedTaskEvaluator(mode=mode)
            logger.info(f"Initialized RuleBasedTaskEvaluator (mode: {mode})")
        else:  # Default to llm
            evaluator = LLMTaskEvaluator(
                model_name=self.config.task_evaluator_model_name,
                enable_checklist=self.config.enable_checklist,
                enable_trace=bool(
                    self.config.enable_debug
                ),
            )
            logger.info(f"Initialized LLMTaskEvaluator (model: {self.config.task_evaluator_model_name}, checklist: {self.config.enable_checklist})")
        
        return evaluator
    
    def set_benchmark(self, benchmark: BaseBenchmark):
        self.benchmark = benchmark
        logger.info(f"Benchmark set to: {benchmark.name} v{benchmark.version}")
    
    def execute_test_case(self, test_case: TestCase) -> ExecutionResult:
        if not self.benchmark:
            raise ValueError("No benchmark set. Call set_benchmark() first.")
        
        test_id = TestCaseAdapter.get_id(test_case)
        logger.info(f"Executing test case: {test_id}")
        
        execution_result = ExecutionResult(test_id=test_id)
        execution_result.metadata.update({
            'benchmark': self.benchmark.name,
            'model': self.config.model_name,
            'config': self.config.to_dict()
        })
        execution_result.llm_costs = {}
        
        try:
            if self._is_multi_turn_test(test_case):
                # Multi-turn execution
                # Default path uses the main MultiTurnExecutor (SimSolver + real task-agent flow).
                if getattr(self.config, 'use_sim_solver', True):
                    logger.info(
                        "Creating MultiTurnExecutor (SimSolver-guided real execution) "
                        "with config: enable_real_execution=%s",
                        self.config.enable_real_execution,
                    )
                    executor = MultiTurnExecutor(self.config)
                elif USE_REFACTORED_EXECUTOR:
                    logger.info(
                        "Creating refactored MultiTurnExecutor (fallback) "
                        "with config: enable_real_execution=%s",
                        self.config.enable_real_execution,
                    )
                    executor = MultiTurnExecutorRefactored(self.config)
                else:
                    logger.info(
                        "Creating MultiTurnExecutor (fallback to main executor) "
                        "with config: enable_real_execution=%s",
                        self.config.enable_real_execution,
                    )
                    executor = MultiTurnExecutor(self.config)
                multi_turn_result = executor.execute_multi_turn(test_case)
                
                execution_result.success = multi_turn_result.success
                execution_result.outputs = []
                
                for turn in multi_turn_result.turns:
                    turn_metadata = {}
                    
                    if hasattr(turn, 'all_attempts') and turn.all_attempts:
                        for idx, attempt_data in enumerate(turn.all_attempts):
                            turn_metadata[f'attempt_{idx}_tool_calls'] = attempt_data.get('tool_calls', [])
                            
                            # Add mock_config for each attempt
                            if 'mock_config' in attempt_data:
                                turn_metadata[f'attempt_{idx}_mock_config'] = attempt_data.get('mock_config', {})
                                logger.debug(f"[DEBUG ENGINE] Added mock_config for attempt {idx}")
                            
                            # Add agent_response and tools_count if available
                            if 'agent_response' in attempt_data:
                                turn_metadata[f'attempt_{idx}_agent_response'] = attempt_data.get('agent_response', '')
                                logger.debug(f"[DEBUG ENGINE] Added agent_response for attempt {idx}: {len(attempt_data.get('agent_response', ''))} chars")
                            if 'tools_count' in attempt_data:
                                turn_metadata[f'attempt_{idx}_tools_count'] = attempt_data.get('tools_count', 0)
                                logger.debug(f"[DEBUG ENGINE] Added tools_count for attempt {idx}: {attempt_data.get('tools_count', 0)}")
                            
                            evaluation_data = {
                                'passed': attempt_data.get('judgment', {}).get('passed', False),
                                'score': attempt_data.get('judgment', {}).get('score', 0),
                                'retry_reason': '',
                                'checklist': turn.checklist if idx == 0 and hasattr(turn, 'checklist') else []
                            }
                            # Include feedback if available
                            judgment = attempt_data.get('judgment', {})
                            if 'feedback' in judgment:
                                evaluation_data['feedback'] = judgment['feedback']
                            
                            turn_metadata[f'attempt_{idx}_evaluation'] = evaluation_data
                    else:
                        turn_metadata['attempt_0_tool_calls'] = turn.real_tool_calls if turn.real_tool_calls else turn.mock_tool_calls
                        turn_metadata['attempt_0_evaluation'] = {
                            'passed': turn.success,
                            'score': turn.judge_score,
                            'retry_reason': turn.error if turn.error else '',
                            'checklist': turn.checklist if hasattr(turn, 'checklist') else []
                        }
                    
                    execution_result.add_output({
                        'type': 'turn_result',
                        'turn_index': turn.turn_idx,
                        'content': turn_metadata,
                        'success': turn.success,
                        'tool_calls': turn.real_tool_calls if turn.real_tool_calls else turn.mock_tool_calls,
                        'mock_tool_calls': turn.mock_tool_calls,
                        'real_tool_calls': turn.real_tool_calls,
                        'mock_config': turn.mock_config,
                        'calibrated_config': turn.calibrated_config,
                        'task_agent_response': getattr(turn, 'task_agent_response', ''),
                        'judge_score': turn.judge_score,
                        'execution_time': turn.execution_time,
                        'error': turn.error
                    })
                
                execution_result.metadata['total_time'] = multi_turn_result.total_time
                execution_result.metadata['final_score'] = multi_turn_result.judge_score
                execution_result.metadata['config_history'] = getattr(multi_turn_result, 'config_history', [])
                execution_result.metadata['total_turns'] = getattr(multi_turn_result, 'total_turns', 0)
                execution_result.metadata['executed_turns'] = getattr(multi_turn_result, 'executed_turns', 0)
                execution_result.metadata['max_turns_applied'] = getattr(multi_turn_result, 'max_turns_applied', None)
                execution_result.metadata['process_trace'] = getattr(multi_turn_result, 'process_trace', {})
                execution_result.metadata['real_execution_enabled'] = self.config.enable_real_execution
                
                if hasattr(multi_turn_result, 'total_llm_costs') and multi_turn_result.total_llm_costs:
                    execution_result.llm_costs = multi_turn_result.total_llm_costs
                    logger.info(f"Multi-turn LLM costs: {multi_turn_result.total_llm_costs}")
                
                all_tool_calls = []
                for turn in multi_turn_result.turns:
                    turn_calls = turn.real_tool_calls if turn.real_tool_calls else turn.mock_tool_calls
                    if turn_calls:
                        all_tool_calls.extend(turn_calls)
                execution_result.tool_calls = all_tool_calls
                
                if USE_REFACTORED_EXECUTOR and getattr(self.config, 'use_sim_solver', True):
                    if hasattr(multi_turn_result, 'examples'):
                        execution_result.metadata['extracted_examples'] = [
                            {
                                'task_id': ex.task_id,
                                'success': ex.success,
                                'description': ex.description,
                                'total_attempts': ex.total_attempts,
                                'turns': len(ex.turns)
                            } for ex in multi_turn_result.examples
                        ]
                        logger.info(f"Extracted {len(multi_turn_result.examples)} examples from execution")
                
                result = execution_result
            else:
                if getattr(self.config, 'use_sim_solver', True):
                    result = self._execute_single_turn_with_simsolver(test_case, execution_result)
                else:
                    result = self._execute_single_turn(test_case, execution_result)
            
            disable_mock_server_metrics = bool(
                (self.config.custom_config or {}).get("disable_mock_server_metrics", False)
            )

            if (
                not disable_mock_server_metrics
                and hasattr(self, '_current_session_id')
                and self._current_session_id
            ):
                try:
                    if not hasattr(self, '_mock_client'):
                        from inference.client_engine import MockServerClient
                        self._mock_client = MockServerClient(self.config.mock_server_url)
                    
                    mock_costs = self._mock_client.get_session_cost(self._current_session_id)
                    
                    mock_server_categories = [
                        'mock_server_response_generation',
                        'mock_server_config_update', 
                        'mock_server_request_validation'
                    ]
                    
                    for category in mock_server_categories:
                        execution_result.llm_costs[category] = 0.0
                    
                    if mock_costs and mock_costs.get('costs'):
                        for category, cost in mock_costs['costs'].items():
                            execution_result.llm_costs[f'mock_server_{category}'] = cost
                        
                        logger.info(f"Retrieved mock server costs for session {self._current_session_id}: ${mock_costs.get('total', 0):.6f}")
                except Exception as e:
                    logger.warning(f"Failed to retrieve mock server costs: {e}")
            
            if result.end_time is None:
                result.mark_completed(success=result.success)
            return result
            
        except Exception as e:
            if "UNTESTED_TASK:" in str(e):
                logger.error(f"Test {test_id} marked as untested: {e}")
                raise
            error_msg = f"Execution failed: {str(e)}"
            logger.error(f"Test {test_id} failed: {error_msg}")
            execution_result.mark_completed(success=False, error=error_msg)
            return execution_result
    
    def _is_multi_turn_test(self, test_case: TestCase) -> bool:
        metadata = getattr(test_case, "metadata", {}) or {}
        test_type = metadata.get("type", "single_turn")
        if test_type == "multi_turn":
            return True

        # BFCL v4 data may label multi-turn items as single_turn in metadata.type.
        # Fall back to category / id prefix checks for robust routing.
        category = str(metadata.get("category", "") or "")
        if category.startswith("multi_turn"):
            return True

        test_id = str(TestCaseAdapter.get_id(test_case) or "")
        return test_id.startswith("multi_turn")
    
    def _execute_single_turn(self, test_case: TestCase, execution_result: ExecutionResult) -> ExecutionResult:
        test_id = TestCaseAdapter.get_id(test_case)
        logger.debug(f"Executing single-turn test: {test_id}")
        
        agent = None
        session_id = None
        
        previous_evaluation = None
        previous_tool_calls = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt == 0 or agent is None:
                    session_id = self._get_or_create_session(test_case)
                    agent = self._create_agent_with_tools_for_session(test_case, session_id)
                    logger.debug(f"Created fresh agent for attempt {attempt}")
                else:
                    logger.debug(f"Creating fresh agent and session for retry attempt {attempt}")
                    session_id = self._get_or_create_session(test_case)
                    agent = self._create_agent_with_retry_context(
                        test_case, session_id, 
                        previous_evaluation=previous_evaluation,
                        previous_tool_calls=previous_tool_calls,
                        attempt=attempt
                    )
                
                response = self._generate_response_with_agent_direct(test_case, agent, session_id, attempt)
                
                tool_calls = self._parse_tool_calls(response)
                
                if tool_calls:
                    execution_results = self._execute_tools(tool_calls, test_case)
                    for result in execution_results:
                        execution_result.add_output(result)
                
                tool_calls = getattr(self, '_current_tool_calls', [])
                
                execution_result.add_output({
                    'type': 'response',
                    'content': response,
                    'attempt': attempt,
                    'tool_calls': tool_calls,
                    'llm_cost': getattr(self, '_current_llm_cost', 0.0),
                    'input_tokens': getattr(self, '_current_input_tokens', 0),
                    'output_tokens': getattr(self, '_current_output_tokens', 0)
                })
                
                if tool_calls:
                    execution_result.set_tool_calls(tool_calls)
                    logger.info(f"Stored {len(tool_calls)} tool calls in execution result")
                
                task_description = self._extract_task_description(test_case)
                current_state = self._extract_current_state(execution_result)
                
                # Filter tool definitions to only those actually used
                filtered_tool_definitions = []
                if tool_calls and hasattr(self, '_current_tool_definitions'):
                    # Extract and normalize tool names from tool calls
                    used_tool_names = set()
                    for tc in tool_calls:
                        tool_name = tc.get('name', tc.get('function', ''))
                        if tool_name:
                            # Extract base name (handle paths like /multiple_7_0/tool_name)
                            base_name = tool_name.split('/')[-1] if '/' in tool_name else tool_name
                            # Normalize by replacing underscores with dots for comparison
                            normalized_name = base_name.replace('_', '.')
                            used_tool_names.add(base_name)
                            used_tool_names.add(normalized_name)
                            used_tool_names.add(tool_name)  # Also keep original
                            logger.debug(f"Tool call name variants: {base_name}, {normalized_name}, {tool_name}")
                    
                    # Filter definitions to only include used tools
                    for td in self._current_tool_definitions:
                        tool_def_name = td.get('function', {}).get('name', '')
                        if tool_def_name:
                            # Check multiple formats for matching
                            normalized_def = tool_def_name.replace('_', '.')
                            underscore_def = tool_def_name.replace('.', '_')
                            
                            # Also check if any used tool name is a suffix of the tool definition
                            # This handles cases like multiple_7_0post_wildlife_population_assess_growth
                            # matching wildlife_population_assess_growth
                            is_suffix_match = any(
                                tool_def_name.endswith(used_name) or 
                                normalized_def.endswith(used_name) or
                                underscore_def.endswith(used_name)
                                for used_name in used_tool_names
                            )
                            
                            logger.debug(f"Checking tool def: {tool_def_name} (normalized: {normalized_def}, underscore: {underscore_def})")
                            # Match if any format matches
                            if (tool_def_name in used_tool_names or 
                                normalized_def in used_tool_names or
                                underscore_def in used_tool_names or
                                is_suffix_match):
                                filtered_tool_definitions.append(td)
                                logger.debug(f"Matched tool definition: {tool_def_name}")
                    
                    logger.debug(f"Used tool names: {used_tool_names}")
                    logger.debug(f"Filtered tool definitions from {len(self._current_tool_definitions)} to {len(filtered_tool_definitions)}")
                
                if self.config.max_retries > 0 and self._task_evaluator:
                    evaluation = self._task_evaluator.evaluate_completion(
                        task=task_description,
                        tool_calls=tool_calls,
                        current_state=current_state,
                        initial_state=None,  # Remove initial_state to save tokens
                        tool_definitions=filtered_tool_definitions
                    )
                else:
                    evaluation = TaskEvaluationResult(
                        score=1.0,
                        passed=True,
                        retry_reason=None,
                        checklist=[],
                        judgment=[],
                        metadata={'checklist_cost': 0.0, 'judge_cost': 0.0}
                    )
                    logger.debug("Evaluation skipped (max_retries=0), auto-pass")
                
                evaluation_dict = evaluation.to_dict()
                evaluation_dict['evaluator_type'] = self.config.task_evaluator_type if self.config.max_retries > 0 else 'disabled'
                execution_result.metadata[f'attempt_{attempt}_evaluation'] = evaluation_dict
                
                if not hasattr(execution_result, 'llm_costs'):
                    execution_result.llm_costs = {}
                
                if 'inference' not in execution_result.llm_costs:
                    execution_result.llm_costs['inference'] = 0.0
                execution_result.llm_costs['inference'] += getattr(self, '_current_llm_cost', 0.0)
                
                if 'evaluation_checklist' not in execution_result.llm_costs:
                    execution_result.llm_costs['evaluation_checklist'] = 0.0
                if 'evaluation_judge' not in execution_result.llm_costs:
                    execution_result.llm_costs['evaluation_judge'] = 0.0
                
                if 'metadata' in evaluation_dict:
                    eval_metadata = evaluation_dict['metadata']
                    
                    if 'checklist_cost' in eval_metadata:
                        execution_result.llm_costs['evaluation_checklist'] += eval_metadata['checklist_cost']
                    
                    if 'judge_cost' in eval_metadata:
                        execution_result.llm_costs['evaluation_judge'] += eval_metadata['judge_cost']
                
                execution_result.metadata[f'attempt_{attempt}_tool_calls'] = tool_calls
                
                if evaluation.passed or evaluation.score >= self.config.target_score:
                    logger.info(f"✅ Task completed (score: {evaluation.score:.2f})")
                    execution_result.success = True
                    execution_result.metadata['final_score'] = evaluation.score
                    # Print tool calls in evaluation format for single-turn
                    formatted_calls = self._format_tool_calls_for_evaluation(tool_calls)
                    self._trace_print(f"{test_id}[1/1]: {formatted_calls}")
                    break
                elif attempt < self.config.max_retries:
                    logger.info(f"🔄 Retry {attempt+1}/{self.config.max_retries}: {evaluation.retry_reason}")
                    previous_evaluation = evaluation
                    previous_tool_calls = tool_calls
                else:
                    logger.warning(f"❌ Max retries reached (final score: {evaluation.score:.2f})")
                    execution_result.success = False
                    # Print tool calls in evaluation format even if failed
                    formatted_calls = self._format_tool_calls_for_evaluation(tool_calls)
                    self._trace_print(f"{test_id}[1/1]: {formatted_calls}")
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt == self.config.max_retries:
                    raise
        
        execution_result.mark_completed(success=execution_result.success if hasattr(execution_result, 'success') else True)
        return execution_result

    def _execute_single_turn_with_simsolver(self, test_case: TestCase, execution_result: ExecutionResult) -> ExecutionResult:
        """Execute single-turn test with SimSolver for unified judge/retry behavior."""
        test_id = TestCaseAdapter.get_id(test_case)
        logger.debug(f"Executing single-turn test with SimSolver: {test_id}")

        # Extract question text and task-specific system messages from BFCL data.
        # BFCL entries may have multi-role messages: [{"role":"system",...},{"role":"user",...}]
        bfcl_test_case = test_case.content if hasattr(test_case, 'content') and test_case.content else test_case
        questions = TestCaseAdapter.get_questions(bfcl_test_case)
        question_text = ""
        task_system_messages: list = []
        if questions:
            first = questions[0]
            messages = first if isinstance(first, list) else [first]
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    if role == 'system' and content:
                        task_system_messages.append(content)
                    elif role == 'user' and content:
                        question_text = content
                    elif not role and content:
                        question_text = content
                elif isinstance(msg, str):
                    question_text = msg
            # Fallback: if no user message found, use first message content
            if not question_text and messages:
                first_msg = messages[0]
                if isinstance(first_msg, dict):
                    question_text = first_msg.get('content', str(first_msg))
                else:
                    question_text = str(first_msg)

        # Build per-task agent system prompt: base config prompt + task system messages
        effective_agent_system_prompt = self.config.agent_system_prompt or ""
        if task_system_messages:
            parts = []
            if effective_agent_system_prompt:
                parts.append(effective_agent_system_prompt)
            parts.extend(task_system_messages)
            effective_agent_system_prompt = "\n\n".join(parts)

        # Resolve OpenAPI tool paths (or use configured overrides)
        openapi_tool_paths = self.config.openapi_tool_paths or self._get_openapi_schemas(test_case)

        # Optional checklist prompt/base items from custom config
        base_checklist_items = None
        checklist_system_prompt = getattr(self.config, "checklist_system_prompt", None)
        if isinstance(self.config.custom_config, dict):
            base_checklist_items = self.config.custom_config.get('base_checklist_items')
            if checklist_system_prompt is None:
                checklist_system_prompt = self.config.custom_config.get('checklist_system_prompt')

        try:
            from inference.core import SimSolver

            solver = SimSolver(
                test_case=test_case,
                model_name=self.config.model_name,
                max_retries=self.config.max_retries,
                agent_timeout=getattr(self.config, "timeout", None),
                mock_server_url=self.config.mock_server_url,
                override_openapi_server=getattr(self.config, "override_openapi_server", True),
                enable_evaluation=self.config.max_retries > 0,
                enable_checklist=self.config.enable_checklist,
                agent_system_prompt=effective_agent_system_prompt,
                judge_system_prompt=self.config.judge_system_prompt,
                openapi_tool_paths=openapi_tool_paths,
                agent_persistence_mode=self.config.agent_persistence_mode,
                base_checklist_items=base_checklist_items,
                checklist_system_prompt=checklist_system_prompt,
                enable_tool_result_folding=getattr(self.config, "enable_tool_result_folding", True),
                collect_mock_server_usage=not bool(
                    (self.config.custom_config or {}).get("disable_mock_server_metrics", False)
                ),
                enable_debug=bool(self.config.enable_debug),
                verbose_debug=False,
            )

            turn = solver.process(question_text)

            # Track session id for mock server cost retrieval (best attempt)
            attempt_sessions = {}
            for ev in turn.events:
                if ev.type == "attempt_start" and ev.attempt is not None:
                    attempt_sessions[int(ev.attempt)] = ev.data.get("session_id")
            self._current_session_id = attempt_sessions.get(turn.best_attempt)

            # Reconstruct attempts and checklist from events
            attempts_by_idx: Dict[int, Dict[str, Any]] = {}
            checklist = []
            for ev in turn.events:
                if ev.type == "checklist":
                    raw = ev.data.get("checklist", []) or []
                    checklist = [{"description": it.get("description", "")} for it in raw if isinstance(it, dict)]
                    continue
                if ev.attempt is None:
                    continue
                idx = int(ev.attempt)
                attempts_by_idx.setdefault(
                    idx,
                    {
                        "attempt": idx,
                        "tool_calls": [],
                        "mock_config": {},
                        "agent_response": "",
                        "tools_count": 0,
                        "judgment": {"score": 0.0, "passed": False, "feedback": {}},
                    },
                )
                rec = attempts_by_idx[idx]
                if ev.type == "agent_response":
                    rec["agent_response"] = ev.data.get("response", "") or ""
                elif ev.type == "tool_calls":
                    rec["tool_calls"] = ev.data.get("tool_calls", []) or []
                    rec["tools_count"] = int(ev.data.get("tools_count") or 0)
                elif ev.type == "attempt_config":
                    rec["mock_config"] = ev.data.get("config", {}) or {}
                elif ev.type == "judge":
                    score = float(ev.data.get("score") or 0.0)
                    feedback = ev.data.get("feedback", {}) or {}
                    rec["judgment"] = {"score": score, "passed": score >= 1.0, "feedback": feedback}

            all_attempts = [attempts_by_idx[k] for k in sorted(attempts_by_idx)]

            # Populate metadata similar to legacy single-turn path
            for attempt in all_attempts:
                idx = attempt["attempt"]
                execution_result.metadata[f"attempt_{idx}_tool_calls"] = attempt.get("tool_calls", [])
                execution_result.metadata[f"attempt_{idx}_mock_config"] = attempt.get("mock_config", {})
                execution_result.metadata[f"attempt_{idx}_agent_response"] = attempt.get("agent_response", "")
                execution_result.metadata[f"attempt_{idx}_tools_count"] = attempt.get("tools_count", 0)

                judgment = attempt.get("judgment", {}) or {}
                feedback = judgment.get("feedback", {}) or {}
                judgment_list = feedback.get("judgment_results", []) if isinstance(feedback, dict) else []
                critical = feedback.get("critical_responses", []) if isinstance(feedback, dict) else []
                retry_reason = critical[0] if critical else ""

                evaluation_data = {
                    "passed": judgment.get("passed", False),
                    "score": judgment.get("score", 0.0),
                    "retry_reason": retry_reason,
                    "checklist": checklist if idx == 0 else [],
                    "judgment": judgment_list,
                }
                if feedback:
                    evaluation_data["feedback"] = feedback
                execution_result.metadata[f"attempt_{idx}_evaluation"] = evaluation_data

            filtered_final_tool_calls = self._filter_failed_tool_calls_for_single_turn(
                turn.final_tool_calls
            )

            # Final result fields
            execution_result.success = turn.score >= 1.0
            execution_result.metadata["final_score"] = turn.score
            execution_result.metrics["execution_time"] = turn.execution_time
            execution_result.set_tool_calls(filtered_final_tool_calls)

            # Add best attempt response output for traceability
            best_attempt = next((a for a in all_attempts if a["attempt"] == turn.best_attempt), None)
            if best_attempt:
                execution_result.add_output(
                    {
                        "type": "response",
                        "content": best_attempt.get("agent_response", ""),
                        "attempt": turn.best_attempt,
                        "tool_calls": filtered_final_tool_calls,
                    }
                )

            # Print tool calls in evaluation format
            formatted_calls = self._format_tool_calls_for_evaluation(filtered_final_tool_calls)
            self._trace_print(f"{test_id}[1/1]: {formatted_calls}")

            execution_result.mark_completed(success=execution_result.success)
            return execution_result

        except Exception as e:
            logger.error(f"SimSolver single-turn execution failed for {test_id}: {e}")
            # Fall back to legacy single-turn evaluator for debugging/compare
            return self._execute_single_turn(test_case, execution_result)

    @staticmethod
    def _extract_tool_call_error_text(result: Any) -> Optional[str]:
        """Return a normalized error string when a tool result clearly indicates failure."""
        if isinstance(result, dict):
            error = result.get("error")
            if error:
                return str(error)

            detail = result.get("detail")
            if isinstance(detail, dict):
                error_message = detail.get("error_message")
                if error_message:
                    return str(error_message)
            elif isinstance(detail, str) and detail:
                return detail

            if result.get("success") is False:
                return str(result.get("message") or "success=false")

        if isinstance(result, str) and "error" in result.lower():
            return result

        return None

    @classmethod
    def _filter_failed_tool_calls_for_single_turn(
        cls, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Drop validation/runtime-failed calls from single-turn final answers.

        Single-turn BFCL official eval expects only the final executable answer.
        Attempts may legitimately probe and then correct arguments inside one turn,
        but failed calls should remain only in debug traces, not in the exported answer.
        """
        filtered: List[Dict[str, Any]] = []
        for tool_call in tool_calls or []:
            if not isinstance(tool_call, dict):
                continue
            if cls._extract_tool_call_error_text(tool_call.get("result")):
                continue
            filtered.append(tool_call)
        return filtered
    
    
    def _generate_response_with_agent(self, test_case: TestCase, agent, attempt: int = 0) -> str:
        try:
            system_messages, user_question = self._extract_system_messages_and_question(test_case)
            
            question_str = str(user_question) if user_question else "No question"
            
            agent_response = agent.generate_response(question_str)
            
            if agent_response.success:
                self._current_tool_calls = agent_response.tool_calls
                return agent_response.raw_response or "No response"
            else:
                logger.error(f"Agent response failed: {agent_response.error_message}")
                return f"Error: {agent_response.error_message}"
                
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return f"Failed to generate response: {e}"
    
    def _generate_response_with_agent_direct(self, test_case: TestCase, agent, session_id: str, attempt: int = 0) -> str:
        try:
            system_messages, user_question = self._extract_system_messages_and_question(test_case)
            
            question_str = str(user_question) if user_question else "No question"
            
            self._current_session_id = session_id
            
            # ==================== INFERENCE COST DEBUG START ====================
            logger.info("="*80)
            test_id = TestCaseAdapter.get_id(test_case)
            logger.info(f"[INFERENCE DEBUG] Test ID: {test_id}, Attempt: {attempt}")
            logger.info(f"[INFERENCE DEBUG] === COMPLETE INPUT TO MODEL ===")
            logger.info(f"[INFERENCE DEBUG] Question (Full):\n{question_str}")
            if system_messages:
                logger.info(f"[INFERENCE DEBUG] System Messages Count: {len(system_messages)}")
                for i, msg in enumerate(system_messages):
                    logger.info(f"[INFERENCE DEBUG] System Message {i} (Full):\n{msg}")
            
            # Print the actual system prompt that will be used
            if hasattr(agent, '_camel_agent') and hasattr(agent._camel_agent, 'system_message'):
                logger.info(f"[INFERENCE DEBUG] === ACTUAL SYSTEM PROMPT TO MODEL ===")
                logger.info(f"{agent._camel_agent.system_message.content}")
            
            # Print tool definitions
            if hasattr(self, '_current_tool_definitions') and self._current_tool_definitions:
                import json
                logger.info(f"[INFERENCE DEBUG] === TOOL DEFINITIONS ({len(self._current_tool_definitions)}) ===")
                for i, tool_def in enumerate(self._current_tool_definitions):
                    logger.info(f"[INFERENCE DEBUG] Tool {i}: {json.dumps(tool_def, indent=2)}")
            # ==================== INFERENCE COST DEBUG END ====================
            
            agent_response = agent.generate_response(question_str)
            
            if hasattr(agent_response, 'metadata') and agent_response.metadata:
                self._current_llm_cost = agent_response.metadata.get('llm_cost', 0.0)
                self._current_input_tokens = agent_response.metadata.get('input_tokens', 0)
                self._current_output_tokens = agent_response.metadata.get('output_tokens', 0)
            else:
                self._current_llm_cost = 0.0
                self._current_input_tokens = 0
                self._current_output_tokens = 0
            
            if agent_response.success:
                # ==================== INFERENCE COST DEBUG RESULTS ====================
                logger.info(f"[INFERENCE DEBUG] === MODEL RESPONSE ===")
                logger.info(f"[INFERENCE DEBUG] Response Generated Successfully")
                logger.info(f"[INFERENCE DEBUG] Cost: ${self._current_llm_cost:.6f}")
                logger.info(f"[INFERENCE DEBUG] Input Tokens: {self._current_input_tokens}")
                logger.info(f"[INFERENCE DEBUG] Output Tokens: {self._current_output_tokens}")
                logger.info(f"[INFERENCE DEBUG] === COMPLETE RESPONSE FROM MODEL ===")
                logger.info(f"{str(agent_response.raw_response)}")
                
                # Print tool calls if any
                if hasattr(agent_response, 'tool_calls') and agent_response.tool_calls:
                    logger.info(f"[INFERENCE DEBUG] === TOOL CALLS EXTRACTED ===")
                    import json
                    for i, tc in enumerate(agent_response.tool_calls):
                        logger.info(f"[INFERENCE DEBUG] Tool Call {i}: {json.dumps(tc, indent=2)}")
                logger.info("="*80)
                # ==================== INFERENCE COST DEBUG END ====================
                logger.info(f"Agent generated response with potential tool execution (Cost: ${self._current_llm_cost:.4f})")
                
                tool_calls_from_server = self._mock_client.get_tool_calls_from_session(session_id, 0)
                
                self._current_tool_calls = tool_calls_from_server
                logger.info(f"Retrieved {len(tool_calls_from_server)} tool calls from Mock Server")
                
                for call in tool_calls_from_server:
                    logger.debug(f"Tool call: {call.get('name', 'unknown')} -> result: {call.get('result', 'no result')}")
                
                return agent_response.raw_response
            else:
                logger.error(f"Agent failed to generate response: {agent_response.error_message}")
                self._current_tool_calls = []
                return f"I need to calculate the area of a triangle with base {question_str}."
                
        except Exception as e:
            logger.error(f"Error in response generation: {e}")
            self._current_tool_calls = []
            return f"Error occurred while processing: {str(e)}"
    
    def _generate_response(self, test_case: TestCase, attempt: int = 0) -> str:
        try:
            system_messages, user_question = self._extract_system_messages_and_question(test_case)
            
            question_str = str(user_question) if user_question else "No question"
            
            session_id = self._get_or_create_session(test_case)
            self._current_session_id = session_id
            
            agent = self._create_agent_with_tools_for_session(test_case, session_id)
            
            agent_response = agent.generate_response(question_str)
            
            if agent_response.success:
                logger.info(f"Agent generated response with potential tool execution")
                
                tool_calls_from_server = self._mock_client.get_tool_calls_from_session(session_id, 0)
                
                self._current_tool_calls = tool_calls_from_server
                logger.info(f"Retrieved {len(tool_calls_from_server)} tool calls from Mock Server")
                
                for call in tool_calls_from_server:
                    logger.debug(f"Tool call: {call.get('name', 'unknown')} -> result: {call.get('result', 'no result')}")
                
                return agent_response.raw_response
            else:
                logger.error(f"Agent failed to generate response: {agent_response.error_message}")
                self._current_tool_calls = []
                return f"I need to calculate the area of a triangle with base {question_str}."
                
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            if hasattr(test_case.content, 'question'):
                if isinstance(test_case.content.question, list):
                    question = test_case.content.question[0] if test_case.content.question else "No question"
                else:
                    question = str(test_case.content.question)
            else:
                question = str(test_case.content)
            return f"I need to calculate the area of a triangle."
    
    def _create_agent_with_tools_for_session(self, test_case: TestCase, session_id: str):
        try:
            openapi_schema_paths = self._get_openapi_schemas(test_case)
            
            system_messages, _ = self._extract_system_messages_and_question(test_case)
            
            agent = self._create_agent_with_tools_impl(
                openapi_schema_paths=openapi_schema_paths,
                session_id=session_id,
                base_url=self.config.mock_server_url,
                model_name=self.config.model_name,
                enable_tool_filtering=self.config.enable_tool_filtering,
                task_content=self._get_task_content(test_case),
                system_messages=system_messages
            )
            
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create agent with tools: {e}")
            from inference.agents.chat_agent import ChatAgent
            return ChatAgent(
                model_name=self.config.model_name
            )
    
    def _create_agent_with_retry_context(self, test_case: TestCase, session_id: str, 
                                        previous_evaluation=None,
                                        previous_tool_calls=None,
                                        attempt: int = 0):
        try:
            openapi_schema_paths = self._get_openapi_schemas(test_case)
            
            system_messages, _ = self._extract_system_messages_and_question(test_case)
            
            if previous_evaluation and attempt > 0:
                retry_context = self._format_retry_context(previous_evaluation, previous_tool_calls, attempt)
                system_messages.append(retry_context)
                logger.info(f"Added retry context for attempt {attempt}")
            
            agent = self._create_agent_with_tools_impl(
                openapi_schema_paths=openapi_schema_paths,
                session_id=session_id,
                base_url=self.config.mock_server_url,
                model_name=self.config.model_name,
                enable_tool_filtering=self.config.enable_tool_filtering,
                task_content=self._get_task_content(test_case),
                system_messages=system_messages
            )
            
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create agent with retry context: {e}")
            from inference.agents.chat_agent import ChatAgent
            return ChatAgent(
                model_name=self.config.model_name
            )

    def _create_agent_with_tools(self, test_case: TestCase):
        session_id = self._get_or_create_session(test_case)
        return self._create_agent_with_tools_for_session(test_case, session_id)
    
    def _get_or_create_session(self, test_case: TestCase) -> str:
        if not self._mock_client:
            from inference.client_engine import MockServerClient
            self._mock_client = MockServerClient(self.config.mock_server_url)
        
        session_id = self._mock_client.create_session(test_case)
        test_id = TestCaseAdapter.get_id(test_case)
        logger.info(f"Created new session for {test_id}: {session_id}")
        
        return session_id
    
    def _get_openapi_schemas(self, test_case: TestCase) -> List[str]:
        import os

        def _validate_paths(paths: List[str]) -> List[str]:
            for path in paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"OpenAPI schema file not found: {path}")
            return paths

        # Check if content has openapi_schema_path
        if isinstance(test_case.content, dict):
            openapi_path = test_case.content.get('openapi_schema_path')
            if openapi_path:
                return _validate_paths([openapi_path])

        if hasattr(test_case, 'metadata') and test_case.metadata:
            schemas = test_case.metadata.get('openapi_schemas', [])
            if schemas:
                return _validate_paths(list(schemas))

        test_id = TestCaseAdapter.get_id(test_case)
        schema_base_dir = "data/bfcl_v4/openapi"
        single_turn_dir = os.path.join(schema_base_dir, "single_turn")

        if 'multi_turn' in test_id:
            involved_classes: List[str] = []
            if hasattr(test_case, 'metadata') and test_case.metadata:
                involved_classes = test_case.metadata.get('involved_classes', [])

            if not involved_classes:
                if hasattr(test_case.content, 'involved_classes'):
                    involved_classes = list(test_case.content.involved_classes)
                elif isinstance(test_case.content, dict):
                    involved_classes = list(test_case.content.get('involved_classes', []))

            if not involved_classes:
                raise ValueError(
                    f"Multi-turn test {test_id} missing 'involved_classes' metadata for schema resolution"
                )

            schema_file_map = {
                'GorillaFileSystem': 'GorillaFileSystem.json',
                'TwitterAPI': 'TwitterAPI.json',
                'MathAPI': 'MathAPI.json',
                'MessageAPI': 'MessageAPI.json',
                'PostingAPI': 'TwitterAPI.json',
                'TicketAPI': 'TicketAPI.json',
                'TradingBot': 'TradingBot.json',
                'TravelAPI': 'TravelAPI.json',
                'TravelBooking': 'TravelAPI.json',
                'VehicleControl': 'VehicleControlAPI.json',
                'VehicleControlAPI': 'VehicleControlAPI.json'
            }

            schema_paths: List[str] = []
            for class_name in involved_classes:
                schema_filename = schema_file_map.get(class_name)
                if not schema_filename:
                    raise ValueError(
                        f"Unknown involved class '{class_name}' for test {test_id}; "
                        "add it to schema_file_map to proceed"
                    )
                schema_path = f"{schema_base_dir}/multi_turn/{schema_filename}"
                if not os.path.exists(schema_path):
                    raise FileNotFoundError(
                        f"Expected schema for {class_name} not found at {schema_path}"
                    )
                schema_paths.append(schema_path)

            if not schema_paths:
                raise RuntimeError(f"No schema paths resolved for multi-turn test {test_id}")
            return schema_paths

        compact_schema_path = os.path.join(
            single_turn_dir,
            f"{derive_single_turn_schema_name(test_id)}.json",
        )
        if os.path.exists(compact_schema_path):
            return [compact_schema_path]

        def _candidate_test_ids(base_id: str) -> List[str]:
            candidates = [base_id]
            # BFCL v4 simple_* tasks often share the simple_ schema naming convention.
            if base_id.startswith(("simple_python_", "simple_java_", "simple_javascript_")):
                suffix = base_id.split("_", 2)[-1]
                candidates.append(f"simple_{suffix}")
            return candidates

        attempted: List[str] = []
        for candidate_id in _candidate_test_ids(test_id):
            schema_paths = []
            for i in range(10):
                found = False
                for base_dir in (schema_base_dir, single_turn_dir):
                    schema_path = f"{base_dir}/{candidate_id}_{i}.json"
                    attempted.append(schema_path)
                    if os.path.exists(schema_path):
                        schema_paths.append(schema_path)
                        found = True
                        break
                if not found:
                    break
            if schema_paths:
                return schema_paths

        raise FileNotFoundError(
            f"No OpenAPI schemas found for test {test_id} under "
            f"{schema_base_dir} or {single_turn_dir}. Last attempted: {attempted[-1] if attempted else 'n/a'}"
        )
    
    def _get_task_content(self, test_case: TestCase) -> str:
        # Handle dictionary content
        if isinstance(test_case.content, dict):
            return test_case.content.get('question', str(test_case.content))
        # Handle object content (like BFCL)
        elif hasattr(test_case.content, 'question'):
            if isinstance(test_case.content.question, list):
                return test_case.content.question[0] if test_case.content.question else ""
            else:
                return str(test_case.content.question)
        else:
            return str(test_case.content)
    
    def _extract_system_messages_and_question(self, test_case: TestCase) -> tuple:
        system_messages = []
        user_question = ""
        
        # Handle dictionary content
        if isinstance(test_case.content, dict):
            question = test_case.content.get('question', "")
        elif hasattr(test_case.content, 'question'):
            question = test_case.content.question
        else:
            question = None
        
        if question:
            
            if isinstance(question, list):
                
                if len(question) > 0:
                    first_item = question[0]
                    
                    if isinstance(first_item, dict) and 'role' in first_item:
                        for msg in question:
                            if isinstance(msg, dict):
                                role = msg.get('role', '')
                                content = msg.get('content', '')
                                if role == 'system' and content:
                                    system_messages.append(content)
                                elif role == 'user' and content:
                                    user_question = content
                    
                    elif isinstance(first_item, list) and len(first_item) > 0:
                        inner_list = first_item
                        for msg in inner_list:
                            if isinstance(msg, dict) and 'role' in msg:
                                role = msg.get('role', '')
                                content = msg.get('content', '')
                                if role == 'system' and content:
                                    system_messages.append(content)
                                elif role == 'user' and content:
                                    user_question = content
                    
                    if not user_question:
                        if isinstance(first_item, str):
                            user_question = first_item
                        elif isinstance(first_item, dict):
                            user_question = str(first_item)
                        else:
                            user_question = str(first_item)
            else:
                user_question = str(question)
        else:
            user_question = str(test_case.content)
        
        return system_messages, user_question
    
    def _create_agent_with_tools_impl(self, 
                                     openapi_schema_paths: List[str],
                                     session_id: str,
                                     base_url: str,
                                     model_name: str = "gpt-4.1-mini",
                                     enable_tool_filtering: bool = False,
                                     task_content: str = "",
                                     system_messages: Optional[List[str]] = None):
        """
        Create a ChatAgent with tools from OpenAPI schemas.
        
        Args:
            openapi_schema_paths: List of paths to OpenAPI schema files
            session_id: Session ID for the mock server
            base_url: Base URL of the mock server
            model_name: Model name for the agent
            enable_tool_filtering: Whether to filter tools based on task relevance
            task_content: Task content for tool filtering
            system_messages: Optional list of system prompts
            
        Returns:
            ChatAgent instance with tools
        """
        import json
        from camel.toolkits import OpenAPIToolkit
        from camel.toolkits import FunctionTool
        from inference.agents.chat_agent import ChatAgent
        from benchmarks.bfcl.tool_filtering import filter_definitely_irrelevant_tools
        
        try:
            # Load OpenAPI schemas
            openapi_json_list = []
            for path in openapi_schema_paths or []:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        openapi_json_list.append(json.load(f))
                        logger.info(f"Loaded schema: {path}")
                except Exception as e:
                    logger.warning(f"Failed to load schema {path}: {e}")
            
            # Build toolkit and tools
            tools = []
            # Store tool definitions for task evaluation
            self._current_tool_definitions = []
            
            if openapi_json_list:
                openapi_toolkit = OpenAPIToolkit()
                openapi_toolkit.set_session_id(session_id)
                openapi_toolkit.set_override_server_url(base_url)
                
                for openapi_json in openapi_json_list:
                    try:
                        api_name = openapi_json.get("info", {}).get("title") or "api"
                        funcs = openapi_toolkit.generate_openapi_funcs(api_name, openapi_json)
                        schemas = openapi_toolkit.openapi_spec_to_openai_schemas(api_name, openapi_json)
                        
                        # Truncate function names to 64 characters if needed
                        for schema in schemas:
                            if 'function' in schema and 'name' in schema['function']:
                                original_name = schema['function']['name']
                                if len(original_name) > 64:
                                    truncated_name = original_name[:64]
                                    schema['function']['name'] = truncated_name
                                    logger.warning(f"Truncated function name from {len(original_name)} to 64 chars: {original_name} -> {truncated_name}")
                        
                        # Also update function names to match truncated schema names
                        for i, (func, schema) in enumerate(zip(funcs, schemas)):
                            if 'function' in schema and 'name' in schema['function']:
                                func.__name__ = schema['function']['name']
                        
                        tools.extend([FunctionTool(func=f, openai_tool_schema=s) for f, s in zip(funcs, schemas)])
                        
                        # Store tool definitions for efficiency evaluation
                        self._current_tool_definitions.extend(schemas)
                        logger.info(f"Generated {len(funcs)} tools from {api_name}, stored {len(schemas)} tool definitions")
                    except Exception as e:
                        logger.warning(f"Failed to build tools from schema: {e}")

            if enable_tool_filtering and tools and self._current_tool_definitions:
                try:
                    filter_payload = []
                    for tool_def in self._current_tool_definitions:
                        fn = tool_def.get("function", {}) if isinstance(tool_def, dict) else {}
                        params = fn.get("parameters", {}) if isinstance(fn, dict) else {}
                        filter_payload.append(
                            {
                                "name": fn.get("name", ""),
                                "description": fn.get("description", ""),
                                "parameters": {
                                    "required": params.get("required", []),
                                    "properties": params.get("properties", {}),
                                },
                            }
                        )

                    irrelevant_tool_names = filter_definitely_irrelevant_tools(
                        task=task_content,
                        tools=filter_payload,
                        model_name=model_name,
                        timeout=min(self.config.timeout, 60),
                    )
                    if irrelevant_tool_names:
                        irrelevant_set = set(irrelevant_tool_names)
                        filtered_triplets = [
                            (tool, tool_def)
                            for tool, tool_def in zip(tools, self._current_tool_definitions)
                            if tool_def.get("function", {}).get("name") not in irrelevant_set
                        ]
                        tools = [tool for tool, _ in filtered_triplets]
                        self._current_tool_definitions = [
                            tool_def for _, tool_def in filtered_triplets
                        ]
                        logger.info(
                            "Tool filtering removed %d/%d tools: %s",
                            len(irrelevant_tool_names),
                            len(filter_payload),
                            irrelevant_tool_names,
                        )
                    else:
                        logger.info(
                            "Tool filtering kept all %d tools for task",
                            len(filter_payload),
                        )
                except Exception as e:
                    logger.warning(f"Tool filtering failed; keeping all tools: {e}")
            
            # Build the agent.
            default_system_message = "You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\nPay attention to the description of the functions, which decide whether the function can be used to solve the task. If none of the functions can be used, point it out by saying \"irrelevant\". If the given question lacks the parameters required by the function, also point it out by saying \"missing parameters\".\n\nAt each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.\nPay attention to the tool call results. If a tool call fails, you will see the error message in the tool call result. Use the error message to help you solve the task better.\nFor the files mentioned in the task, you can assume that files are exist in the current directory."

            configured_system_prompt = getattr(self.config, "agent_system_prompt", default_system_message)
            task_system_message = "\n\n".join(system_messages) if system_messages else ""

            if configured_system_prompt is None:
                if task_system_message:
                    agent = ChatAgent(
                        model_name=model_name,
                        system_message=task_system_message,
                    )
                else:
                    agent = ChatAgent(model_name=model_name)
            else:
                base_system_prompt = (
                    configured_system_prompt
                    if configured_system_prompt != ""
                    else default_system_message
                )
                if task_system_message:
                    system_prompt = (
                        f"{base_system_prompt}\n\n--- Task-Specific Instructions ---\n"
                        f"{task_system_message}"
                    )
                else:
                    system_prompt = base_system_prompt
                agent = ChatAgent(
                    model_name=model_name,
                    system_message=system_prompt
                )
            
            # Set tools
            if tools:
                agent.set_tools(tools)
                logger.info(f"Set {len(tools)} tools for agent")
            
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create agent with tools: {e}")
            return ChatAgent(model_name=model_name)
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        if hasattr(self, '_current_tool_calls') and self._current_tool_calls:
            logger.info(f"Using pre-extracted tool calls: {len(self._current_tool_calls)}")
            return self._current_tool_calls
        
        tool_calls = []
        response_lower = response.lower()
        
        
        if not tool_calls:
            import re
            import json
            from utils.model_utils import sanitize_llm_json_text

            response = sanitize_llm_json_text(response)
            
            pattern = r'<function_call>(.*?)</function_call>'
            matches = re.findall(pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    call_data = json.loads(match.strip())
                    tool_calls.append({
                        'type': 'function_call',
                        'name': call_data.get('name'),
                        'arguments': call_data.get('arguments', {})
                    })
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse function call: {match}")
        
        return tool_calls
    
    def _execute_tools(self, tool_calls: List[Dict[str, Any]], test_case: TestCase) -> List[Dict[str, Any]]:
        results = []
        
        for tool_call in tool_calls:
            try:
                result = self._execute_tool_via_mock_server(tool_call, test_case)
                results.append(result)
                
                real_result = self._real_execute_tool(tool_call, test_case)
                results.append(real_result)
                    
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                results.append({
                    'type': 'execution_error',
                    'tool_call': tool_call,
                    'error': str(e)
                })
        
        return results
    
    def _execute_tool_via_mock_server(self, tool_call: Dict[str, Any], test_case: TestCase) -> Dict[str, Any]:
        try:
            function_name = tool_call.get('function') or tool_call.get('name', '')
            arguments = tool_call.get('arguments', {})
            
            if 'triangle' in function_name or 'calculate' in function_name:
                if 'requestBody' in arguments:
                    body = arguments['requestBody']
                    base = body.get('base', 0)
                    height = body.get('height', 0)
                    area = (base * height) / 2
                    result = {'area': area, 'unit': 'square units'}
                else:
                    result = {'area': 25.0, 'unit': 'square units'}
            else:
                result = f"Mock result for {function_name}"
            
            logger.info(f"Mock Server executed {function_name}: {result}")
            
            return {
                'type': 'function_call',
                'name': function_name,
                'arguments': arguments,
                'result': result,
                'execution_type': 'mock_server'
            }
            
        except Exception as e:
            logger.error(f"Mock Server execution failed: {e}")
            return {
                'type': 'execution_error',
                'tool_call': tool_call,
                'error': str(e)
            }
    
    def _mock_execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        function_name = tool_call.get('name')
        arguments = tool_call.get('arguments', {})
        
        if function_name == 'add':
            a = arguments.get('a', 0)
            b = arguments.get('b', 0)
            result = a + b
        elif function_name == 'read_file':
            result = f"Content of {arguments.get('path', 'unknown')}"
        else:
            result = f"Mock result for {function_name}"
        
        return {
            'type': 'function_call',
            'name': function_name,
            'arguments': arguments,
            'result': result,
            'execution_type': 'mock'
        }
    
    def _real_execute_tool(self, tool_call: Dict[str, Any], test_case: TestCase) -> Dict[str, Any]:
        
        return {
            'type': 'function_call',
            'name': tool_call.get('name'),
            'arguments': tool_call.get('arguments', {}),
            'result': f"Real execution result for {tool_call.get('name')}",
            'execution_type': 'real'
        }
    
    def _extract_task_description(self, test_case: TestCase) -> str:
        # Handle dictionary content
        if isinstance(test_case.content, dict):
            question = test_case.content.get('question', "Complete the task")
            return str(question) if question else "Complete the task"
        elif hasattr(test_case.content, 'question'):
            question = test_case.content.question
            if isinstance(question, list) and question:
                return str(question[-1]) if question else "Complete the task"
            return str(question) if question else "Complete the task"
        
        return test_case.metadata.get('task', 'Complete the requested task')
    
    def _extract_current_state(self, execution_result: ExecutionResult) -> Dict:
        state = {}
        
        for output in execution_result.outputs:
            if output.get('type') == 'response' and 'tool_calls' in output:
                for tc in output['tool_calls']:
                    if 'result' in tc and tc['result']:
                        state[tc.get('name', 'unknown')] = tc['result']
        
        return state
    
    def _format_tool_calls_for_evaluation(self, tool_calls: List[Dict]) -> List[str]:
        """Format tool calls into evaluation format like ['grep(file_name="...", pattern="...")']"""
        formatted = []
        for call in tool_calls:
            if isinstance(call, str):
                formatted.append(call)
                continue
                
            function_name = call.get('function', call.get('name', ''))
            arguments = call.get('arguments', {})
            
            if not function_name:
                continue
                
            # Remove API prefixes and path components
            if '/' in function_name:
                function_name = function_name.split('/')[-1]
            
            # Remove common API prefixes
            prefixes = ['gorillafilesystem_', 'postingapi_', 'messageapi_', 'ticketapi_', 
                       'mathapi_', 'tradingbot_', 'travelbooking_', 'vehiclecontrol_']
            for prefix in prefixes:
                if function_name.startswith(prefix):
                    function_name = function_name[len(prefix):]
                    break
                    
            # Extract parameters
            params = {}
            if isinstance(arguments, dict):
                if 'requestBody' in arguments:
                    params = arguments['requestBody']
                else:
                    params = arguments
            elif isinstance(arguments, str):
                try:
                    import json
                    params = json.loads(arguments)
                except:
                    params = {}
            
            # Format as function call string
            if params and isinstance(params, dict) and params:
                # Format parameters with proper escaping for BFCL evaluation
                param_parts = []
                for k, v in params.items():
                    if isinstance(v, str):
                        # Escape newlines in string values for BFCL format
                        escaped_v = v.replace('\n', '\\n')
                        param_parts.append(f"{k}='{escaped_v}'")
                    else:
                        param_parts.append(f"{k}={v}")
                param_str = ','.join(param_parts)  # No space after comma for BFCL format
                formatted.append(f"{function_name}({param_str})")
            else:
                formatted.append(f"{function_name}()")
                
        return formatted
    
    
    def batch_execute(self, test_ids: List[str]) -> List[ExecutionResult]:
        if not self.benchmark:
            raise ValueError("No benchmark set. Call set_benchmark() first.")
        
        results = []
        
        for test_id in test_ids:
            try:
                test_case = self.benchmark.load_test_case(test_id)
                
                result = self.execute_test_case(test_case)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to execute test {test_id}: {e}")
                error_result = ExecutionResult(test_id=test_id)
                error_result.mark_completed(success=False, error=str(e))
                results.append(error_result)
        
        return results
    
    def _format_retry_context(self, previous_evaluation, previous_tool_calls, attempt: int) -> str:
        """
        Format retry context message to be added to system messages
        
        Args:
            previous_evaluation: TaskEvaluationResult from previous attempt
            previous_tool_calls: List of tool calls from previous attempt
            attempt: Current attempt number (1-based for retry)
            
        Returns:
            Formatted retry context message
        """
        context_parts = []
        
        # Add header
        context_parts.append(f"\n--- Previous Attempt {attempt} Failed ---")
        
        # Add evaluation feedback
        if previous_evaluation:
            context_parts.append(f"Score: {previous_evaluation.score:.2f}")
            
            if previous_evaluation.retry_reason:
                context_parts.append(f"Main Issue: {previous_evaluation.retry_reason}")
            
            # Add judgment details with actionable hints
            if hasattr(previous_evaluation, 'judgment') and previous_evaluation.judgment:
                failed_items = [
                    item for item in previous_evaluation.judgment 
                    if isinstance(item, dict) and item.get('status') == 'failed'
                ]
                if failed_items:
                    context_parts.append("\n❌ Failed requirements:")
                    for item in failed_items[:3]:  # Limit to 3 items
                        desc = item.get('description', 'Unknown')
                        reasoning = item.get('reasoning', 'No reason provided')
                        context_parts.append(f"  • {desc}")
                        context_parts.append(f"    Problem: {reasoning}")
        
        # Add previous tool calls  
        if previous_tool_calls:
            context_parts.append(f"\nPrevious tool calls ({len(previous_tool_calls)}):")
            for i, tc in enumerate(previous_tool_calls[:5], 1):  # Limit to 5
                name = tc.get('name', tc.get('function', 'unknown'))
                args = tc.get('arguments', {})
                
                # Clean up function name (remove common prefixes)
                clean_name = name.replace('gorillafilesystem_', '').replace('postingapi_', '')
                
                # Format arguments compactly
                if isinstance(args, dict):
                    if 'requestBody' in args:
                        args = args['requestBody']
                    args_str = ', '.join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" 
                                         for k, v in list(args.items())[:2]])
                    if len(args) > 2:
                        args_str += ", ..."
                else:
                    args_str = str(args)
                
                context_parts.append(f"  {i}. {clean_name}({args_str})")
                
                result = tc.get('result')
                error = tc.get('error')
                if error:
                    context_parts.append(f"     → ERROR: {error}")
                elif result and not isinstance(result, dict):
                    context_parts.append(f"     → Result: {str(result)[:50]}")
        
        context_parts.append("---\n")
        
        # Join all parts but limit to avoid too long context
        full_context = "\n".join(context_parts)
        if len(full_context) > 2000:  # Limit context size
            # Truncate the middle part (usually tool call details)
            full_context = full_context[:1500] + "\n... [truncated] ...\n" + full_context[-400:]
        
        return full_context
    
    def get_engine_info(self) -> Dict[str, Any]:
        return {
            'engine_type': 'UniversalInferenceEngine',
            'config': self.config.to_dict(),
            'benchmark': self.benchmark.name if self.benchmark else None,
            'capabilities': {
                'single_turn': True,
                'multi_turn': True,
                'mock_execution': self.config.enable_mock_execution,
                'real_execution': True,
                'batch_processing': True
            }
        }

#!/usr/bin/env python3

import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from inference.core import SimSolver, TurnOutcome
from inference.client_engine import MockServerClient, InferenceConfig, TurnResult, InferenceResult
from benchmarks.bfcl.data.loader import TestCase
from utils.test_case_adapter import TestCaseAdapter

logger = logging.getLogger(__name__)

# Load environment variables for API keys
try:
    from utils.env_loader import load_environment_variables
    load_environment_variables()
except ImportError:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.warning("Unable to load environment variables")


class MultiTurnExecutor:
    """
    Refactored Multi-turn executor using SimSolver
    
    Key improvements:
    - Clean separation between mock and real execution
    - SimSolver handles all mock server interactions
    - Example extraction built-in
    """
    
    def __init__(self, config: InferenceConfig):
        """
        Initialize executor
        
        Args:
            config: Inference configuration
        """
        self.config = config
        self.mock_client = MockServerClient(config.mock_server_url)
        self.real_tool_instances = None

    def _trace_print(self, message: str) -> None:
        if bool(getattr(self.config, "enable_debug", False)):
            print(message)
        
    def execute_multi_turn(self, test_case: TestCase) -> InferenceResult:
        """
        Execute multi-turn test using SimSolver
        
        Args:
            test_case: Test case to execute
            
        Returns:
            InferenceResult with all turn data and examples
        """
        # Extract BFCL test case if wrapped
        if hasattr(test_case, 'content') and test_case.content:
            bfcl_test_case = test_case.content
        else:
            bfcl_test_case = test_case
        
        test_id = TestCaseAdapter.get_id(test_case)
        start_time = time.time()

        logger.info(f"Starting multi-turn execution for {test_id}")
        self._trace_print(f"[FLOW] Loaded test case {test_id}")

        # Determine OpenAPI tool paths based on involved_classes for multi-turn tests
        openapi_tool_paths = self._get_openapi_paths_for_test(test_case)
        if openapi_tool_paths:
            self._trace_print(f"[FLOW] OpenAPI tool paths: {len(openapi_tool_paths)}")
        else:
            self._trace_print("[FLOW] OpenAPI tool paths: 0")

        # Initialize SimSolver with all configuration options
        base_checklist_items = None
        checklist_system_prompt = getattr(self.config, "checklist_system_prompt", None)
        custom_config = getattr(self.config, "custom_config", {}) or {}
        if isinstance(custom_config, dict):
            base_checklist_items = custom_config.get("base_checklist_items")
            if checklist_system_prompt is None:
                checklist_system_prompt = custom_config.get("checklist_system_prompt")

        solver = SimSolver(
            test_case=test_case,
            model_name=self.config.model_name,
            max_retries=self.config.max_retries,
            agent_timeout=getattr(self.config, "timeout", None),
            mock_server_url=self.config.mock_server_url,
            override_openapi_server=getattr(self.config, "override_openapi_server", True),
            agent_max_iteration=getattr(self.config, "agent_max_iteration", None),
            agent_summarize_threshold=getattr(self.config, "agent_summarize_threshold", None),
            enable_evaluation=self.config.max_retries > 0,
            enable_checklist=getattr(self.config, 'enable_checklist', True),
            agent_system_prompt=getattr(self.config, 'agent_system_prompt', None),
            judge_system_prompt=getattr(self.config, 'judge_system_prompt', None),
            openapi_tool_paths=openapi_tool_paths or getattr(self.config, 'openapi_tool_paths', None),
            agent_persistence_mode=getattr(self.config, 'agent_persistence_mode', False),
            base_checklist_items=base_checklist_items,
            checklist_system_prompt=checklist_system_prompt,
            include_agent_response_in_judge=False,
            enable_tool_result_folding=getattr(self.config, "enable_tool_result_folding", True),
            # include_state_in_prompt removed (always false)
            enable_debug=bool(getattr(self.config, "enable_debug", False)),
            verbose_debug=False,
        )
        self._trace_print(f"[FLOW] SimSolver initialized (max_retries={self.config.max_retries})")
        
        # Initialize real tools
        try:
            self.real_tool_instances = self._initialize_shared_real_tools(test_case)
            logger.info(f"Initialized real tool instances: {list(self.real_tool_instances.keys())}")
        except Exception as e:
            logger.error(f"Failed to initialize real tools: {e}")
            self.real_tool_instances = None
        
        # Process each turn
        questions = TestCaseAdapter.get_questions(bfcl_test_case)
        self._trace_print(f"[FLOW] Parsed questions: {len(questions)} turns")
        turns = []
        
        for turn_idx, question_turn in enumerate(questions):
            # Extract question text
            question_text = question_turn[0]['content'] if question_turn and question_turn[0].get('content') else str(question_turn)
            
            logger.info(f"Processing turn {turn_idx}: {question_text[:100]}...")
            
            # Capture start config for optional real sync
            turn_start_config = solver.current_config

            # Use SimSolver to process the turn
            self._trace_print(f"[FLOW] Turn {turn_idx} start")
            turn_result = solver.process(question_text)
            self._trace_print(
                f"[FLOW] Turn {turn_idx} done: score={turn_result.score:.2f}, "
                f"tool_calls={len(turn_result.final_tool_calls)}"
            )
            
            # Print tool calls in evaluation format
            formatted_calls = self._format_tool_calls_for_evaluation(turn_result.final_tool_calls)
            self._trace_print(f"{test_id}[{turn_idx + 1}/{len(questions)}]: {formatted_calls}")
            
            # Execute real tools when turn is successful
            real_tool_calls = []
            calibrated_config = None
            if self.real_tool_instances:
                if turn_result.score >= solver.target_score or self.config.final_state_only:
                    real_results = self._execute_real_tools(turn_result.final_tool_calls)
                    
                    if real_results:
                        real_tool_calls = real_results
                        calibrated_config = self.mock_client.sync_state_from_real_results(
                            base_config=turn_start_config,
                            tool_calls=real_results,
                            test_case=test_case,
                        )
                        logger.info(f"Synced config for turn {turn_idx} with {len(real_results)} real tool calls")
                        if calibrated_config:
                            solver.set_initial_config(calibrated_config)
                            logger.info(f"Applied calibrated config for next turn {turn_idx + 1}")
            
            # Convert to legacy TurnResult format for compatibility
            legacy_turn = self._convert_to_legacy_turn(
                turn_result,
                turn_idx,
                real_tool_calls=real_tool_calls,
                calibrated_config=calibrated_config,
            )
            turns.append(legacy_turn)
            
            # Continue with remaining turns even if score < 1.0
            # Previously we would stop execution here, but now we continue
            if turn_result.score < solver.target_score:
                logger.warning(
                    f"Turn {turn_idx} score {turn_result.score:.2f} < target {solver.target_score}, but continuing with remaining turns"
                )
        
        # Extract examples for future use
        # Examples are now represented as an event trace (see solver.get_events()).
        
        # Calculate overall results
        overall_success = all(turn.success for turn in turns)
        final_score = sum(turn.judge_score for turn in turns) / len(turns) if turns else 0.0
        
        # Aggregate LLM costs
        total_llm_costs = self._aggregate_costs(turns)
        
        # Build result
        result = InferenceResult(
            test_id=test_id,
            success=overall_success,
            turns=turns,
            total_time=time.time() - start_time,
            judge_score=final_score,
            config_history=solver._config_history,
            total_llm_costs=total_llm_costs
        )
        
        return result
    
    def _convert_to_legacy_turn(
        self,
        turn: TurnOutcome,
        turn_idx: int,
        *,
        real_tool_calls: List[Dict[str, Any]] | None = None,
        calibrated_config: Optional[Dict[str, Any]] = None,
    ) -> TurnResult:
        """Convert SimSolver turn to legacy TurnResult format"""
        # Reconstruct attempt data from events
        attempts_by_idx: Dict[int, Dict[str, Any]] = {}
        for ev in turn.events:
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

        # Extract checklist template if present
        checklist = []
        for ev in turn.events:
            if ev.type == "checklist":
                raw = ev.data.get("checklist", []) or []
                checklist = [{"description": it.get("description", "")} for it in raw if isinstance(it, dict)]
                break
        
        return TurnResult(
            turn_idx=turn_idx,
            success=turn.score >= 1.0,
            mock_tool_calls=turn.final_tool_calls,
            real_tool_calls=real_tool_calls or [],
            judge_score=turn.score,
            execution_time=turn.execution_time,
            mock_config=turn.final_config,
            calibrated_config=calibrated_config,
            llm_costs={},  # Cost accounting removed; tokens are in event trace
            checklist=checklist,
            all_attempts=all_attempts
        )
    
    def _initialize_shared_real_tools(self, test_case: TestCase) -> Dict[str, Any]:
        """Initialize shared real tool instances with initial config - only called once per test case"""
        return self.mock_client._initialize_real_tool_instances(test_case)
    
    def _execute_real_tools(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute tool calls with real instances"""
        if not self.real_tool_instances:
            return []
        
        results = []
        for tool_call in tool_calls:
            function_name = tool_call.get('function', '')
            arguments = tool_call.get('arguments', {})
            
            result = self.mock_client._execute_single_real_tool(
                self.real_tool_instances, function_name, arguments
            )
            
            results.append({
                'function': function_name,
                'arguments': arguments,
                'result': result
            })
        
        return results
    
    def _get_openapi_paths_for_test(self, test_case: TestCase) -> List[str]:
        """Get OpenAPI file paths based on involved_classes for multi-turn tests.

        Args:
            test_case: Test case with metadata containing involved_classes

        Returns:
            List of OpenAPI file paths, or None if not a multi-turn test
        """
        # Check if this is a multi-turn test with involved_classes
        metadata = getattr(test_case, 'metadata', {})
        involved_classes = metadata.get('involved_classes', [])

        if not involved_classes:
            return None

        # Mapping from involved_classes to OpenAPI file names (matching actual files)
        MULTI_TURN_TOOLKIT_MAPPING = {
            "GorillaFileSystem": "GorillaFileSystem",
            "PostingAPI": "TwitterAPI",  # PostingAPI uses TwitterAPI
            "TwitterAPI": "TwitterAPI",
            "VehicleControlAPI": "VehicleControlAPI",  # Fixed: was VehicleControl
            "MessageAPI": "MessageAPI",
            "TravelAPI": "TravelAPI",
            "BookingAPI": "TravelAPI",  # BookingAPI also uses TravelAPI
            "MathAPI": "MathAPI",
            "TradingBot": "TradingBot",
            "TicketAPI": "TicketAPI"  # Added missing TicketAPI
        }

        import os
        openapi_paths = []
        base_dir = "/media/zeyu/DA1474031473E145/FuncCallDataGen/data/bfcl_v4/openapi/multi_turn"

        for class_name in involved_classes:
            if class_name in MULTI_TURN_TOOLKIT_MAPPING:
                schema_name = MULTI_TURN_TOOLKIT_MAPPING[class_name]
                path = os.path.join(base_dir, f"{schema_name}.json")

                if os.path.exists(path):
                    if path not in openapi_paths:  # Avoid duplicates
                        openapi_paths.append(path)
                        logger.info(f"Added OpenAPI path for {class_name}: {path}")
                else:
                    logger.warning(f"OpenAPI file not found for {class_name}: {path}")
            else:
                logger.warning(f"Unknown involved class: {class_name}")

        if openapi_paths:
            logger.info(f"Loaded {len(openapi_paths)} OpenAPI files for test with involved_classes: {involved_classes}")
        else:
            logger.warning(f"No OpenAPI files found for involved_classes: {involved_classes}")

        return openapi_paths if openapi_paths else None

    def _format_tool_calls_for_evaluation(self, tool_calls: List[Dict]) -> List[str]:
        """Format tool calls into evaluation format like ['grep(file_name="...", pattern="...")']"""
        formatted = []
        for call in tool_calls:
            if isinstance(call, str):
                formatted.append(call)
                continue
                
            function_name = call.get('function', '')
            arguments = call.get('arguments', {})
            
            if not function_name:
                continue
                
            # Remove API prefixes
            if 'gorillafilesystem_' in function_name:
                function_name = function_name.replace('gorillafilesystem_', '')
            elif 'postingapi_' in function_name:
                function_name = function_name.replace('postingapi_', '')
            elif 'messageapi_' in function_name:
                function_name = function_name.replace('messageapi_', '')
            elif 'ticketapi_' in function_name:
                function_name = function_name.replace('ticketapi_', '')
            elif 'mathapi_' in function_name:
                function_name = function_name.replace('mathapi_', '')
            elif 'tradingbot_' in function_name:
                function_name = function_name.replace('tradingbot_', '')
            elif 'travelbooking_' in function_name:
                function_name = function_name.replace('travelbooking_', '')
            elif 'vehiclecontrol_' in function_name:
                function_name = function_name.replace('vehiclecontrol_', '')
                
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
    
    def _aggregate_costs(self, turns: List[TurnResult]) -> Dict[str, float]:
        """Aggregate LLM costs from all turns"""
        total_costs = {}
        for turn in turns:
            if turn.llm_costs:
                for category, cost in turn.llm_costs.items():
                    if category not in total_costs:
                        total_costs[category] = 0.0
                    total_costs[category] += cost
        return total_costs

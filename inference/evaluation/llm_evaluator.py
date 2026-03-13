"""
LLM-Based Task Evaluator
Intelligent evaluation using language models for sophisticated test time scaling
"""

import logging
from datetime import datetime
from typing import List, Dict

from .task_evaluator import BaseTaskEvaluator, TaskEvaluationResult
from utils.legacy.task_feedback import TaskFeedback

logger = logging.getLogger(__name__)


class LLMTaskEvaluator(BaseTaskEvaluator):
    """
    LLM-based task evaluator for intelligent test time scaling
    
    This evaluator uses a language model to:
    1. Generate a checklist of requirements for the task
    2. Judge whether the execution meets those requirements
    3. Provide intelligent retry reasons if the task is incomplete
    
    Automatically caches checklists per task to avoid regeneration during retries.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        enable_checklist: bool = True,
        enable_trace: bool = False,
    ):
        """
        Initialize LLM-based evaluator
        
        Args:
            model_name: Model to use for evaluation (default: "gpt-4.1-mini")
            enable_checklist: Whether to generate detailed checklist (default: True)
                             If False, uses simpler single-item checklist for speed
        """
        self.model_name = model_name
        self.enable_checklist = enable_checklist
        self.enable_trace = bool(enable_trace)
        self.task_feedback = TaskFeedback(model_name=model_name)
        self._checklist_cache = {}  # Cache checklists by task hash
        
        logger.info(f"Initialized LLMTaskEvaluator with model: {model_name}, checklist: {enable_checklist}")

    def _trace_print(self, message: str) -> None:
        if self.enable_trace:
            print(message)
    
    def evaluate_completion(
        self, 
        task: str,
        tool_calls: List[Dict],
        current_state: Dict = None,
        initial_state: Dict = None,
        tool_definitions: List[Dict] = None
    ) -> TaskEvaluationResult:
        """
        Evaluate task completion using LLM judgment
        
        Args:
            task: Task description to evaluate against
            tool_calls: List of tool calls made by the model
            current_state: Current state after execution
            initial_state: Initial state before execution
            tool_definitions: List of available tool definitions for efficiency analysis
            
        Returns:
            TaskEvaluationResult with LLM-based evaluation
        """
        try:
            # Step 1: Generate or retrieve cached checklist
            checklist_cost = 0.0
            checklist_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            judge_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            
            # Create a stable cache key from task content
            import hashlib
            task_key = hashlib.md5(task.encode()).hexdigest()
            
            if task_key in self._checklist_cache:
                # Use cached checklist for retry
                checklist = self._checklist_cache[task_key]
                logger.debug(f"Using cached checklist with {len(checklist)} items for task {task_key[:8]}... (no generation cost)")
            elif self.enable_checklist:
                # Generate new checklist (first evaluation)
                # ==================== CHECKLIST GENERATION DEBUG START ====================
                if self.enable_trace:
                    logger.info("="*80)
                    logger.info(f"[CHECKLIST DEBUG] Generating checklist for task {task_key[:8]}...")
                    logger.info(f"[CHECKLIST DEBUG] Task Input: {task[:200]}...")
                    logger.info(f"[CHECKLIST DEBUG] Tool Definitions Count: {len(tool_definitions) if tool_definitions else 0}")
                # ==================== CHECKLIST GENERATION DEBUG END ====================
                
                logger.debug(f"Generating checklist for task {task_key[:8]}...: {task[:100]}...")
                _t0 = datetime.now()
                self._trace_print(
                    f"[TIME] Checklist LLM START { _t0.strftime('%H:%M:%S') } (model={self.model_name})"
                )
                checklist = self.task_feedback.generate_checklist(
                    task=task,
                    initial_config=None,  # No longer pass initial_state to save tokens
                    previous_tasks=[],  # No previous tasks in single-turn
                    tool_definitions=tool_definitions
                )
                _t1 = datetime.now()
                self._trace_print(
                    f"[TIME] Checklist LLM END   { _t1.strftime('%H:%M:%S') } "
                    f"(elapsed={( _t1 - _t0 ).total_seconds():.3f}s, model={self.model_name})"
                )

                # ==================== CHECKLIST GENERATION RESULTS DEBUG ====================
                if self.enable_trace:
                    logger.info(f"[CHECKLIST DEBUG] === COMPLETE CHECKLIST GENERATED ===")
                    logger.info(f"[CHECKLIST DEBUG] Generated {len(checklist)} checklist items")
                    import json
                    for i, item in enumerate(checklist):
                        logger.info(f"[CHECKLIST DEBUG] Item {i}: {json.dumps(item, indent=2)}")
                    logger.info("="*80)
                # ==================== CHECKLIST GENERATION RESULTS DEBUG END ====================
                
                # Cache the checklist for future retries
                self._checklist_cache[task_key] = checklist
                # Capture token usage for checklist generation
                if isinstance(getattr(self.task_feedback, "last_checklist_usage", None), dict):
                    checklist_usage = self.task_feedback.last_checklist_usage
                logger.debug(f"Generated and cached {len(checklist)} checklist items, cost: ${checklist_cost:.4f}")
            else:
                # Simple checklist for faster evaluation
                checklist = [{"description": f"Complete the requested task: {task}"}]
                # Also cache simple checklists
                self._checklist_cache[task_key] = checklist
                logger.debug("Using and caching simple checklist (checklist generation disabled)")
            
            # Step 2: Format tool calls for judgment
            tool_call_strings = self._format_tool_calls(tool_calls)
            
            # Step 3: Judge execution against checklist
            # ==================== JUDGE EXECUTION DEBUG START ====================
            if self.enable_trace:
                logger.info("="*80)
                logger.info(f"[JUDGE DEBUG] Judging execution for task {task_key[:8]}...")
                logger.info(f"[JUDGE DEBUG] === COMPLETE INPUT TO JUDGE ===")

                logger.info(f"[JUDGE DEBUG] === CHECKLIST ({len(checklist)} items) ===")
                import json
                for item in checklist:
                    logger.info(f"[JUDGE DEBUG] Checklist Item: {json.dumps(item, indent=2)}")

                logger.info(f"[JUDGE DEBUG] === TOOL CALLS ({len(tool_call_strings)} calls) ===")
                for i, tc in enumerate(tool_call_strings):
                    logger.info(f"[JUDGE DEBUG] Tool Call {i}: {tc}")

                if current_state:
                    logger.info(f"[JUDGE DEBUG] === CURRENT STATE ===")
                    logger.info(f"[JUDGE DEBUG] {json.dumps(current_state, indent=2)}")

                if tool_definitions:
                    logger.info(f"[JUDGE DEBUG] === TOOL DEFINITIONS FOR EFFICIENCY CHECK ===")
                    for i, td in enumerate(tool_definitions[:2]):
                        logger.info(f"[JUDGE DEBUG] Tool Def {i}: {json.dumps(td, indent=2)}")
                    if len(tool_definitions) > 2:
                        logger.info(f"[JUDGE DEBUG] ... and {len(tool_definitions) - 2} more tool definitions")
            # ==================== JUDGE EXECUTION DEBUG END ====================
            
            logger.debug("Judging execution against checklist...")
            _jt0 = datetime.now()
            self._trace_print(
                f"[TIME] Judge LLM START      { _jt0.strftime('%H:%M:%S') } (model={self.model_name})"
            )
            judgment_results, critical_responses, score = self.task_feedback.judge_execution(
                checklist=checklist,
                current_config=current_state or {},
                tool_calls=tool_call_strings,
                tool_definitions=tool_definitions,
                task=task
            )
            _jt1 = datetime.now()
            self._trace_print(
                f"[TIME] Judge LLM END        { _jt1.strftime('%H:%M:%S') } "
                f"(elapsed={( _jt1 - _jt0 ).total_seconds():.3f}s, model={self.model_name})"
            )
            # Capture token usage for judge call
            if isinstance(getattr(self.task_feedback, "last_judge_usage", None), dict):
                judge_usage = self.task_feedback.last_judge_usage

            # ==================== JUDGE EXECUTION RESULTS DEBUG ====================
            if self.enable_trace:
                logger.info(f"[JUDGE DEBUG] === COMPLETE JUDGMENT RESULTS ===")
                logger.info(f"[JUDGE DEBUG] Score: {score:.2f}")
                logger.info(f"[JUDGE DEBUG] Passed: {score >= 0.8}")

                if critical_responses:
                    logger.info(f"[JUDGE DEBUG] === CRITICAL RESPONSES ({len(critical_responses)}) ===")
                    for i, cr in enumerate(critical_responses):
                        logger.info(f"[JUDGE DEBUG] Critical Response {i}: {cr}")

                logger.info(f"[JUDGE DEBUG] === DETAILED JUDGMENTS ({len(judgment_results)}) ===")
                import json
                for result in judgment_results:
                    logger.info(f"[JUDGE DEBUG] Judgment: {json.dumps(result, indent=2)}")

                logger.info("="*80)
            # ==================== JUDGE EXECUTION RESULTS DEBUG END ====================
            
            # Step 4: Determine pass/fail and retry reason
            passed = score >= 0.8  # 80% threshold for passing
            
            retry_reason = None
            if not passed:
                if critical_responses:
                    # Use the first critical response as retry reason
                    retry_reason = critical_responses[0]
                else:
                    # Generic retry reason based on score
                    retry_reason = f"Task incomplete (score: {score:.2f}). The model needs to better address the requirements."
            
            # Costs are not derived here; record token usage instead for accounting upstream.
            judge_cost = 0.0
            total_evaluation_cost = checklist_cost + judge_cost
            logger.info(f"LLM evaluation complete - Score: {score:.2f}, Passed: {passed}, Cost: ${total_evaluation_cost:.4f}")
            if not passed:
                logger.info(f"Retry reason: {retry_reason}")

            total_tokens = {
                "input_tokens": int(checklist_usage.get("input_tokens", 0) or 0)
                + int(judge_usage.get("input_tokens", 0) or 0),
                "output_tokens": int(checklist_usage.get("output_tokens", 0) or 0)
                + int(judge_usage.get("output_tokens", 0) or 0),
                "total_tokens": int(checklist_usage.get("total_tokens", 0) or 0)
                + int(judge_usage.get("total_tokens", 0) or 0),
            }
            return TaskEvaluationResult(
                score=score,
                passed=passed,
                retry_reason=retry_reason,
                checklist=checklist,
                judgment=judgment_results,
                metadata={
                    "checklist_cost": checklist_cost,
                    "judge_cost": judge_cost,
                    "total_evaluation_cost": total_evaluation_cost,
                    "checklist_tokens": checklist_usage,
                    "judge_tokens": judge_usage,
                    "total_evaluation_tokens": total_tokens,
                }
            )
            
        except Exception as e:
            # If LLM evaluation fails, fall back to a conservative approach
            logger.error(f"LLM evaluation failed: {e}")
            logger.warning("Falling back to conservative evaluation")
            
            # Check if we at least have tool calls
            if tool_calls and len(tool_calls) > 0:
                # Assume partial success if tools were called
                return TaskEvaluationResult(
                    score=0.5,
                    passed=False,
                    retry_reason=f"Evaluation error: {str(e)}. Retrying may help.",
                    checklist=None,
                    judgment=None
                )
            else:
                # No tool calls - likely incomplete
                return TaskEvaluationResult(
                    score=0.0,
                    passed=False,
                    retry_reason="No tool calls made and evaluation failed. The model needs to use tools.",
                    checklist=None,
                    judgment=None
                )
    
    def _format_tool_calls(self, tool_calls: List[Dict]) -> List[str]:
        """
        Format tool calls for TaskFeedback judgment
        
        Args:
            tool_calls: List of tool call dictionaries
            
        Returns:
            List of formatted tool call strings
        """
        formatted = []
        for tc in tool_calls:
            name = tc.get('name', 'unknown')
            args = tc.get('arguments', {})
            result = tc.get('result')
            
            # Format as function call string
            call_str = f"{name}({args})"
            
            # Add result if available
            if result:
                call_str += f" -> {result}"
            
            formatted.append(call_str)
        
        return formatted
    
    def clear_cache(self):
        """Clear the checklist cache"""
        self._checklist_cache.clear()
        logger.debug("Cleared checklist cache")
    
    def get_cache_size(self) -> int:
        """Get the number of cached checklists"""
        return len(self._checklist_cache)
    
    def __repr__(self) -> str:
        cache_info = f", cache_size={len(self._checklist_cache)}" if self._checklist_cache else ""
        return f"LLMTaskEvaluator(model='{self.model_name}', checklist={self.enable_checklist}{cache_info})"

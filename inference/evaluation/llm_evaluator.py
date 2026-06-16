"""
LLM-Based Task Evaluator
Intelligent evaluation using language models for sophisticated test time scaling
"""

import logging
from typing import List, Dict, Optional

from .task_evaluator import BaseTaskEvaluator, TaskEvaluationResult
from inference.evaluation.task_feedback import TaskFeedback

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
        checklist_system_prompt: Optional[str] = None,
    ):
        """
        Initialize LLM-based evaluator
        
        Args:
            model_name: Model to use for evaluation (default: "gpt-4.1-mini")
            enable_checklist: Whether to generate detailed checklist. This only
                             takes effect when checklist_system_prompt is provided.
        """
        self.model_name = model_name
        self.enable_checklist = bool(enable_checklist and checklist_system_prompt)
        self.task_feedback = TaskFeedback(
            model_name=model_name,
            checklist_system_prompt=checklist_system_prompt,
        )
        self._checklist_cache = {}  # Cache checklists by task hash
        
        logger.info(f"Initialized LLMTaskEvaluator with model: {model_name}, checklist: {self.enable_checklist}")
    
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
            tool_definitions: List of available tool definitions
            
        Returns:
            TaskEvaluationResult with LLM-based evaluation
        """
        try:
            import hashlib
            task_key = hashlib.md5(task.encode()).hexdigest()
            
            if task_key in self._checklist_cache:
                checklist = self._checklist_cache[task_key]
                logger.debug(f"Using cached checklist with {len(checklist)} items for task {task_key[:8]}")
            elif self.enable_checklist:
                logger.debug(f"Generating checklist for task {task_key[:8]}...: {task[:100]}...")
                checklist = self.task_feedback.generate_checklist(
                    task=task,
                    initial_config=None,
                    previous_tasks=[],
                    tool_definitions=tool_definitions
                )
                
                self._checklist_cache[task_key] = checklist
                logger.debug(f"Generated and cached {len(checklist)} checklist items")
            else:
                checklist = []
                self._checklist_cache[task_key] = checklist
                logger.debug("Checklist generation disabled")
            
            tool_call_strings = self._format_tool_calls(tool_calls)
            
            logger.debug("Judging execution against checklist...")
            judgment_results, critical_responses, score = self.task_feedback.judge_execution(
                checklist=checklist,
                current_config=current_state or {},
                tool_calls=tool_call_strings,
                tool_definitions=tool_definitions,
                task=task
            )
            
            passed = score >= 0.8
            
            retry_reason = None
            if not passed:
                if critical_responses:
                    retry_reason = critical_responses[0]
                else:
                    retry_reason = f"Task incomplete (score: {score:.2f}). The model needs to better address the requirements."
            
            logger.info(f"LLM evaluation complete - Score: {score:.2f}, Passed: {passed}")
            if not passed:
                logger.info(f"Retry reason: {retry_reason}")

            return TaskEvaluationResult(
                score=score,
                passed=passed,
                retry_reason=retry_reason,
                checklist=checklist,
                judgment=judgment_results,
                metadata={},
            )
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            logger.warning("Falling back to conservative evaluation")
            
            if tool_calls and len(tool_calls) > 0:
                return TaskEvaluationResult(
                    score=0.5,
                    passed=False,
                    retry_reason=f"Evaluation error: {str(e)}. Retrying may help.",
                    checklist=None,
                    judgment=None
                )
            else:
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
            
            call_str = f"{name}({args})"
            
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

"""
Rule-Based Task Evaluator
Simple and fast evaluation based on predefined rules
"""

import logging
from typing import List, Dict

from .task_evaluator import BaseTaskEvaluator, TaskEvaluationResult

logger = logging.getLogger(__name__)


class RuleBasedTaskEvaluator(BaseTaskEvaluator):
    """
    Rule-based task evaluator for fast test time scaling
    
    This evaluator uses simple rules to determine task completion:
    - require_tool_calls: Task passes if tool calls were made
    - always_pass: Task always passes (effectively disables retry)
    """
    
    def __init__(self, mode: str = "require_tool_calls"):
        """
        Initialize rule-based evaluator
        
        Args:
            mode: Evaluation mode
                - "require_tool_calls": Require at least one tool call (default)
                - "always_pass": Always pass (disable retry mechanism)
        """
        if mode not in ["require_tool_calls", "always_pass"]:
            raise ValueError(f"Unknown mode: {mode}. Must be 'require_tool_calls' or 'always_pass'")
        
        self.mode = mode
        logger.info(f"Initialized RuleBasedTaskEvaluator with mode: {mode}")
    
    def evaluate_completion(
        self, 
        task: str,
        tool_calls: List[Dict],
        current_state: Dict = None,
        initial_state: Dict = None,
        tool_definitions: List[Dict] = None
    ) -> TaskEvaluationResult:
        """
        Evaluate task completion based on rules
        
        Args:
            task: Task description (not used in rule-based evaluation)
            tool_calls: List of tool calls made
            current_state: Current state (not used in rule-based evaluation)
            initial_state: Initial state (not used in rule-based evaluation)
            tool_definitions: Tool definitions (not used in rule-based evaluation)
            
        Returns:
            TaskEvaluationResult based on the configured mode
        """
        if self.mode == "always_pass":
            # Always pass mode - effectively disables retry
            logger.debug("Always pass mode - task automatically passes")
            return TaskEvaluationResult(
                score=1.0,
                passed=True,
                retry_reason=None
            )
        
        elif self.mode == "require_tool_calls":
            # Check if any tool calls were made
            if not tool_calls or len(tool_calls) == 0:
                logger.debug("No tool calls found - task fails")
                return TaskEvaluationResult(
                    score=0.0,
                    passed=False,
                    retry_reason="No tool calls were generated. The model needs to use available tools to complete the task."
                )
            
            # Check for errors in tool calls
            errors = []
            for i, tc in enumerate(tool_calls):
                if tc.get('error'):
                    errors.append(f"Tool '{tc.get('name', f'#{i}')}' failed: {tc.get('error')}")
            
            if errors:
                error_msg = " | ".join(errors)
                logger.debug(f"Tool execution errors found: {error_msg}")
                return TaskEvaluationResult(
                    score=0.5,  # Partial score for attempted but failed calls
                    passed=False,
                    retry_reason=f"Tool execution failed: {error_msg}"
                )
            
            # Tool calls made successfully
            logger.debug(f"Found {len(tool_calls)} successful tool calls - task passes")
            return TaskEvaluationResult(
                score=1.0,
                passed=True,
                retry_reason=None
            )
    
    def __repr__(self) -> str:
        return f"RuleBasedTaskEvaluator(mode='{self.mode}')"
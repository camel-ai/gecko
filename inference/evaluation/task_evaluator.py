"""
Task Evaluator Base Classes
Core components for test time scaling - evaluating task completion for retry decisions
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class TaskEvaluationResult:
    """
    Task evaluation result for test time scaling decisions
    
    This is used to determine whether to retry or accept the current attempt.
    """
    score: float  # 0-1 score indicating completion quality
    passed: bool  # Whether the task is considered complete
    retry_reason: Optional[str] = None  # Reason for retry if not passed
    checklist: Optional[List[Dict]] = None  # Task checklist (if generated)
    judgment: Optional[List[Dict]] = None  # Detailed judgment results
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata (e.g., costs)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'score': self.score,
            'passed': self.passed,
            'retry_reason': self.retry_reason,
            'checklist': self.checklist,
            'judgment': self.judgment,
            'metadata': self.metadata
        }


class BaseTaskEvaluator(ABC):
    """
    Abstract base class for task evaluators
    
    Task evaluators are part of the test time scaling mechanism.
    They determine whether a task is complete and if retry is needed.
    This is NOT about comparing with ground truth - it's about
    evaluating if the model has successfully completed the requested task.
    """
    
    @abstractmethod
    def evaluate_completion(
        self, 
        task: str,
        tool_calls: List[Dict],
        current_state: Dict = None,
        initial_state: Dict = None,
        tool_definitions: List[Dict] = None
    ) -> TaskEvaluationResult:
        """
        Evaluate whether the task has been completed successfully
        
        Args:
            task: Task description or question
            tool_calls: List of tool calls made by the model
            current_state: Current state after tool execution
            initial_state: Initial state before task execution
            tool_definitions: List of available tool definitions/schemas for efficiency analysis
            
        Returns:
            TaskEvaluationResult with score, pass/fail, and retry reason
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
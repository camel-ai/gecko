
from abc import ABC, abstractmethod
from typing import Any, Dict
from .test_case import TestCase
from .execution_result import ExecutionResult


class BaseEvaluator(ABC):
    
    @abstractmethod
    def evaluate(self, test_case: TestCase, execution_result: ExecutionResult) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_evaluation_metrics(self) -> Dict[str, str]:
        pass
    
    def validate_evaluation_result(self, evaluation: Dict[str, Any]) -> bool:
        required_keys = ['score', 'passed', 'details']
        return all(key in evaluation for key in required_keys)
    
    def create_evaluation_result(self, 
                                score: float, 
                                passed: bool, 
                                details: Dict[str, Any] = None) -> Dict[str, Any]:
        if not 0.0 <= score <= 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
        
        return {
            'score': score,
            'passed': passed,
            'details': details or {},
            'evaluator': self.__class__.__name__
        }
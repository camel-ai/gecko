
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from .test_case import TestCase
from .execution_result import ExecutionResult
from .evaluator import BaseEvaluator


class BaseBenchmark(ABC):
    
    def __init__(self, **config):
        self.config = config
        self._evaluator: Optional[BaseEvaluator] = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod  
    def version(self) -> str:
        pass
    
    @abstractmethod
    def list_test_ids(self, **filters) -> List[str]:
        pass
    
    @abstractmethod
    def load_test_case(self, test_id: str) -> TestCase:
        pass
    
    @abstractmethod
    def create_evaluator(self) -> BaseEvaluator:
        pass
    
    def get_evaluator(self) -> BaseEvaluator:
        if self._evaluator is None:
            self._evaluator = self.create_evaluator()
        return self._evaluator
    
    def evaluate(self, test_case: TestCase, execution_result: ExecutionResult) -> Dict[str, Any]:
        evaluator = self.get_evaluator()
        return evaluator.evaluate(test_case, execution_result)
    
    @abstractmethod
    def format_result(self, test_case: TestCase, execution_result: ExecutionResult, 
                     evaluation: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    def get_test_count(self, **filters) -> int:
        return len(self.list_test_ids(**filters))
    
    def validate_test_id(self, test_id: str) -> bool:
        try:
            all_test_ids = self.list_test_ids()
            return test_id in all_test_ids
        except Exception:
            return False
    
    def get_benchmark_info(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'version': self.version,
            'config': self.config,
            'total_tests': self.get_test_count()
        }
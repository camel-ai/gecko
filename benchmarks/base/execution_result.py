
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionResult:
    
    test_id: str
    
    success: bool = False
    
    outputs: List[Any] = field(default_factory=list)
    
    metrics: Dict[str, float] = field(default_factory=dict)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    error: Optional[str] = None
    
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    def __post_init__(self):
        if not self.test_id:
            raise ValueError("Test ID cannot be empty")
        
        if not isinstance(self.outputs, list):
            raise ValueError("Outputs must be a list")
        
        if not isinstance(self.metrics, dict):
            raise ValueError("Metrics must be a dictionary")
        
        if not isinstance(self.metadata, dict):
            raise ValueError("Metadata must be a dictionary")
    
    def mark_completed(self, success: bool = True, error: str = None):
        self.end_time = time.time()
        self.success = success
        if error:
            self.error = error
            self.success = False
        
        if self.end_time and self.start_time:
            self.metrics['execution_time'] = self.end_time - self.start_time
    
    def add_output(self, output: Any):
        self.outputs.append(output)
    
    def add_metric(self, name: str, value: float):
        self.metrics[name] = value
    
    def get_execution_time(self) -> Optional[float]:
        return self.metrics.get('execution_time')
    
    def get_total_outputs(self) -> int:
        return len(self.outputs)
    
    def has_error(self) -> bool:
        return self.error is not None
    
    def add_tool_call(self, function_name: str, arguments: Dict[str, Any], result: Any = None):
        tool_call = {
            'function': function_name,
            'arguments': arguments
        }
        if result is not None:
            tool_call['result'] = result
        self.tool_calls.append(tool_call)
    
    def set_tool_calls(self, tool_calls: List[Dict[str, Any]]):
        self.tool_calls = tool_calls if tool_calls else []
    
    def get_tool_calls_count(self) -> int:
        return len(self.tool_calls)

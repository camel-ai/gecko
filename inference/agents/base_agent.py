
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    success: bool
    tool_calls: List[Dict[str, Any]]
    raw_response: Optional[str] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAgent(ABC):
    
    def __init__(self, model_name: str = "gpt-4.1-mini", 
                 timeout: int = 300,
                 system_message: Optional[str] = None,
                 **kwargs):
        self.model_name = model_name
        self.timeout = timeout
        self.system_message = system_message
        self.config = kwargs
        
        self._model = None
        self._init_model()
    
    @abstractmethod
    def _init_model(self):
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str, 
                         context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        pass
    
    def parse_tool_calls(self, response_text: str) -> List[Dict[str, Any]]:
        try:
            return self._parse_tool_calls_impl(response_text)
        except Exception as e:
            logger.error(f"Failed to parse tool calls: {e}")
            return []
    
    @abstractmethod
    def _parse_tool_calls_impl(self, response_text: str) -> List[Dict[str, Any]]:
        pass
    
    def build_prompt(self, task: str, 
                    available_functions: List[Dict[str, Any]] = None,
                    context: Optional[Dict[str, Any]] = None,
                    previous_attempts: List[str] = None) -> str:
        prompt_parts = []
        
        if self.system_message:
            prompt_parts.append(f"System: {self.system_message}")
        
        prompt_parts.append(f"Task: {task}")
        
        if available_functions:
            prompt_parts.append("Available Functions:")
            for func in available_functions:
                func_name = func.get("name", "unknown")
                func_desc = func.get("description", "No description")
                prompt_parts.append(f"- {func_name}: {func_desc}")
        
        if context:
            prompt_parts.append(f"Context: {context}")
        
        if previous_attempts:
            prompt_parts.append("Previous Attempts:")
            for i, attempt in enumerate(previous_attempts, 1):
                prompt_parts.append(f"{i}. {attempt}")
        
        prompt_parts.append(
            "Please provide your response with function calls in the following format:\n"
            "Function calls should be clearly structured with function names and arguments."
        )
        
        return "\n\n".join(prompt_parts)
    
    def validate_response(self, response: AgentResponse) -> bool:
        if not response.success:
            return False
        
        if not isinstance(response.tool_calls, list):
            return False
        
        for tool_call in response.tool_calls:
            if not isinstance(tool_call, dict):
                return False
            
            if "function" not in tool_call:
                return False
            
            if "arguments" in tool_call and not isinstance(tool_call["arguments"], dict):
                return False
        
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "timeout": self.timeout,
            "system_message": self.system_message,
            "config": self.config
        }
    
    def update_config(self, **kwargs):
        self.config.update(kwargs)
        
        model_related_keys = ["model_name", "timeout"]
        if any(key in kwargs for key in model_related_keys):
            for key in model_related_keys:
                if key in kwargs:
                    setattr(self, key, kwargs[key])
            self._init_model()
    
    def reset(self):
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
    
    def __repr__(self) -> str:
        return self.__str__()

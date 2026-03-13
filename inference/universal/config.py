
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class InferenceConfig:

    model_name: str = "gpt-4.1-mini"

    model_config: Dict[str, Any] = field(default_factory=dict)

    max_retries: int = 3

    target_score: float = 1.0

    timeout: int = 120

    max_turns: Optional[int] = None

    trace_compact: bool = False

    agent_max_iteration: int = 10

    agent_summarize_threshold: Optional[int] = None

    enable_real_execution: bool = True

    mock_server_url: str = field(
        default_factory=lambda: os.getenv(
            "MOCK_SERVER_BASE",
            f"http://localhost:{os.getenv('MOCK_SERVER_PORT', '8000')}"
        )
    )

    override_openapi_server: bool = True
    
    enable_tool_filtering: bool = False
    
    task_evaluator_type: str = "llm"
    
    task_evaluator_model_name: str = "gpt-4.1-mini"
    
    rule_evaluator_mode: str = "require_tool_calls"
    
    enable_checklist: bool = True

    enable_tool_result_folding: bool = True
    
    use_sim_solver: bool = True
    
    agent_persistence_mode: bool = False
    
    final_state_only: bool = False
    
    agent_system_prompt: Optional[str] = None
    
    judge_system_prompt: Optional[str] = None

    checklist_system_prompt: Optional[str] = None

    judge_model_name: Optional[str] = None
    
    openapi_tool_paths: Optional[List[str]] = None

    include_state_in_prompt: bool = True
    
    max_workers: int = 1
    
    batch_size: int = 10
    
    enable_debug: bool = False
    
    log_level: str = "INFO"
    
    save_intermediate_results: bool = False
    
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Real execution is mandatory in current workflow.
        self.enable_real_execution = True

        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        
        if not 0 <= self.target_score <= 1:
            raise ValueError("target_score must be between 0 and 1")
        
        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")

        if self.max_turns is not None and self.max_turns <= 0:
            raise ValueError("max_turns must be > 0 when provided")
        
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        
        if not self.model_config:
            self.model_config = self._get_default_model_config()
    
    def _get_default_model_config(self) -> Dict[str, Any]:
        default_configs = {
            "gpt-5": {
                "temperature": 0.1,
                "max_tokens": 4096,
                "top_p": 1.0
            },
            "gpt-4": {
                "temperature": 0.1,
                "max_tokens": 4096,
                "top_p": 1.0
            },
            "4o": {  # GPT-4o
                "temperature": 0.1,
                "max_tokens": 4096,
                "top_p": 1.0
            },
            "gpt-3.5-turbo": {
                "temperature": 0.1,
                "max_tokens": 4096,
                "top_p": 1.0
            },
            "claude-3": {
                "temperature": 0.1,
                "max_tokens": 4096
            },
            "gemini": {
                "temperature": 0.1,
                "max_output_tokens": 4096
            }
        }
        
        for model_key, config in default_configs.items():
            if model_key.lower() in self.model_name.lower():
                return config.copy()
        
        return {
            "temperature": 0.1,
            "max_tokens": 4096
        }
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.custom_config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.custom_config.get(key, default)
    
    def merge(self, other_config: 'InferenceConfig') -> 'InferenceConfig':
        merged = InferenceConfig()
        
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, dict):
                setattr(merged, field_name, value.copy())
            elif isinstance(value, list):
                setattr(merged, field_name, value.copy())
            else:
                setattr(merged, field_name, value)
        
        for field_name in other_config.__dataclass_fields__:
            other_value = getattr(other_config, field_name)
            
            if field_name in ['model_config', 'custom_config'] and isinstance(other_value, dict):
                current_value = getattr(merged, field_name)
                current_value.update(other_value)
            else:
                setattr(merged, field_name, other_value)
        
        return merged
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            result[field_name] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'InferenceConfig':
        valid_fields = cls.__dataclass_fields__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        custom_config = {k: v for k, v in config_dict.items() if k not in valid_fields}
        if custom_config:
            filtered_dict['custom_config'] = custom_config
        
        return cls(**filtered_dict)
    
    def copy(self) -> 'InferenceConfig':
        return InferenceConfig.from_dict(self.to_dict())
    
    def __repr__(self) -> str:
        return f"InferenceConfig(model={self.model_name}, retries={self.max_retries}, score={self.target_score})"

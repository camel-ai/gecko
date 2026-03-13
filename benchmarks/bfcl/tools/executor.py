
import json
import logging
import sys
import os
import copy
from typing import Any, Dict, List, Optional, Type, Tuple
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    success: bool
    result: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    tool_state: Optional[Dict[str, Any]] = None


@dataclass
class ToolInstance:
    class_name: str
    instance: Any
    initial_state: Dict[str, Any] = field(default_factory=dict)
    current_state: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class BFCLToolExecutor:
    
    def __init__(self):
        self._tool_classes: Dict[str, Type] = {}
        self._tool_instances: Dict[str, ToolInstance] = {}
        self._function_to_class_mapping: Dict[str, str] = {}
        
        self._load_tool_classes()
        self._setup_function_mapping()
    
    def _load_tool_classes(self):
        try:
            from benchmarks.bfcl.multi_turn.func_source_code.travel_booking import TravelAPI
            from benchmarks.bfcl.multi_turn.func_source_code.math_api import MathAPI
            from benchmarks.bfcl.multi_turn.func_source_code.vehicle_control import VehicleControlAPI
            from benchmarks.bfcl.multi_turn.func_source_code.gorilla_file_system import GorillaFileSystem
            from benchmarks.bfcl.multi_turn.func_source_code.trading_bot import TradingBot
            from benchmarks.bfcl.multi_turn.func_source_code.message_api import MessageAPI
            from benchmarks.bfcl.multi_turn.func_source_code.posting_api import TwitterAPI
            from benchmarks.bfcl.multi_turn.func_source_code.ticket_api import TicketAPI
        except ImportError as e:
            raise ImportError(
                "Failed to load BFCL tool classes from benchmarks.bfcl.multi_turn.func_source_code."
            ) from e

        self._tool_classes = {
            "TravelAPI": TravelAPI,
            "MathAPI": MathAPI,
            "VehicleControlAPI": VehicleControlAPI,
            "GorillaFileSystem": GorillaFileSystem,
            "TradingBot": TradingBot,
            "MessageAPI": MessageAPI,
            "TwitterAPI": TwitterAPI,
            "TicketAPI": TicketAPI,
        }

        logger.info(f"Loaded {len(self._tool_classes)} BFCL tool classes")
    
    def _setup_function_mapping(self):
        for class_name, tool_class in self._tool_classes.items():
            try:
                for attr_name in dir(tool_class):
                    if not attr_name.startswith('_') and callable(getattr(tool_class, attr_name, None)):
                        if attr_name not in self._function_to_class_mapping:
                            self._function_to_class_mapping[attr_name] = class_name
            except Exception as e:
                logger.warning(f"Failed to setup function mapping for {class_name}: {e}")
    
    def create_tool_instance(self, class_name: str, config: Optional[Dict] = None) -> str:
        if class_name not in self._tool_classes:
            raise ValueError(f"Unknown tool class: {class_name}")
        
        tool_class = self._tool_classes[class_name]
        
        try:
            if config:
                instance = tool_class(**config)
            else:
                instance = tool_class()
        except Exception as e:
            logger.error(f"Failed to create instance of {class_name}: {e}")
            raise
        
        initial_state = self._capture_tool_state(instance)
        
        instance_id = f"{class_name}_{len(self._tool_instances)}"
        
        tool_instance = ToolInstance(
            class_name=class_name,
            instance=instance,
            initial_state=initial_state,
            current_state=copy.deepcopy(initial_state)
        )
        
        self._tool_instances[instance_id] = tool_instance
        
        logger.info(f"Created tool instance {instance_id} for {class_name}")
        return instance_id
    
    def execute_function(self, function_name: str, arguments: Dict[str, Any],
                        instance_id: Optional[str] = None) -> ExecutionResult:
        start_time = time.time()
        
        try:
            tool_instance = self._get_or_create_instance(function_name, instance_id)
            
            if not hasattr(tool_instance.instance, function_name):
                raise ValueError(f"Function {function_name} not found in {tool_instance.class_name}")
            
            function = getattr(tool_instance.instance, function_name)
            result = function(**arguments)
            
            tool_instance.current_state = self._capture_tool_state(tool_instance.instance)
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=True,
                result=result,
                execution_time=execution_time,
                tool_state=copy.deepcopy(tool_instance.current_state)
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Function execution failed: {error_msg}")
            
            return ExecutionResult(
                success=False,
                error_message=error_msg,
                execution_time=execution_time
            )
    
    def execute_tool_calls(self, tool_calls: List[Dict[str, Any]], 
                          reset_instances: bool = True) -> List[ExecutionResult]:
        if reset_instances:
            self.reset_all_instances()
        
        results = []
        
        for tool_call in tool_calls:
            function_name = tool_call.get("function")
            arguments = tool_call.get("arguments", {})
            
            if not function_name:
                results.append(ExecutionResult(
                    success=False,
                    error_message="Missing function name in tool call"
                ))
                continue
            
            result = self.execute_function(function_name, arguments)
            results.append(result)
        
        return results
    
    def _get_or_create_instance(self, function_name: str, instance_id: Optional[str]) -> ToolInstance:
        if instance_id and instance_id in self._tool_instances:
            return self._tool_instances[instance_id]
        
        class_name = self._function_to_class_mapping.get(function_name)
        if not class_name:
            raise ValueError(f"Unknown function: {function_name}")
        
        existing_instance = None
        for inst_id, tool_inst in self._tool_instances.items():
            if tool_inst.class_name == class_name:
                existing_instance = tool_inst
                break
        
        if existing_instance:
            return existing_instance
        else:
            new_instance_id = self.create_tool_instance(class_name)
            return self._tool_instances[new_instance_id]
    
    def _capture_tool_state(self, instance: Any) -> Dict[str, Any]:
        state = {}
        
        try:
            for attr_name in dir(instance):
                if not attr_name.startswith('_') and not callable(getattr(instance, attr_name, None)):
                    try:
                        attr_value = getattr(instance, attr_name)
                        if isinstance(attr_value, (str, int, float, bool, list, dict, tuple, set, type(None))):
                            state[attr_name] = copy.deepcopy(attr_value)
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Failed to capture tool state: {e}")
        
        return state
    
    def get_tool_state(self, instance_id: str) -> Optional[Dict[str, Any]]:
        if instance_id not in self._tool_instances:
            return None
        
        return copy.deepcopy(self._tool_instances[instance_id].current_state)
    
    def get_state_difference(self, instance_id: str) -> Optional[Dict[str, Any]]:
        if instance_id not in self._tool_instances:
            return None
        
        tool_instance = self._tool_instances[instance_id]
        initial = tool_instance.initial_state
        current = tool_instance.current_state
        
        diff = {}
        
        for key, value in current.items():
            if key not in initial:
                diff[key] = {"action": "added", "value": value}
            elif initial[key] != value:
                diff[key] = {"action": "modified", "old": initial[key], "new": value}
        
        for key in initial:
            if key not in current:
                diff[key] = {"action": "removed", "value": initial[key]}
        
        return diff
    
    def reset_tool_instance(self, instance_id: str) -> bool:
        if instance_id not in self._tool_instances:
            return False
        
        tool_instance = self._tool_instances[instance_id]
        
        try:
            tool_class = self._tool_classes[tool_instance.class_name]
            new_instance = tool_class()
            
            tool_instance.instance = new_instance
            tool_instance.current_state = self._capture_tool_state(new_instance)
            
            logger.info(f"Reset tool instance {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset tool instance {instance_id}: {e}")
            return False
    
    def reset_all_instances(self):
        for instance_id in list(self._tool_instances.keys()):
            self.reset_tool_instance(instance_id)
    
    def list_tool_classes(self) -> List[str]:
        return list(self._tool_classes.keys())
    
    def list_tool_instances(self) -> List[str]:
        return list(self._tool_instances.keys())
    
    def get_function_mapping(self) -> Dict[str, str]:
        return self._function_to_class_mapping.copy()


_tool_executor_instance = None

def get_bfcl_tool_executor() -> BFCLToolExecutor:
    global _tool_executor_instance
    if _tool_executor_instance is None:
        _tool_executor_instance = BFCLToolExecutor()
    return _tool_executor_instance

"""
Multi-Turn Tool Executor

Handles stateful execution of tools across multiple turns.
"""

import importlib
import inspect
import json
import re
import copy
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

# Global instance cache to replace globals() usage
# This provides better isolation and explicit management
_INSTANCE_CACHE: Dict[str, Any] = {}


def clear_instance_cache(test_entry_id: str = None):
    """
    Clear the instance cache for a specific test or all tests.

    Args:
        test_entry_id: If provided, only clear instances for this test.
                      If None, clear all instances.
    """
    global _INSTANCE_CACHE
    if test_entry_id is None:
        _INSTANCE_CACHE.clear()
    else:
        # Clear only instances for this specific test
        keys_to_remove = [k for k in _INSTANCE_CACHE.keys() if test_entry_id in k]
        for key in keys_to_remove:
            del _INSTANCE_CACHE[key]


def get_cache_stats():
    """Get statistics about the instance cache."""
    return {
        "total_instances": len(_INSTANCE_CACHE),
        "instance_keys": list(_INSTANCE_CACHE.keys())
    }


# Mapping of class names to their module paths
CLASS_FILE_PATH_MAPPING = {
    "GorillaFileSystem": "benchmarks.bfcl.multi_turn.func_source_code.gorilla_file_system",
    "MathAPI": "benchmarks.bfcl.multi_turn.func_source_code.math_api",
    "MessageAPI": "benchmarks.bfcl.multi_turn.func_source_code.message_api",
    "TwitterAPI": "benchmarks.bfcl.multi_turn.func_source_code.posting_api",
    "TicketAPI": "benchmarks.bfcl.multi_turn.func_source_code.ticket_api",
    "TradingBot": "benchmarks.bfcl.multi_turn.func_source_code.trading_bot",
    "TravelAPI": "benchmarks.bfcl.multi_turn.func_source_code.travel_booking",
    "VehicleControlAPI": "benchmarks.bfcl.multi_turn.func_source_code.vehicle_control",
}

# These classes are stateless and do not require any initial configuration
STATELESS_CLASSES = ["MathAPI"]


class MultiTurnExecutor:
    """
    Executor for multi-turn function calls with state management.
    """
    
    def __init__(self):
        self.instances = {}
        self.execution_globals = {}
    
    def reset(self):
        """Reset all instances and execution state."""
        self.instances = {}
        self.execution_globals = {}
    
    def execute_turn(
        self,
        func_call_list: List[str],
        initial_config: Dict,
        involved_classes: List[str],
        test_id: str,
        turn_index: int,
        long_context: bool = False
    ) -> Tuple[List[str], Dict]:
        """
        Execute function calls for a single turn.
        
        Args:
            func_call_list: List of function call strings
            initial_config: Initial configuration for classes
            involved_classes: Classes involved in this test
            test_id: Test case ID
            turn_index: Current turn index
            long_context: Whether this is a long context test
            
        Returns:
            Tuple of (execution_results, instances)
        """
        # Initialize instances if first turn
        if turn_index == 0 or not self.instances:
            self._initialize_instances(initial_config, involved_classes, test_id, long_context)
        
        # Execute function calls
        execution_results = []
        for func_call in func_call_list:
            result = self._execute_single_call(func_call)
            execution_results.append(result)
        
        return execution_results, self.instances
    
    def _initialize_instances(
        self,
        initial_config: Dict,
        involved_classes: List[str],
        test_id: str,
        long_context: bool = False
    ):
        """Initialize class instances for the test."""
        for class_name in involved_classes:
            if class_name not in CLASS_FILE_PATH_MAPPING:
                continue
                
            module_name = CLASS_FILE_PATH_MAPPING[class_name]
            instance_name = f"{test_id}_{class_name.lower()}_instance"
            
            # Import and instantiate class
            try:
                module = importlib.import_module(module_name)
                class_ = getattr(module, class_name)
                class_instance = class_()
                
                # Load initial configuration for stateful classes
                if class_name not in STATELESS_CLASSES:
                    class_initial_config = initial_config.get(class_name, {})
                    if hasattr(class_instance, '_load_scenario'):
                        class_instance._load_scenario(
                            copy.deepcopy(class_initial_config),
                            long_context=long_context
                        )
                
                self.instances[class_name] = class_instance
                self.execution_globals[instance_name] = class_instance
                
                # Map all methods to execution globals
                for method_name, method in inspect.getmembers(class_instance, predicate=inspect.ismethod):
                    if not method_name.startswith("_"):
                        # Create a direct method reference
                        self.execution_globals[method_name] = method
                        
            except Exception as e:
                logger.error("Error initializing %s: %s", class_name, e)
    
    def _execute_single_call(self, func_call: str) -> str:
        """Execute a single function call."""
        try:
            # Safety check for dangerous functions
            if self._is_dangerous_call(func_call):
                return f"Error: Function call {func_call} is not allowed."
            
            # Execute the function call
            result = eval(func_call, {"__builtins__": {}}, self.execution_globals)
            
            # Convert result to string
            if isinstance(result, str):
                return result
            elif isinstance(result, dict):
                try:
                    return json.dumps(result)
                except:
                    return str(result)
            else:
                return str(result)
                
        except Exception as e:
            return f"Error during execution: {str(e)}"
    
    def _is_dangerous_call(self, func_call: str) -> bool:
        """Check if a function call is dangerous."""
        dangerous_funcs = ["kill", "exit", "quit", "remove", "unlink", "popen", "Popen", "run", "__import__", "eval", "exec", "compile"]
        
        # Extract function name
        func_name = func_call.split("(")[0] if "(" in func_call else func_call
        if "." in func_name:
            func_name = func_name.split(".")[-1]
        
        return func_name in dangerous_funcs


def execute_multi_turn_func_call(
    func_call_list: List[str],
    initial_config: Dict,
    involved_classes: List[str],
    model_name: str,
    test_entry_id: str,
    long_context: bool = False,
    is_eval_run: bool = False,
) -> Tuple[List[str], Dict]:
    """
    Execute multi-turn function calls (compatibility function).
    
    This function maintains compatibility with the official BFCL interface.
    """
    if is_eval_run:
        model_name += "_eval"
    
    class_method_name_mapping = {}
    involved_instances = {}
    
    for class_name in involved_classes:
        if class_name not in CLASS_FILE_PATH_MAPPING:
            continue
            
        module_name = CLASS_FILE_PATH_MAPPING[class_name]
        instance_name = (
            f"{model_name.replace('-', '_').replace('.', '_').replace('/', '_')}_{test_entry_id}_{class_name.lower()}_instance"
        )

        # Check if instance already exists in cache
        if instance_name not in _INSTANCE_CACHE:
            try:
                module = importlib.import_module(module_name)
                class_ = getattr(module, class_name)
                class_instance = class_()

                if class_name not in STATELESS_CLASSES:
                    class_initial_config = initial_config.get(class_name, {})
                    if hasattr(class_instance, '_load_scenario'):
                        class_instance._load_scenario(
                            copy.deepcopy(class_initial_config),
                            long_context=long_context
                        )

                _INSTANCE_CACHE[instance_name] = class_instance
            except Exception as e:
                logger.error("Error creating instance %s: %s", instance_name, e)
                continue
        else:
            class_instance = _INSTANCE_CACHE[instance_name]
        
        involved_instances[class_name] = class_instance

        # Map method names to instance
        for method_name, method in inspect.getmembers(class_instance, predicate=inspect.ismethod):
            if not method_name.startswith("_"):
                class_method_name_mapping[method_name] = instance_name

    # Create execution namespace with instances from cache
    exec_namespace = {}
    for inst_name, inst_obj in _INSTANCE_CACHE.items():
        exec_namespace[inst_name] = inst_obj

    execution_results = []
    for func_call in func_call_list:
        # Add instance name to method calls
        func_call = _process_method_calls(func_call, class_method_name_mapping)
        
        # Execute the function call
        try:
            # Safety check
            func_call_copy = func_call
            if "(" in func_call_copy:
                func_call_copy = func_call_copy.split("(")[0]
            if "." in func_call_copy:
                func_call_copy = func_call_copy.split(".")[1]
            
            dangerous_funcs = ["kill", "exit", "quit", "remove", "unlink", "popen", "Popen", "run", "__import__", "eval", "exec", "compile"]
            if func_call_copy in dangerous_funcs:
                raise Exception(f"Function call {func_call_copy} is not allowed.")

            # Execute in isolated namespace with only necessary instances
            func_call_result = eval(func_call, {"__builtins__": {}}, exec_namespace)
            
            # Convert result to string
            if isinstance(func_call_result, str):
                pass
            elif isinstance(func_call_result, dict):
                try:
                    func_call_result = json.dumps(func_call_result)
                except:
                    func_call_result = str(func_call_result)
            else:
                func_call_result = str(func_call_result)
            
            execution_results.append(func_call_result)
        except Exception as e:
            execution_results.append(f"Error during execution: {str(e)}")
    
    return execution_results, involved_instances


def _process_method_calls(function_call_string: str, instance_mapping: Dict) -> str:
    """
    Prepend instance name to function names in the call string.
    
    Args:
        function_call_string: Function call string
        instance_mapping: Mapping of method names to instance names
        
    Returns:
        Processed function call string
    """
    def replace_function(match):
        func_name = match.group(1)
        if func_name in instance_mapping:
            return f"{instance_mapping[func_name]}.{func_name}("
        return match.group(0)  # Return the full match if not in mapping
    
    # Regular expression to match function names
    pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    result = re.sub(pattern, replace_function, function_call_string)
    
    return result

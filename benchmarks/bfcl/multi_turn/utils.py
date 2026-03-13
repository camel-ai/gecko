"""
Multi-Turn Evaluation Utilities

Helper functions for multi-turn BFCL evaluation.
"""

from typing import Any, List, Tuple, Dict


def is_empty_execute_response(input_list: list) -> bool:
    """
    Check if execution response is empty.
    
    Args:
        input_list: List of execution responses
        
    Returns:
        True if the response is empty
    """
    if len(input_list) == 0:
        return True
    if len(input_list) == 1 and len(input_list[0]) == 0:
        return True
    return False


def compare_instances(model_object: Any, ground_truth_object: Any) -> Tuple[bool, Dict]:
    """
    Compare two instances to check if they have the same attributes.
    
    Args:
        model_object: Model instance
        ground_truth_object: Ground truth instance
        
    Returns:
        Tuple of (valid, differences)
    """
    assert type(model_object) == type(ground_truth_object), "Objects are not of the same type."
    
    differences = {}
    valid = True
    
    for attr_name in vars(ground_truth_object):
        # Skip private attributes
        if attr_name.startswith("_"):
            continue
            
        model_attr = getattr(model_object, attr_name)
        ground_truth_attr = getattr(ground_truth_object, attr_name)
        
        if model_attr != ground_truth_attr:
            valid = False
            differences[attr_name] = {
                "model": model_attr, 
                "ground_truth": ground_truth_attr
            }
    
    return valid, differences


def is_subsequence(list1: list, list2: list) -> Tuple[bool, list]:
    """
    Check if list1 is a subsequence of list2 (preserving order).
    
    Args:
        list1: Expected subsequence
        list2: List to check against
        
    Returns:
        Tuple of (is_subsequence, missing_items)
    """
    # Convert list2 to an iterator to ensure elements are consumed only once
    iter_list2 = iter(list2)
    missing = []
    
    for item in list1:
        if item not in list2:
            missing.append(item)
        else:
            # Try to find item in remaining elements
            found = False
            for elem in iter_list2:
                if elem == item:
                    found = True
                    break
            if not found:
                missing.append(item)
    
    return len(missing) == 0, missing


def is_subsequence_unordered(list1: list, list2: list) -> Tuple[bool, list]:
    """
    Check if all elements of list1 are present in list2, regardless of order.
    Handles duplicates correctly.
    
    Args:
        list1: Expected elements
        list2: List to check against
        
    Returns:
        Tuple of (is_subsequence, missing_items)
    """
    # Copy list2 to avoid modifying the original list
    list2_copy = list2[:]
    
    # Check each item in list1 to see if it exists in list2_copy
    missing_elements = []
    for item in list1:
        try:
            # Attempt to remove one occurrence of item from list2_copy
            list2_copy.remove(item)
        except ValueError:
            # If item is not found, add it to missing_elements
            missing_elements.append(item)
    
    # If there are missing elements, list1 is not a subsequence of list2
    is_subsequence = len(missing_elements) == 0
    return is_subsequence, missing_elements


def extract_function_calls_from_turn(turn_data: Dict) -> List[Dict]:
    """
    Extract function calls from a turn's retry data.
    
    Args:
        turn_data: Turn data containing retries
        
    Returns:
        List of function call dictionaries
    """
    tool_calls = []
    
    # Extract from retries
    if 'retries' in turn_data:
        for retry in turn_data['retries']:
            if 'tool_calls' in retry:
                for call in retry['tool_calls']:
                    # Convert to expected format
                    if 'function' in call and 'arguments' in call:
                        func_name = call['function']
                        args = call['arguments']
                        tool_calls.append({func_name: args})
    
    return tool_calls


def format_function_call(func_name: str, arguments: Dict) -> str:
    """
    Format a function call as a string for execution.
    
    Args:
        func_name: Function name
        arguments: Function arguments
        
    Returns:
        Formatted function call string
    """
    # Convert arguments to proper format
    arg_strings = []
    for key, value in arguments.items():
        if isinstance(value, str):
            arg_strings.append(f"{key}='{value}'")
        elif isinstance(value, dict):
            # Handle nested dicts (like requestBody)
            if 'requestBody' in arguments and key == 'requestBody':
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str):
                        arg_strings.append(f"{sub_key}='{sub_value}'")
                    else:
                        arg_strings.append(f"{sub_key}={sub_value}")
            else:
                arg_strings.append(f"{key}={value}")
        else:
            arg_strings.append(f"{key}={value}")
    
    return f"{func_name}({', '.join(arg_strings)})"


def parse_ground_truth_function_call(func_call_str: str) -> Tuple[str, Dict]:
    """
    Parse a ground truth function call string into function name and arguments.
    
    Args:
        func_call_str: Function call string like "cd(folder='documents')"
        
    Returns:
        Tuple of (function_name, arguments_dict)
    """
    import re
    
    # Extract function name
    match = re.match(r'(\w+)\((.*)\)', func_call_str)
    if not match:
        return None, None
    
    func_name = match.group(1)
    args_str = match.group(2)
    
    # Parse arguments
    arguments = {}
    if args_str:
        # Simple argument parsing (handles basic cases)
        # This is a simplified parser - may need enhancement for complex cases
        arg_pairs = re.findall(r"(\w+)=(['\"]?)([^,'\"]*)\2", args_str)
        for key, _, value in arg_pairs:
            # Try to convert to appropriate type
            if value.isdigit():
                arguments[key] = int(value)
            elif value in ('True', 'False'):
                arguments[key] = value == 'True'
            else:
                arguments[key] = value
    
    return func_name, arguments
# BFCL Single Turn Evaluation Utils
"""
Utility functions for BFCL single-turn evaluation.
Adapted from official BFCL evaluation code at:
https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
"""

import re
import ast
import json
from typing import List, Dict, Any


def is_function_calling_format_output(decoded_output):
    """
    Ensure the output is a list of dictionaries of the form:
    `[{func1: {param1: val1, param2: val2, ...}}, {func2: {param1: val1, param2: val2, ...}}, ...]`
    Sometimes the model handler's `decode_ast` method will return successfully, but the output is not in the correct format, and that will mess up the downstream evaluation that expects this format.
    This is especially the case when the model doesn't predict any function calls, and the output is an human-readable string.
    Note: Empty list `[]` is considered the correct format in this check.
    """
    if type(decoded_output) != list:
        return False
    for item in decoded_output:
        if type(item) != dict:
            return False
        # Check for `{func1: {param1: val1, param2: val2, ...}}`, should only have one key-value pair
        if len(item) != 1:
            return False
        # Check for `{param1: val1, param2: val2, ...}`; the parameter-value pairs should be a dictionary
        if type(list(item.values())[0]) != dict:
            return False
    return True


def is_empty_output(decoded_output):
    """
    This function is a patch to the ast decoder for relevance detection
    Sometimes the ast decoder will parse successfully, but the input doens't really have a function call
    [], [{}], and anything that is not in function calling format is considered empty (and thus should be marked as correct)
    """
    if not is_function_calling_format_output(decoded_output):
        return True
    if len(decoded_output) == 0:
        return True
    if len(decoded_output) == 1 and len(decoded_output[0]) == 0:
        return True
    return False


def standardize_string(input_string: str):
    """
    This function standardizes the string by removing all the spaces, ",./-_*^" punctuation, and converting it to lowercase
    It will also convert all the single quotes to double quotes
    This is used to compare the model output with the possible answers
    We don't want to punish model for answer like April 1, 2024 vs April 1,2024, vs April 1 2024
    """
    regex_string = r"[ \,\.\/\-\_\*\^]"
    return re.sub(regex_string, "", input_string).lower().replace("'", '"')


def get_possible_answer_type(possible_answer: list):
    """Get the type of the first non-empty value in possible_answer."""
    for answer in possible_answer:
        if answer != "":  # Optional parameter
            return type(answer)
    return None


def find_description(func_descriptions, name):
    """Find function description by name."""
    if type(func_descriptions) == list:
        for func_description in func_descriptions:
            if func_description["name"] == name:
                return func_description
        return None
    else:
        # it is a dict, there is only one function
        return func_descriptions


def convert_func_name(function_name, model_name: str):
    """
    Convert function name for models that don't support dots in function names.
    OAI does not support "." in the function name so we replace it with "_". ^[a-zA-Z0-9_-]{1,64}$ is the regex for the name.
    This happens for OpenAI, Mistral, and Google models
    """
    # For our implementation, we'll make this configurable
    # Currently we don't have MODEL_CONFIG_MAPPING, so we'll use a simple heuristic
    if "." in function_name:
        # Common models that need underscore conversion
        underscore_models = ["gpt", "openai", "mistral", "google", "gemini"]
        if any(model in model_name.lower() for model in underscore_models):
            return re.sub(r"\.", "_", function_name)
    return function_name


def convert_to_function_call(function_call_list) -> List[str]:
    """
    Convert function call list to execution format.

    Args:
        function_call_list: List of function calls in dict format or single dict

    Returns:
        List of function calls in string execution format
    """
    if isinstance(function_call_list, dict):
        function_call_list = [function_call_list]

    execution_list = []
    for function_call in function_call_list:
        for key, value in function_call.items():
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except:
                    value = {}
            execution_list.append(
                f"{key}({','.join([f'{k}={repr(v)}' for k, v in value.items()])})"
            )

    return execution_list


def resolve_ast_by_type(value):
    """Resolve AST value to Python object."""
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(value, ast.NameConstant):
        output = value.value
    elif isinstance(value, ast.BinOp):
        output = eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    else:
        raise NotImplementedError(f"Unsupported AST type: {type(value)}")
    return output


def resolve_ast_call(elem: ast.Call) -> Dict:
    """
    Resolve an AST Call node to function call format.

    Args:
        elem: AST Call node

    Returns:
        Dictionary with function name and arguments
    """
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))

    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output

    return {func_name: args_dict}


def ast_parse(input_str: str, language: str = "Python") -> List[Dict]:
    """
    Parse input string to AST format for function calls.

    Args:
        input_str: String to parse
        language: Programming language (Python/Java/JavaScript)

    Returns:
        List of parsed function calls
    """
    if language == "Python":
        cleaned_input = input_str.strip("[]'")
        parsed = ast.parse(cleaned_input, mode="eval")
        extracted = []
        if isinstance(parsed.body, ast.Call):
            extracted.append(resolve_ast_call(parsed.body))
        else:
            for elem in parsed.body.elts:
                assert isinstance(elem, ast.Call)
                extracted.append(resolve_ast_call(elem))
        return extracted
    elif language in ["Java", "JavaScript"]:
        # For simplicity, assume string format for non-Python languages
        try:
            return json.loads(input_str)
        except json.JSONDecodeError:
            raise ValueError(f"Cannot parse {language} function calls: {input_str}")
    else:
        raise NotImplementedError(f"Unsupported language: {language}")
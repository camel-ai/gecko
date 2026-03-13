"""
BFCL Common Utilities

Shared utility functions for BFCL benchmark operations that are used
across different modules (single_turn, multi_turn, evaluation, etc.).
"""

import os
import re
from typing import Union


def extract_numeric_id(test_id: str) -> int:
    """
    Extract the primary numeric identifier from BFCL test IDs for proper sorting.
    
    BFCL test IDs follow patterns like:
    - simple_0, simple_1, simple_10, simple_100
    - multiple_3-0-7, multiple_15-2-1  
    - live_simple_0-0-0, live_multiple_125-11-0
    
    For complex IDs with dashes, extracts the first number after the category.
    For simple IDs, extracts the number after the underscore.
    
    Args:
        test_id: BFCL test ID string
        
    Returns:
        int: Primary numeric identifier for sorting
        
    Examples:
        >>> extract_numeric_id("simple_0")
        0
        >>> extract_numeric_id("simple_267")
        267
        >>> extract_numeric_id("multiple_3-0-7")
        3
        >>> extract_numeric_id("live_simple_125-11-0")
        125
    """
    # Handle live test cases (e.g., "live_simple_125-11-0" -> "simple_125-11-0")
    if test_id.startswith("live_"):
        test_id = test_id[5:]  # Remove "live_" prefix
    
    # Split by underscore to separate category and numeric part
    parts = test_id.split("_", 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid BFCL test ID format: {test_id}")
    
    numeric_part = parts[1]  # e.g., "267" or "3-0-7"
    
    # Extract first number (handles both simple and complex cases)
    match = re.match(r"(\d+)", numeric_part)
    if not match:
        raise ValueError(f"No numeric ID found in test ID: {test_id}")
    
    return int(match.group(1))


def sort_bfcl_test_ids(test_ids: list) -> list:
    """
    Sort BFCL test IDs numerically by their primary identifier.
    
    Args:
        test_ids: List of BFCL test ID strings
        
    Returns:
        list: Sorted list of test IDs
        
    Examples:
        >>> sort_bfcl_test_ids(["simple_10", "simple_2", "simple_1"])
        ["simple_1", "simple_2", "simple_10"]
        >>> sort_bfcl_test_ids(["multiple_15-0-0", "multiple_3-1-2", "multiple_100-0-1"])
        ["multiple_3-1-2", "multiple_15-0-0", "multiple_100-0-1"]
    """
    return sorted(test_ids, key=extract_numeric_id)


def derive_single_turn_schema_name(test_id: str) -> str:
    """
    Derive the compact single-turn schema/toolkit name used by generated OpenAPI specs.

    Examples:
        live_multiple_1050-277-0 -> LM1050
        live_simple_0-0-0 -> LS0
        simple_python_5 -> SP5
    """
    patterns = [
        (r"^live_parallel_multiple_(\d+)-", "LPM"),
        (r"^live_multiple_(\d+)-", "LM"),
        (r"^live_parallel_(\d+)-", "LP"),
        (r"^live_simple_(\d+)-", "LS"),
        (r"^live_irrelevance_(\d+)-", "LI"),
        (r"^live_relevance_(\d+)-", "LR"),
        (r"^parallel_multiple_(\d+)$", "PM"),
        (r"^parallel_(\d+)$", "P"),
        (r"^multiple_(\d+)$", "M"),
        (r"^simple_python_(\d+)$", "SP"),
        (r"^simple_javascript_(\d+)$", "SJ"),
        (r"^simple_java_(\d+)$", "SJA"),
        (r"^irrelevance_(\d+)$", "IR"),
        (r"^format_sensitivity_(\d+)$", "FS"),
        (r"^memory_(\d+)$", "MEM"),
        (r"^web_search_(\d+)$", "WS"),
    ]
    for pattern, prefix in patterns:
        match = re.match(pattern, test_id)
        if match:
            return f"{prefix}{match.group(1)}"
    return test_id


def compress_single_turn_function_name(function_name: str) -> str:
    """
    Compress BFCL single-turn function names for runtime tool naming.

    If the original name has more than two dot-separated segments, keep only the
    last two segments. Otherwise keep the original name.
    """
    parts = [part for part in str(function_name).split(".") if part]
    if len(parts) > 2:
        return ".".join(parts[-2:])
    return str(function_name)


def sanitize_single_turn_function_name(function_name: str) -> str:
    """
    Convert a compressed single-turn function name into the endpoint/tool form
    used in generated OpenAPI specs.
    """
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", str(function_name).strip())
    value = re.sub(r"_+", "_", value).strip("_")
    if value and value[0].isdigit():
        value = f"_{value}"
    return value


def derive_single_turn_endpoint_name(function_name: str) -> str:
    """
    Endpoint/operation name used inside the generated per-task OpenAPI spec.
    """
    return sanitize_single_turn_function_name(
        compress_single_turn_function_name(function_name)
    )


def derive_single_turn_runtime_function_name(test_id: str, function_name: str) -> str:
    """
    Runtime tool name as seen by the agent after OpenAPI toolkit loading.
    """
    schema_name = derive_single_turn_schema_name(test_id)
    endpoint_name = derive_single_turn_endpoint_name(function_name)
    return f"{schema_name}_{endpoint_name}"


def extract_test_category_from_id(test_entry_id: str, remove_prereq: bool = False) -> str:
    """
    Extract the test category from a BFCL test entry ID.
    """
    if remove_prereq:
        test_entry_id = test_entry_id.replace("_prereq", "")
    if ":" in test_entry_id:
        test_entry_id = test_entry_id.split(":")[0]
    return test_entry_id.rsplit("_", 1)[0]


def is_format_sensitivity(test_category: str) -> bool:
    return "format_sensitivity" in test_category


def is_web_search(test_category: str) -> bool:
    return "web_search" in test_category


def is_memory(test_category: str) -> bool:
    return "memory" in test_category


def is_first_memory_prereq_entry(test_entry_id: str) -> bool:
    return "prereq" in test_entry_id and test_entry_id.endswith("-0")


def is_memory_prereq(test_category: str) -> bool:
    return "prereq" in test_category


def is_agentic(test_category: str) -> bool:
    return is_web_search(test_category) or is_memory(test_category)


def is_multi_turn(test_category: str) -> bool:
    return "multi_turn" in test_category


def is_live(test_category: str) -> bool:
    return "live" in test_category


def is_non_live(test_category: str) -> bool:
    return not any(
        (
            is_format_sensitivity(test_category),
            is_live(test_category),
            is_multi_turn(test_category),
            is_agentic(test_category),
        )
    )


def extract_memory_backend_type(test_category: str) -> str:
    if not is_memory(test_category):
        raise ValueError(f"Test category {test_category} is not a memory category.")
    return test_category[len("memory_") :]


def get_general_grouping(test_id: str) -> str:
    if is_format_sensitivity(test_id):
        return "format_sensitivity"
    if is_non_live(test_id):
        return "non_live"
    if is_live(test_id):
        return "live"
    if is_multi_turn(test_id):
        return "multi_turn"
    if is_agentic(test_id):
        return "agentic"
    raise ValueError(f"Invalid test category: {test_id}")


def get_directory_structure_by_id(test_id: str) -> str:
    """
    Map a BFCL test entry id to the result directory structure used by agentic tests.
    """
    group = get_general_grouping(test_id)
    if is_memory(test_id):
        return os.path.join(
            group,
            "memory",
            extract_memory_backend_type(
                extract_test_category_from_id(test_id, remove_prereq=True)
            ),
        )
    return group

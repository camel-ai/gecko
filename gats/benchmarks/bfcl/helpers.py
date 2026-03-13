import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from benchmarks.bfcl.utils import derive_single_turn_schema_name
from gats.core.task import GATSTask, GATSResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema resolution
# ---------------------------------------------------------------------------

# Resolve data paths relative to the project root (gecko repo root).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCHEMA_BASE_DIR = os.path.join(_PROJECT_ROOT, "data", "openapi")

# Multi-turn class name -> schema file mapping
MULTI_TURN_SCHEMA_MAP = {
    "GorillaFileSystem": "GorillaFileSystem.json",
    "TwitterAPI": "TwitterAPI.json",
    "MathAPI": "MathAPI.json",
    "MessageAPI": "MessageAPI.json",
    "PostingAPI": "TwitterAPI.json",
    "TicketAPI": "TicketAPI.json",
    "TradingBot": "TradingBot.json",
    "TravelAPI": "TravelAPI.json",
    "TravelBooking": "TravelAPI.json",
    "VehicleControl": "VehicleControlAPI.json",
    "VehicleControlAPI": "VehicleControlAPI.json",
}


def resolve_single_turn_schemas(test_id: str) -> List[str]:
    """Resolve OpenAPI schema path(s) for a single-turn BFCL test."""
    single_turn_dir = os.path.join(SCHEMA_BASE_DIR, "single_turn")

    # Preferred: compact per-task spec (e.g. SP5.json, LM1050.json)
    compact_path = os.path.join(
        single_turn_dir, f"{derive_single_turn_schema_name(test_id)}.json"
    )
    if os.path.exists(compact_path):
        return [compact_path]

    # Fallback: legacy layout by test_id_{idx}.json
    def _candidates(base_id: str) -> List[str]:
        cands = [base_id]
        if base_id.startswith(("simple_python_", "simple_java_", "simple_javascript_")):
            suffix = base_id.split("_", 2)[-1]
            cands.append(f"simple_{suffix}")
        return cands

    for cand_id in _candidates(test_id):
        paths = []
        for i in range(10):
            found = False
            for base in (SCHEMA_BASE_DIR, single_turn_dir):
                p = os.path.join(base, f"{cand_id}_{i}.json")
                if os.path.exists(p):
                    paths.append(p)
                    found = True
                    break
            if not found:
                break
        if paths:
            return paths

    raise FileNotFoundError(
        f"No OpenAPI schema found for single-turn test {test_id} "
        f"(tried {compact_path} and legacy layouts)"
    )


def resolve_multi_turn_schemas(involved_classes: List[str], test_id: str) -> List[str]:
    """Resolve OpenAPI schema paths for a multi-turn BFCL test."""
    if not involved_classes:
        raise ValueError(f"Multi-turn test {test_id} has no involved_classes")

    paths = []
    for cls_name in involved_classes:
        schema_filename = MULTI_TURN_SCHEMA_MAP.get(cls_name)
        if not schema_filename:
            raise ValueError(
                f"Unknown involved class '{cls_name}' for test {test_id}; "
                "add it to MULTI_TURN_SCHEMA_MAP"
            )
        schema_path = os.path.join(SCHEMA_BASE_DIR, "multi_turn", schema_filename)
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema for {cls_name} not found: {schema_path}")
        paths.append(schema_path)
    return paths


# ---------------------------------------------------------------------------
# System message & question extraction
# ---------------------------------------------------------------------------


def extract_question_and_system_messages(
    question_data: Any,
) -> Tuple[str, List[str]]:
    """Extract user question text and system messages from BFCL question data.

    BFCL entries may contain multi-role messages:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

    Returns:
        (question_text, system_messages)
    """
    if isinstance(question_data, str):
        return question_data, []

    question_text = ""
    system_messages: List[str] = []

    if isinstance(question_data, list):
        messages = question_data
        # List-of-lists means multi-turn -- take first turn for single-turn use.
        if messages and isinstance(messages[0], list):
            messages = messages[0]

        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system" and content:
                    system_messages.append(content)
                elif role == "user" and content:
                    question_text = content
                elif not role and content:
                    question_text = content
            elif isinstance(msg, str):
                question_text = msg

        # Fallback
        if not question_text and messages:
            first = messages[0]
            if isinstance(first, dict):
                question_text = first.get("content", str(first))
            else:
                question_text = str(first)

    return question_text, system_messages


def extract_multi_turn_questions(
    question_data: Any,
) -> List[Tuple[str, List[str]]]:
    """Extract per-turn (question, system_messages) from BFCL multi-turn data.

    Returns list of (question_text, system_messages) tuples, one per turn.
    """
    if not isinstance(question_data, list):
        return [(str(question_data), [])]

    results: List[Tuple[str, List[str]]] = []
    for turn_data in question_data:
        if isinstance(turn_data, list):
            question_text = ""
            sys_msgs: List[str] = []
            for msg in turn_data:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system" and content:
                        sys_msgs.append(content)
                    elif role == "user" and content:
                        question_text = content
                elif isinstance(msg, str):
                    question_text = msg
            if not question_text and turn_data:
                first = turn_data[0]
                question_text = (
                    first.get("content", str(first))
                    if isinstance(first, dict)
                    else str(first)
                )
            results.append((question_text, sys_msgs))
        elif isinstance(turn_data, str):
            results.append((turn_data, []))
        else:
            results.append((str(turn_data), []))

    return results


def build_agent_prompt_with_system_messages(
    base_prompt: Optional[str],
    system_messages: List[str],
) -> Optional[str]:
    """Build effective agent system prompt by appending task-specific system messages."""
    if not system_messages:
        return base_prompt
    parts = []
    if base_prompt:
        parts.append(base_prompt)
    parts.extend(system_messages)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

BFCL_TASK_DIR = os.path.join(_PROJECT_ROOT, "data", "bfcl_v4", "task")


def load_single_turn_tasks(
    category: str,
    base_agent_prompt: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[GATSTask]:
    """Load BFCL single-turn test cases as GATSTasks.

    System messages (found in live_* categories) are merged into each
    task's ``agent_prompt`` so the solver picks them up automatically.
    """
    task_file = os.path.join(BFCL_TASK_DIR, f"BFCL_v4_{category}.json")
    if not os.path.exists(task_file):
        raise FileNotFoundError(f"BFCL task file not found: {task_file}")

    tasks: List[GATSTask] = []
    with open(task_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if limit and line_idx >= limit:
                break
            if not line.strip():
                continue
            data = json.loads(line)

            raw_id = str(data.get("id", f"{category}_{line_idx}"))
            test_category = data.get("test_category", category)
            initial_config = data.get("initial_config", {})
            involved_classes = data.get("involved_classes", [])

            # Extract question and task-level system messages
            question_text, system_messages = extract_question_and_system_messages(
                data.get("question", [])
            )

            # Resolve schema (skip tasks with missing schemas)
            try:
                tool_schemas = resolve_single_turn_schemas(raw_id)
            except FileNotFoundError:
                logger.debug(f"Skipping {raw_id}: no OpenAPI schema found")
                continue

            # Build per-task agent prompt (base + system messages)
            effective_prompt = build_agent_prompt_with_system_messages(
                base_agent_prompt, system_messages
            )

            tasks.append(
                GATSTask(
                    id=raw_id,
                    turns=[question_text],
                    tool_schemas=tool_schemas,
                    initial_config=initial_config,
                    metadata={
                        "category": test_category,
                        "involved_classes": involved_classes,
                    },
                    agent_prompt=effective_prompt,
                )
            )

    return tasks


def load_multi_turn_tasks(
    category: str,
    base_agent_prompt: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[GATSTask]:
    """Load BFCL multi-turn test cases as GATSTasks."""
    task_file = os.path.join(BFCL_TASK_DIR, f"BFCL_v4_{category}.json")
    if not os.path.exists(task_file):
        raise FileNotFoundError(f"BFCL task file not found: {task_file}")

    tasks: List[GATSTask] = []
    with open(task_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if limit and line_idx >= limit:
                break
            if not line.strip():
                continue
            data = json.loads(line)

            raw_id = str(data.get("id", f"{category}_{line_idx}"))
            test_category = data.get("test_category", category)
            initial_config = data.get("initial_config", {})
            involved_classes = data.get("involved_classes", [])

            # Resolve schemas via alias map
            tool_schemas = resolve_multi_turn_schemas(involved_classes, raw_id)

            # Extract per-turn questions and system messages
            turn_info = extract_multi_turn_questions(data.get("question", []))
            turns = [q for q, _ in turn_info]

            # For multi-turn, combine all system messages from all turns
            all_system_messages = []
            for _, sys_msgs in turn_info:
                all_system_messages.extend(sys_msgs)

            effective_prompt = build_agent_prompt_with_system_messages(
                base_agent_prompt, all_system_messages
            )

            tasks.append(
                GATSTask(
                    id=raw_id,
                    turns=turns,
                    tool_schemas=tool_schemas,
                    initial_config=initial_config,
                    metadata={
                        "category": test_category,
                        "involved_classes": involved_classes,
                    },
                    agent_prompt=effective_prompt,
                )
            )

    return tasks


# ---------------------------------------------------------------------------
# Task selection helpers
# ---------------------------------------------------------------------------


def resolve_test_ids(
    category: str,
    ids: Optional[str] = None,
    ids_file: Optional[str] = None,
    pattern: Optional[str] = None,
    run_all: bool = False,
) -> Optional[List[str]]:
    """Resolve CLI arguments to a list of test IDs, or None for 'all'.

    Supports short numeric IDs with auto-prefix: ``--ids 0,1,2 --category simple_python``
    becomes ``["simple_python_0", "simple_python_1", "simple_python_2"]``.
    """
    if run_all:
        return None  # Caller should load all tasks

    if ids_file:
        with open(ids_file, "r") as f:
            return [line.strip() for line in f if line.strip()]

    if pattern:
        # Return as-is; caller will filter tasks
        return None  # pattern filtering is done later

    if ids:
        processed: List[str] = []
        for raw in ids.split(","):
            item = raw.strip()
            if not item:
                continue
            needs_prefix = (
                category
                and not item.startswith(category)
                and ("_" not in item or item.replace("_", "").isdigit())
            )
            processed.append(f"{category}_{item}" if needs_prefix else item)
        return processed

    return None


def filter_tasks_by_ids(
    tasks: List[GATSTask],
    ids: Optional[List[str]] = None,
    pattern: Optional[str] = None,
) -> List[GATSTask]:
    """Filter loaded tasks by IDs or regex pattern."""
    if ids is not None:
        id_set = set(ids)
        return [t for t in tasks if t.id in id_set]
    if pattern:
        regex = re.compile(pattern)
        return [t for t in tasks if regex.search(t.id)]
    return tasks


# ---------------------------------------------------------------------------
# Tool call filtering (single-turn)
# ---------------------------------------------------------------------------


def _extract_tool_call_error(result: Any) -> Optional[str]:
    """Return error string from a tool call result if it indicates failure."""
    if isinstance(result, dict):
        error = result.get("error")
        if error:
            return str(error)
        detail = result.get("detail")
        if isinstance(detail, dict):
            err_msg = detail.get("error_message")
            if err_msg:
                return str(err_msg)
        elif isinstance(detail, str) and detail:
            return detail
        if result.get("success") is False:
            return str(result.get("message") or "success=false")
    if isinstance(result, str) and "error" in result.lower():
        return result
    return None


def filter_failed_tool_calls(
    tool_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop failed tool calls from single-turn results for BFCL eval."""
    return [
        tc
        for tc in (tool_calls or [])
        if isinstance(tc, dict) and not _extract_tool_call_error(tc.get("result"))
    ]


# ---------------------------------------------------------------------------
# BFCL eval format conversion
# ---------------------------------------------------------------------------


def format_result_for_bfcl_eval(
    result: GATSResult,
    is_multi: bool = False,
    benchmark=None,
) -> Dict[str, Any]:
    """Convert GATSResult to BFCL official eval JSONL format.

    Args:
        result: The GATSResult to format.
        is_multi: Whether this is a multi-turn result.
        benchmark: Optional BFCLBenchmark instance for function name mapping.

    Returns:
        Dict ready to be serialized as one JSONL line for bfcl_evaluate.py.
    """
    if is_multi:
        return _format_multi_turn_eval(result, benchmark)
    else:
        return _format_single_turn_eval(result, benchmark)


def _format_single_turn_eval(
    result: GATSResult, benchmark=None
) -> Dict[str, Any]:
    """Single-turn BFCL eval: {"id": ..., "result": [{"func": "args_json"}], ...}"""
    calls: List[Dict[str, Any]] = []
    if result.turns:
        turn = result.turns[0]
        if 0 <= turn.best_attempt < len(turn.attempts):
            calls = filter_failed_tool_calls(
                turn.attempts[turn.best_attempt].tool_calls
            )

    bfcl_result: List[Dict[str, str]] = []
    for tc in calls:
        raw_name = tc.get("name", "") or tc.get("function", "")
        # Map runtime tool name back to original BFCL function name
        if benchmark and hasattr(benchmark, "map_tool_call_to_original_function"):
            func_name = benchmark.map_tool_call_to_original_function(
                result.task_id, raw_name
            )
        else:
            func_name = raw_name

        arguments = tc.get("arguments", {})
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                if isinstance(parsed, dict) and "requestBody" in parsed:
                    args_json = json.dumps(parsed["requestBody"])
                else:
                    args_json = arguments
            except (json.JSONDecodeError, TypeError):
                args_json = arguments
        elif isinstance(arguments, dict):
            if "requestBody" in arguments:
                args_json = json.dumps(arguments["requestBody"])
            else:
                args_json = json.dumps(arguments)
        else:
            args_json = json.dumps(arguments)

        bfcl_result.append({func_name: args_json})

    return {
        "id": result.task_id,
        "result": bfcl_result,
        "input_token_count": 0,
        "output_token_count": 0,
        "latency": result.total_time,
        "reasoning_content": "",
    }


def _format_tool_call_as_string(
    tc: Dict[str, Any], test_id: str, benchmark=None
) -> str:
    """Format a single tool call as ``func_name(arg=val, ...)`` for multi-turn eval."""
    raw_name = tc.get("name", "") or tc.get("function", "")
    if benchmark and hasattr(benchmark, "map_tool_call_to_original_function"):
        func_name = benchmark.map_tool_call_to_original_function(test_id, raw_name)
    else:
        func_name = raw_name

    arguments = tc.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return f"{func_name}({arguments})"

    if isinstance(arguments, dict):
        param_strs = []
        for key, value in arguments.items():
            if key == "requestBody" and isinstance(value, dict):
                for k, v in value.items():
                    param_strs.append(f"{k}={repr(v)}")
            else:
                param_strs.append(f"{key}={repr(value)}")
        return f"{func_name}({', '.join(param_strs)})"
    return f"{func_name}({arguments})"


def _format_multi_turn_eval(
    result: GATSResult, benchmark=None
) -> Dict[str, Any]:
    """Multi-turn BFCL eval: {"id": ..., "result": [["func(a=1)"], ...], ...}"""
    all_turn_calls: List[List[str]] = []
    for turn in result.turns:
        calls: List[Dict[str, Any]] = []
        if 0 <= turn.best_attempt < len(turn.attempts):
            calls = turn.attempts[turn.best_attempt].tool_calls

        formatted = [
            _format_tool_call_as_string(tc, result.task_id, benchmark)
            for tc in calls
        ]
        all_turn_calls.append(formatted)

    return {
        "id": result.task_id,
        "result": all_turn_calls,
        "input_token_count": 0,
        "output_token_count": 0,
        "latency": result.total_time,
        "reasoning_content": "",
    }


def append_bfcl_eval_line(
    result: GATSResult,
    eval_file: str,
    is_multi: bool = False,
    benchmark=None,
) -> None:
    """Append one result as a BFCL eval JSONL line (incremental saving)."""
    entry = format_result_for_bfcl_eval(result, is_multi=is_multi, benchmark=benchmark)
    os.makedirs(os.path.dirname(eval_file) or ".", exist_ok=True)
    with open(eval_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

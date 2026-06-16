import glob
import json
import logging
import os
import re
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from benchmarks.bfcl.utils import derive_single_turn_schema_name
from gats.core.task import GATSTask, GATSResult
from utils.openapi_toolkit_fix import _normalize_kwargs_for_export

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema resolution
# ---------------------------------------------------------------------------

# Resolve data paths relative to the project root (gecko repo root).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCHEMA_BASE_DIR = os.path.join(_PROJECT_ROOT, "data", "bfcl", "openapi")
DEFAULT_MULTI_TURN_SCHEMA_DIR = os.path.join(SCHEMA_BASE_DIR, "multi_turn")
SchemaDirs = Optional[Union[str, Sequence[str]]]

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

# Lazy cache: {(schema_dir, ...): {func_name: {param_name: schema_type}}}
_MULTI_TURN_PARAM_TYPES: Dict[Tuple[str, ...], Dict[str, Dict[str, str]]] = {}
_SINGLE_TURN_PARAM_SCHEMAS: Dict[str, Dict[str, Dict[str, Any]]] = {}
_SINGLE_TURN_REQUIRED_PARAMS: Dict[str, Dict[str, set]] = {}
_SCHEMA_FALLBACK_WARNED: set = set()


def _expand_multi_turn_schema_dir(schema_dir: str) -> str:
    path = os.path.abspath(os.path.expandvars(os.path.expanduser(schema_dir)))
    nested = os.path.join(path, "multi_turn")
    if os.path.isdir(nested):
        return nested
    return path


def normalize_multi_turn_schema_dirs(schema_dirs: SchemaDirs = None) -> List[str]:
    """Return prioritized multi-turn schema directories with the default fallback.

    The runner can point at a partial candidate directory; schemas not present
    there fall back to the default repository schemas.
    """
    if schema_dirs is None:
        raw_dirs: List[str] = [DEFAULT_MULTI_TURN_SCHEMA_DIR]
    elif isinstance(schema_dirs, (str, os.PathLike)):
        raw_dirs = [os.fspath(schema_dirs)]
    else:
        raw_dirs = [os.fspath(item) for item in schema_dirs if item]

    normalized: List[str] = []
    for raw_dir in raw_dirs:
        path = _expand_multi_turn_schema_dir(raw_dir)
        if path not in normalized:
            normalized.append(path)

    default_dir = _expand_multi_turn_schema_dir(DEFAULT_MULTI_TURN_SCHEMA_DIR)
    if default_dir not in normalized:
        normalized.append(default_dir)
    return normalized


def _get_multi_turn_param_types(schema_dirs: SchemaDirs = None) -> Dict[str, Dict[str, str]]:
    """Build and cache a mapping of {func_name: {param_name: type}} from multi-turn OpenAPI schemas."""
    dirs = normalize_multi_turn_schema_dirs(schema_dirs)
    cache_key = tuple(dirs)
    cached = _MULTI_TURN_PARAM_TYPES.get(cache_key)
    if cached is not None:
        return cached

    result: Dict[str, Dict[str, str]] = {}
    seen_filenames = set()
    for schema_dir in dirs:
        for schema_file in glob.glob(os.path.join(schema_dir, "*.json")):
            filename = os.path.basename(schema_file)
            if filename in seen_filenames:
                continue
            seen_filenames.add(filename)
            with open(schema_file) as f:
                schema = json.load(f)
            for path, methods in schema.get("paths", {}).items():
                for method, details in methods.items():
                    func_name = details.get("operationId", "")
                    if not func_name:
                        continue
                    param_types: Dict[str, str] = {}
                    body = details.get("requestBody", {})
                    if body:
                        content = body.get("content", {}).get("application/json", {}).get("schema", {})
                        for pname, pschema in content.get("properties", {}).items():
                            param_types[pname] = pschema.get("type", "")
                    for p in details.get("parameters", []):
                        param_types[p["name"]] = p.get("schema", {}).get("type", "")
                    result[func_name] = param_types

    _MULTI_TURN_PARAM_TYPES[cache_key] = result
    return result


def _resolve_multi_turn_schema_path(
    schema_filename: str,
    cls_name: str,
    test_id: str,
    schema_dirs: List[str],
    *,
    override_supplied: bool,
) -> str:
    for idx, schema_dir in enumerate(schema_dirs):
        schema_path = os.path.join(schema_dir, schema_filename)
        if not os.path.exists(schema_path):
            continue
        if override_supplied and idx > 0:
            warn_key = (tuple(schema_dirs), schema_filename)
            if warn_key not in _SCHEMA_FALLBACK_WARNED:
                _SCHEMA_FALLBACK_WARNED.add(warn_key)
                logger.info(
                    "Schema %s for %s not found in override dirs; using fallback %s",
                    schema_filename,
                    cls_name,
                    schema_path,
                )
        return schema_path

    raise FileNotFoundError(
        f"Schema for {cls_name} not found: {schema_filename} "
        f"(searched: {', '.join(schema_dirs)})"
    )


def _get_single_turn_param_schemas(test_id: str) -> Dict[str, Dict[str, Any]]:
    """Build and cache {operation_id: {param_name: param_schema}} for one single-turn task."""
    cached = _SINGLE_TURN_PARAM_SCHEMAS.get(test_id)
    if cached is not None:
        return cached

    param_schemas: Dict[str, Dict[str, Any]] = {}
    try:
        schema_path = resolve_single_turn_schemas(test_id)[0]
        with open(schema_path) as f:
            schema = json.load(f)
        for methods in schema.get("paths", {}).values():
            if not isinstance(methods, dict):
                continue
            for details in methods.values():
                if not isinstance(details, dict):
                    continue
                operation_id = details.get("operationId", "")
                if not operation_id:
                    continue
                body_schema = (
                    details.get("requestBody", {})
                    .get("content", {})
                    .get("application/json", {})
                    .get("schema", {})
                )
                properties = body_schema.get("properties", {})
                if isinstance(properties, dict):
                    param_schemas[operation_id] = properties
    except Exception as exc:
        logger.debug("Failed to load single-turn param schemas for %s: %s", test_id, exc)

    _SINGLE_TURN_PARAM_SCHEMAS[test_id] = param_schemas
    return param_schemas


def _lookup_param_schema(
    schemas: Dict[str, Dict[str, Any]], func_name: str
) -> Dict[str, Any]:
    """Look up param schema with BFCL dot→underscore fallback.

    BFCL function names use dot notation (e.g. ``investment.predictProfit``)
    while OpenAPI operationIds use underscores (``investment_predictProfit``).
    """
    result = schemas.get(func_name)
    if result:
        return result
    if '.' in func_name:
        result = schemas.get(func_name.replace('.', '_'))
        if result:
            return result
    return {}


def _get_single_turn_required_params(test_id: str) -> Dict[str, set]:
    """Build and cache {operation_id: set(required_param_names)} for one single-turn task."""
    cached = _SINGLE_TURN_REQUIRED_PARAMS.get(test_id)
    if cached is not None:
        return cached

    required_map: Dict[str, set] = {}
    try:
        schema_path = resolve_single_turn_schemas(test_id)[0]
        with open(schema_path) as f:
            schema = json.load(f)
        for methods in schema.get("paths", {}).values():
            if not isinstance(methods, dict):
                continue
            for details in methods.values():
                if not isinstance(details, dict):
                    continue
                operation_id = details.get("operationId", "")
                if not operation_id:
                    continue
                body_schema = (
                    details.get("requestBody", {})
                    .get("content", {})
                    .get("application/json", {})
                    .get("schema", {})
                )
                required_map[operation_id] = set(body_schema.get("required", []))
    except Exception:
        pass

    _SINGLE_TURN_REQUIRED_PARAMS[test_id] = required_map
    return required_map


def _fill_optional_defaults(
    args: Dict[str, Any],
    param_schemas: Dict[str, Any],
    required_params: set,
) -> Dict[str, Any]:
    """Fill schema defaults for optional params the agent omitted.

    Only fills when the schema has an explicit ``default`` field.
    Returns a new dict (does not mutate *args*).
    """
    result = dict(args)
    for param_name, param_schema in param_schemas.items():
        if param_name in result:
            continue
        if param_name in required_params:
            continue
        if not isinstance(param_schema, dict):
            continue
        if "default" in param_schema:
            default_val = param_schema["default"]
            schema_type = param_schema.get("type")
            if isinstance(default_val, str) and schema_type:
                if schema_type == "boolean":
                    default_val = default_val.lower() not in ("false", "0", "")
                elif schema_type == "integer":
                    try:
                        default_val = int(default_val)
                    except (ValueError, TypeError):
                        pass
                elif schema_type == "number":
                    try:
                        default_val = float(default_val)
                    except (ValueError, TypeError):
                        pass
            result[param_name] = default_val
    return result


def _strip_dontcare_booleans(
    args: Dict[str, Any],
    param_schemas: Dict[str, Any],
) -> Dict[str, Any]:
    result = dict(args)
    for param_name, value in list(result.items()):
        if not isinstance(value, str) or value.lower() != "dontcare":
            continue
        schema = param_schemas.get(param_name)
        if not isinstance(schema, dict):
            continue
        enum = schema.get("enum")
        if not isinstance(enum, list):
            continue
        enum_lower = {str(v).lower() for v in enum}
        if enum_lower == {"true", "false", "dontcare"}:
            del result[param_name]
    return result


def _strip_empty_optional_values(
    value: Any,
    schema: Dict[str, Any],
    required: set,
) -> Any:
    """Drop empty placeholders for optional fields before BFCL export.

    BFCL single-turn ground truth treats omitted optional fields as equivalent
    to many empty/default placeholders. Exporting explicit empty optional keys
    can turn an otherwise correct call into a strict dict-key failure.
    """
    if not isinstance(schema, dict):
        return value

    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        nested_required = schema.get("required") or []
        nested_required_set = (
            set(nested_required) if isinstance(nested_required, list) else set()
        )
        result: Dict[str, Any] = {}
        for key, item in value.items():
            item_schema = properties.get(key, {}) if isinstance(properties, dict) else {}
            is_required = key in required or key in nested_required_set
            cleaned = _strip_empty_optional_values(
                item,
                item_schema if isinstance(item_schema, dict) else {},
                set(),
            )
            if not is_required and cleaned in ("", None, "?", [], {}):
                continue
            if (
                not is_required
                and _is_current_date_default_value(cleaned, item_schema)
            ):
                continue
            result[key] = cleaned
        return result

    if isinstance(value, list):
        item_schema = schema.get("items") if isinstance(schema.get("items"), dict) else {}
        return [
            _strip_empty_optional_values(item, item_schema, set())
            for item in value
        ]

    return value


def _is_current_date_default_value(value: Any, schema: Any) -> bool:
    if not isinstance(value, str) or not isinstance(schema, dict):
        return False
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False
    description = str(schema.get("description") or "").lower()
    if "current date" not in description and "today" not in description:
        return False
    default = schema.get("default")
    if default not in (None, "", "current", "today"):
        return False
    return value == date.today().isoformat()


def resolve_single_turn_schemas(test_id: str) -> List[str]:
    """Resolve OpenAPI schema path(s) for a single-turn BFCL test."""
    single_turn_dir = os.path.join(SCHEMA_BASE_DIR, "single_turn")

    # Preferred: compact per-task spec (e.g. SP5.json, LM1050.json)
    compact_path = os.path.join(
        single_turn_dir, f"{derive_single_turn_schema_name(test_id)}.json"
    )
    if os.path.exists(compact_path):
        return [compact_path]

    # Alternate layout by test_id_{idx}.json
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


def resolve_multi_turn_schemas(
    involved_classes: List[str],
    test_id: str,
    schema_dirs: SchemaDirs = None,
) -> List[str]:
    """Resolve OpenAPI schema paths for a multi-turn BFCL test."""
    if not involved_classes:
        raise ValueError(f"Multi-turn test {test_id} has no involved_classes")

    normalized_schema_dirs = normalize_multi_turn_schema_dirs(schema_dirs)
    override_supplied = schema_dirs is not None
    paths = []
    for cls_name in involved_classes:
        schema_filename = MULTI_TURN_SCHEMA_MAP.get(cls_name)
        if not schema_filename:
            raise ValueError(
                f"Unknown involved class '{cls_name}' for test {test_id}; "
                "add it to MULTI_TURN_SCHEMA_MAP"
            )
        schema_path = _resolve_multi_turn_schema_path(
            schema_filename,
            cls_name,
            test_id,
            normalized_schema_dirs,
            override_supplied=override_supplied,
        )
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

BFCL_TASK_DIR = os.path.join(_PROJECT_ROOT, "data", "bfcl", "task")
BFCL_ANSWER_DIR = os.path.join(_PROJECT_ROOT, "data", "bfcl", "possible_answer")


def _load_ground_truth_map(category: str) -> Dict[str, Any]:
    """Load ground truth answers keyed by test ID for a category."""
    answer_file = os.path.join(BFCL_ANSWER_DIR, f"BFCL_v4_{category}.json")
    if not os.path.exists(answer_file):
        return {}
    gt_map: Dict[str, Any] = {}
    try:
        with open(answer_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                tid = str(data.get("id", ""))
                gt = data.get("ground_truth")
                if tid and gt is not None:
                    gt_map[tid] = gt
    except Exception:
        logger.debug(f"Could not load ground truth from {answer_file}")
    return gt_map


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

    gt_map = _load_ground_truth_map(category)

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

            gt = gt_map.get(raw_id)
            if gt is None:
                for gt_id, gt_val in gt_map.items():
                    if gt_id == raw_id or gt_id.startswith(f"{raw_id}-"):
                        gt = gt_val
                        break

            metadata: Dict[str, Any] = {
                "category": test_category,
                "involved_classes": involved_classes,
            }
            if gt is not None:
                metadata["ground_truth"] = gt

            tasks.append(
                GATSTask(
                    id=raw_id,
                    turns=[question_text],
                    tool_schemas=tool_schemas,
                    initial_config=initial_config,
                    metadata=metadata,
                    agent_prompt=effective_prompt,
                )
            )

    return tasks


def load_multi_turn_tasks(
    category: str,
    base_agent_prompt: Optional[str] = None,
    limit: Optional[int] = None,
    schema_dirs: SchemaDirs = None,
) -> List[GATSTask]:
    """Load BFCL multi-turn test cases as GATSTasks."""
    task_file = os.path.join(BFCL_TASK_DIR, f"BFCL_v4_{category}.json")
    if not os.path.exists(task_file):
        raise FileNotFoundError(f"BFCL task file not found: {task_file}")

    normalized_schema_dirs = normalize_multi_turn_schema_dirs(schema_dirs)
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
            tool_schemas = resolve_multi_turn_schemas(
                involved_classes,
                raw_id,
                schema_dirs=normalized_schema_dirs,
            )

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
                        "multi_turn_schema_dirs": normalized_schema_dirs,
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
    if ids is not None:
        task_ids = {t.id for t in tasks}
        matched_ids = set()

        for requested_id in ids:
            if requested_id in task_ids:
                matched_ids.add(requested_id)
                continue

            for task_id in task_ids:
                if (
                    task_id.startswith(requested_id)
                    and len(task_id) > len(requested_id)
                    and task_id[len(requested_id)] in {"-", "_"}
                ):
                    matched_ids.add(task_id)

        return [t for t in tasks if t.id in matched_ids]
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


def _is_gecko_validation_error(result: Any) -> bool:
    if isinstance(result, dict):
        detail = result.get("detail")
        if isinstance(detail, dict) and detail.get("error_message"):
            return True
    return False


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
    multi_turn_schema_dirs: SchemaDirs = None,
) -> Dict[str, Any]:
    """Convert GATSResult to BFCL official eval JSONL format.

    Args:
        result: The GATSResult to format.
        is_multi: Whether this is a multi-turn result.
        benchmark: Optional BFCLBenchmark instance for function name mapping.

    Returns:
        Dict ready to be serialized as one JSONL line for the BFCL evaluator.
    """
    if is_multi:
        return _format_multi_turn_eval(result, benchmark, multi_turn_schema_dirs)
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
            raw_calls = turn.attempts[turn.best_attempt].tool_calls or []
            calls = [
                tc for tc in raw_calls
                if not _is_gecko_validation_error(tc.get("result"))
            ]

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
        export_arguments = tc.get("original_arguments", arguments)
        if isinstance(arguments, str):
            try:
                parsed = json.loads(export_arguments)
                if isinstance(parsed, dict) and "requestBody" in parsed:
                    params = _lookup_param_schema(_get_single_turn_param_schemas(result.task_id), func_name)
                    normalized_args = _normalize_kwargs_for_export(parsed["requestBody"], params)
                    args_json = json.dumps(normalized_args)
                else:
                    params = _lookup_param_schema(_get_single_turn_param_schemas(result.task_id), func_name)
                    normalized_args = _normalize_kwargs_for_export(parsed, params) if isinstance(parsed, dict) else parsed
                    args_json = json.dumps(normalized_args) if isinstance(normalized_args, dict) else export_arguments
            except (json.JSONDecodeError, TypeError):
                args_json = export_arguments
        elif isinstance(export_arguments, dict):
            if "requestBody" in export_arguments:
                params = _lookup_param_schema(_get_single_turn_param_schemas(result.task_id), func_name)
                args_json = json.dumps(_normalize_kwargs_for_export(export_arguments["requestBody"], params))
            else:
                params = _lookup_param_schema(_get_single_turn_param_schemas(result.task_id), func_name)
                args_json = json.dumps(_normalize_kwargs_for_export(export_arguments, params))
        else:
            args_json = json.dumps(export_arguments)

        # Post-process: fill schema defaults for omitted optional params
        try:
            parsed_args = json.loads(args_json)
            if isinstance(parsed_args, dict):
                params = _lookup_param_schema(
                    _get_single_turn_param_schemas(result.task_id), func_name
                )
                req_map = _get_single_turn_required_params(result.task_id)
                required = (
                    req_map.get(func_name)
                    or (req_map.get(func_name.replace('.', '_')) if '.' in func_name else None)
                    or set()
                )
                if params:
                    root_schema = {
                        "type": "object",
                        "properties": params,
                        "required": list(required),
                    }
                    pruned = _strip_empty_optional_values(
                        parsed_args,
                        root_schema,
                        required,
                    )
                    filled = _fill_optional_defaults(pruned, params, required)
                    filled = _strip_dontcare_booleans(filled, params)
                    args_json = json.dumps(filled)
        except (json.JSONDecodeError, TypeError):
            pass

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
    tc: Dict[str, Any],
    test_id: str,
    benchmark=None,
    multi_turn_schema_dirs: SchemaDirs = None,
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
        param_types = _get_multi_turn_param_types(multi_turn_schema_dirs).get(func_name, {})
        for key, value in arguments.items():
            if key == "requestBody" and isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, int) and not isinstance(v, bool) and param_types.get(k) == "number":
                        v = float(v)
                    param_strs.append(f"{k}={repr(v)}")
            else:
                if isinstance(value, int) and not isinstance(value, bool) and param_types.get(key) == "number":
                    value = float(value)
                param_strs.append(f"{key}={repr(value)}")
        return f"{func_name}({','.join(param_strs)})"
    return f"{func_name}({arguments})"


def _format_multi_turn_eval(
    result: GATSResult,
    benchmark=None,
    multi_turn_schema_dirs: SchemaDirs = None,
) -> Dict[str, Any]:
    """Multi-turn BFCL eval: {"id": ..., "result": [["func(a=1)"], ...], ...}"""
    all_turn_calls: List[List[str]] = []
    for turn in result.turns:
        if turn.real_tool_calls:
            calls = turn.real_tool_calls
        elif 0 <= turn.best_attempt < len(turn.attempts):
            calls = turn.attempts[turn.best_attempt].tool_calls
        else:
            calls = []

        formatted = [
            _format_tool_call_as_string(
                tc,
                result.task_id,
                benchmark,
                multi_turn_schema_dirs
                or result.metadata.get("multi_turn_schema_dirs"),
            )
            for tc in calls
        ]
        if len(formatted) > 1:
            seen_last = {}
            for i, s in enumerate(formatted):
                seen_last[s] = i
            formatted = [s for i, s in enumerate(formatted) if seen_last[s] == i]
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
    multi_turn_schema_dirs: SchemaDirs = None,
) -> None:
    """Append one result as a BFCL eval JSONL line (incremental saving)."""
    entry = format_result_for_bfcl_eval(
        result,
        is_multi=is_multi,
        benchmark=benchmark,
        multi_turn_schema_dirs=multi_turn_schema_dirs,
    )
    os.makedirs(os.path.dirname(eval_file) or ".", exist_ok=True)
    with open(eval_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

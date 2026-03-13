#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from dotenv import load_dotenv

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
for _p in (_SCRIPTS_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarks.bfcl.utils import (
    compress_single_turn_function_name,
    derive_single_turn_schema_name,
)
from tools_openapi_converter import ToolToOpenAPIConverter
from openapi_converter.openapi_utils import sanitize_openapi_schema

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_TASK_DIR = Path("data/bfcl_v4/task")
DEFAULT_OUTPUT_DIR = Path("data/openapi/single_turn")
CATEGORY_TO_FILE = {
    "simple_python": "BFCL_v4_simple_python.json",
    "simple_javascript": "BFCL_v4_simple_javascript.json",
    "simple_java": "BFCL_v4_simple_java.json",
    "multiple": "BFCL_v4_multiple.json",
    "parallel": "BFCL_v4_parallel.json",
    "parallel_multiple": "BFCL_v4_parallel_multiple.json",
    "irrelevance": "BFCL_v4_irrelevance.json",
    "format_sensitivity": "BFCL_v4_format_sensitivity.json",
    "memory": "BFCL_v4_memory.json",
    "live_simple": "BFCL_v4_live_simple.json",
    "live_multiple": "BFCL_v4_live_multiple.json",
    "live_parallel": "BFCL_v4_live_parallel.json",
    "live_parallel_multiple": "BFCL_v4_live_parallel_multiple.json",
    "live_irrelevance": "BFCL_v4_live_irrelevance.json",
    "live_relevance": "BFCL_v4_live_relevance.json",
    "web_search": "BFCL_v4_web_search.json",
}


def _parse_task_id_filter(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def _parse_task_id_file(path: str | None) -> set[str]:
    if not path:
        return set()
    ids: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value:
            ids.add(value)
    return ids


def _resolve_task_file(category: str | None, task_file: str | None) -> Path:
    if task_file:
        return Path(task_file)
    if not category:
        raise ValueError("Pass either --task-file or --category")
    try:
        filename = CATEGORY_TO_FILE[category]
    except KeyError as exc:
        supported = ", ".join(sorted(CATEGORY_TO_FILE))
        raise ValueError(f"Unsupported category '{category}'. Supported: {supported}") from exc
    return DEFAULT_TASK_DIR / filename


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object on line {lineno} of {path}")
            yield record


def _select_tasks(
    tasks: Sequence[Dict[str, Any]],
    task_ids: set[str],
    limit: int | None,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("id", "")).strip()
        if not task_id:
            continue
        if task_ids and task_id not in task_ids:
            continue
        selected.append(task)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _ensure_function_payload(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = task.get("function")
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Task {task.get('id')} has no non-empty 'function' list")
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Task {task.get('id')} function[{idx}] is not a JSON object")
    return payload


def _rewrite_function_names_for_bfcl(payload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rewritten: List[Dict[str, Any]] = []
    for item in payload:
        cloned = dict(item)
        raw_name = str(cloned.get("name", "")).strip()
        cloned["name"] = compress_single_turn_function_name(raw_name)
        rewritten.append(cloned)
    return rewritten


def _write_spec(output_dir: Path, task_id: str, spec: Dict[str, Any], schema_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    spec.setdefault("info", {})["title"] = schema_name
    spec["servers"] = [{"url": f"http://localhost:8000/{schema_name}"}]
    spec = sanitize_openapi_schema(spec)
    output_path = output_dir / f"{schema_name}.json"
    temp_path = output_dir / f".{schema_name}.json.tmp.{os.getpid()}"
    temp_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(output_path)
    return output_path


def _resolve_schema_name(task_id: str, keep_full_task_id_as_title: bool) -> str:
    return task_id if keep_full_task_id_as_title else derive_single_turn_schema_name(task_id)


def _resolve_output_path(output_dir: Path, schema_name: str) -> Path:
    return output_dir / f"{schema_name}.json"


def _generate_task_spec(
    task: Dict[str, Any],
    *,
    model_name: str,
    output_dir: Path,
    overwrite: bool,
    keep_full_task_id_as_title: bool,
) -> Tuple[str, str, Path | None, str | None]:
    task_id = str(task["id"])
    schema_name = _resolve_schema_name(task_id, keep_full_task_id_as_title)
    output_path = _resolve_output_path(output_dir, schema_name)
    if output_path.exists() and not overwrite:
        return ("skipped", task_id, output_path, None)

    try:
        payload = _rewrite_function_names_for_bfcl(_ensure_function_payload(task))
    except ValueError as exc:
        return ("skipped", task_id, None, str(exc))
    converter = ToolToOpenAPIConverter(model_name=model_name, include_default_state=False)
    spec = converter.convert(payload, api_name=schema_name)
    written = _write_spec(output_dir, task_id, spec, schema_name)
    return ("generated", task_id, written, None)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one OpenAPI 3.1 spec per BFCL single-turn task."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--task-file",
        help="Path to a BFCL task JSONL file.",
    )
    source.add_argument(
        "--category",
        choices=sorted(CATEGORY_TO_FILE),
        help="BFCL single-turn category shorthand.",
    )
    parser.add_argument(
        "--task-ids",
        default="",
        help="Optional comma-separated task ids to generate.",
    )
    parser.add_argument(
        "--task-ids-file",
        default="",
        help="Optional file containing one task id per line.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of selected tasks to generate.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for per-task OpenAPI specs.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="Model name passed to ToolToOpenAPIConverter.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent task workers. Each worker creates its own converter.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output specs. Default: skip existing files.",
    )
    parser.add_argument(
        "--keep-full-task-id-as-title",
        action="store_true",
        help="Use the original task id as info.title instead of the compact BFCL name.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    task_file = _resolve_task_file(args.category, args.task_file)
    output_dir = Path(args.output_dir)
    selected_ids = _parse_task_id_filter(args.task_ids)
    selected_ids.update(_parse_task_id_file(args.task_ids_file))

    logger.info("Loading BFCL tasks from %s", task_file)
    tasks = list(_iter_jsonl(task_file))
    selected = _select_tasks(tasks, selected_ids, args.limit)
    if not selected:
        logger.error("No tasks selected from %s", task_file)
        return 1

    logger.info("Selected %d task(s)", len(selected))
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    generated = 0
    skipped = 0
    failed: List[Tuple[str, str]] = []

    def _log_planned_work() -> None:
        for idx, task in enumerate(selected, start=1):
            task_id = str(task["id"])
            schema_name = _resolve_schema_name(task_id, args.keep_full_task_id_as_title)
            output_path = _resolve_output_path(output_dir, schema_name)
            if output_path.exists() and not args.overwrite:
                logger.info("[%d/%d] Skipping existing %s", idx, len(selected), output_path)
                continue
            try:
                payload = _rewrite_function_names_for_bfcl(_ensure_function_payload(task))
            except ValueError as exc:
                logger.warning("[%d/%d] Skipping %s: %s", idx, len(selected), task_id, exc)
                continue
            logger.info(
                "[%d/%d] Generating %s -> %s (%d endpoint(s))",
                idx,
                len(selected),
                task_id,
                output_path.name,
                len(payload),
            )

    _log_planned_work()

    if args.workers == 1:
        for task in selected:
            task_id = str(task["id"])
            try:
                status, _, path, _ = _generate_task_spec(
                    task,
                    model_name=args.model,
                    output_dir=output_dir,
                    overwrite=args.overwrite,
                    keep_full_task_id_as_title=args.keep_full_task_id_as_title,
                )
                if status == "skipped":
                    skipped += 1
                    continue
                logger.info("Saved %s", path)
                generated += 1
            except Exception as exc:
                logger.exception("Failed to generate schema for %s: %s", task_id, exc)
                failed.append((task_id, str(exc)))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task_id = {
                executor.submit(
                    _generate_task_spec,
                    task,
                    model_name=args.model,
                    output_dir=output_dir,
                    overwrite=args.overwrite,
                    keep_full_task_id_as_title=args.keep_full_task_id_as_title,
                ): str(task["id"])
                for task in selected
            }
            for future in concurrent.futures.as_completed(future_to_task_id):
                task_id = future_to_task_id[future]
                try:
                    status, _, path, _ = future.result()
                    if status == "skipped":
                        skipped += 1
                        continue
                    logger.info("Saved %s", path)
                    generated += 1
                except Exception as exc:
                    logger.exception("Failed to generate schema for %s: %s", task_id, exc)
                    failed.append((task_id, str(exc)))

    logger.info(
        "BFCL single-turn OpenAPI generation complete | generated=%d skipped=%d failed=%d output_dir=%s",
        generated,
        skipped,
        len(failed),
        output_dir,
    )
    if failed:
        logger.error("Failed tasks: %s", ", ".join(task_id for task_id, _ in failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

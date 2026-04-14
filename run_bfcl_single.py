#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("benchmark_execution.log", mode="w")],
)
logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BFCL single-turn runner (GATS / SimSolver path).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--all", action="store_true", help="Run all tests in category")
    test_group.add_argument(
        "--ids",
        type=str,
        default="",
        help="Comma-separated IDs (supports short numeric IDs with auto-prefix)",
    )
    test_group.add_argument("--ids-file", type=str, help="File containing test IDs")

    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="BFCL single-turn category (e.g., simple_python, live_multiple)",
    )
    parser.add_argument("--limit", type=int, help="Limit number of tests")

    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--agent-timeout", type=int, default=360)

    parser.add_argument(
        "--bfcl-eval",
        dest="bfcl_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run BFCL official eval after inference (default: True)",
    )

    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def setup_logging(args: argparse.Namespace) -> None:
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)


def validate_arguments(args: argparse.Namespace) -> None:
    errors = []
    if args.workers < 1:
        errors.append("--workers must be >= 1")
    if args.max_retries < 0:
        errors.append("--max-retries must be >= 0")
    if args.category.startswith("multi_turn"):
        errors.append(f"--category {args.category} is multi-turn; use run_bfcl_multi.py")
    try:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        errors.append(f"Cannot create output dir {args.output_dir}: {exc}")
    if errors:
        for err in errors:
            print(f"Error: {err}")
        sys.exit(1)


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()
    setup_logging(args)
    validate_arguments(args)

    from functools import partial
    from gats import GATSRunner, GATSSolver
    from gats.benchmarks.bfcl.prompts import (
        BFCL_TASK_AGENT_SYSTEM_PROMPT,
        BFCL_SINGLE_DEFAULT_JUDGE_PROMPT,
        BFCL_SINGLE_TRIAGE_JUDGE_PROMPT,
        BFCL_DEFAULT_CHECKLIST_PROMPT,
    )
    from gats.benchmarks.bfcl.helpers import (
        load_single_turn_tasks,
        resolve_test_ids,
        filter_tasks_by_ids,
        append_bfcl_eval_line,
    )

    # --- Load BFCL data ---
    # base_agent_prompt is passed so that load_single_turn_tasks can merge
    # it with per-task system messages (critical for live_* categories).
    logger.info(f"Loading BFCL single-turn tasks: category={args.category}")
    all_tasks = load_single_turn_tasks(
        category=args.category,
        base_agent_prompt=BFCL_TASK_AGENT_SYSTEM_PROMPT,
        limit=args.limit,
    )
    logger.info(f"Loaded {len(all_tasks)} tasks from category {args.category}")

    # --- Filter by IDs ---
    target_ids = resolve_test_ids(
        category=args.category,
        ids=args.ids or None,
        ids_file=getattr(args, "ids_file", None),
        run_all=args.all,
    )
    tasks = filter_tasks_by_ids(
        all_tasks,
        ids=target_ids,
    )

    if not tasks:
        print(f"No tasks matched for category={args.category}")
        sys.exit(0)
    logger.info(f"Running {len(tasks)} tasks")

    # --- Configure solver factory ---
    solver_factory = partial(
        GATSSolver,
        model=args.model,
        max_retries=args.max_retries,
        agent_timeout=args.agent_timeout,
        # Single-turn BFCL: fixed checklist (no dynamic LLM generation).
        enable_checklist=False,
        enable_tool_result_folding=False,
        judge_prompt=BFCL_SINGLE_DEFAULT_JUDGE_PROMPT,
        triage_judge_prompt=BFCL_SINGLE_TRIAGE_JUDGE_PROMPT,
        checklist_prompt=BFCL_DEFAULT_CHECKLIST_PROMPT,
        base_checklist_items=[
            "Coverage: cover all requested values, combinations, and repeated occurrences. Use one array/list call when that cleanly represents the request; use multiple calls when separate results, repeated samples, or item-by-item combinations are requested.",
            "Parameter provenance and format compliance: For each parameter, (1) verify value came from user's message, schema default, or schema-described reformatting/mapping (e.g. schema says '1 for Bangkok' and user says 'Bangkok' \u2192 province_id=1 is schema-derived, NOT fabrication); (2) CHECK the tool_definitions schema description \u2014 if it specifies a format like 'City, State' or shows examples like 'New York, NY', the value MUST match that format (bare 'Chicago' \u2192 FAIL, 'Chicago, IL' \u2192 PASS). Fabrication of external knowledge (URLs, commands, GPS coords) \u2192 FAIL.",
            "Call adequacy: verify that the selected call set captures the user's requested scope, count, filters, and pairings. Do not require the mock tool's response fields to be a perfect final natural-language answer if the call structure itself is correct.",
        ],
        override_openapi_servers=False,
        collect_gecko_usage=False,  # Reduce overhead for single-turn
        debug=args.debug,
        verbose=args.verbose,
    )

    # --- Prepare output paths ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(args.output_dir, args.model.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)
    eval_file = os.path.join(
        model_dir, f"bfcl_official_{args.category}_{timestamp}.jsonl"
    )
    resume_dir = (
        os.path.join(model_dir, ".resume", args.category) if args.resume else None
    )

    # --- Get BFCLBenchmark for function name mapping ---
    benchmark = None
    try:
        import benchmarks.bfcl  # noqa: F401 (register plugin)
        from benchmarks import get_benchmark

        benchmark = get_benchmark("bfcl")
    except Exception:
        logger.warning("Could not load BFCLBenchmark for function name mapping")

    # --- Run ---
    runner = GATSRunner(solver_factory)
    results = runner.run(
        tasks,
        workers=args.workers,
        resume_dir=resume_dir,
        on_task_done=lambda r: append_bfcl_eval_line(
            r, eval_file, is_multi=False, benchmark=benchmark
        ),
    )

    # --- Summary ---
    total = len(results)
    avg_time = sum(r.total_time for r in results) / total if total else 0
    total_attempts = sum(r.total_attempts for r in results)
    print(f"\nCompleted: {total} tasks, avg time: {avg_time:.1f}s, total attempts: {total_attempts}")
    print(f"Eval file: {eval_file}")

    # --- Run BFCL official eval ---
    if args.bfcl_eval and os.path.exists(eval_file):
        try:
            from bfcl_evaluate import main as bfcl_eval_main

            exit_code = bfcl_eval_main(
                model_names=None,
                test_categories=["all"],
                result_dir=args.output_dir,
                specific_file=eval_file,
            )
            if exit_code != 0:
                print(f"BFCL eval returned non-zero: {exit_code}")

            score_path = Path(eval_file).with_name(
                Path(eval_file).stem + "_score.json"
            )
            if score_path.exists():
                score_data = json.loads(score_path.read_text(encoding="utf-8"))
                if isinstance(score_data, list) and score_data:
                    score_data = score_data[0]
                if isinstance(score_data, dict):
                    print(
                        f"BFCL accuracy: {score_data.get('accuracy', 'N/A')} "
                        f"({score_data.get('correct', '?')}/{score_data.get('total', '?')})"
                    )
        except Exception as e:
            logger.error(f"BFCL eval failed: {e}")


if __name__ == "__main__":
    main()

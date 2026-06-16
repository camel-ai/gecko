#!/usr/bin/env python3
"""Run tau-bench airline/retail with Gecko-owned runtime only."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_loader import load_environment_variables

    load_environment_variables()
except Exception:
    pass

from benchmarks.taubench.config import SUPPORTED_DOMAINS, validate_domain
from benchmarks.taubench.internal.agents import InternalTau2AssistantAgent, InternalTau2UserSimulator
from benchmarks.taubench.internal.evaluator.evaluator import EvaluationType, evaluate_simulation
from benchmarks.taubench.internal.loader import get_environment, get_tasks
from benchmarks.taubench.internal.orchestrator import InternalTau2Orchestrator


TAUBENCH_GATS_MODE = "hybrid"
TAUBENCH_SUPPORTED_GATS_MODES = frozenset({"none", "hybrid", "real", "mock"})
TAUBENCH_CONTEXT_MODE = "projected"
TAUBENCH_REFERENCE_MODE = "plan-card"
TAUBENCH_MAX_ASSISTANT_HISTORY_CHARS = 800
TAUBENCH_MAX_USER_HISTORY_CHARS = 2000
TAUBENCH_MAX_TOOL_RESULT_CHARS = 6000
TAUBENCH_PROJECT_ALL_TOOL_RESULTS = False


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json", exclude_none=False)
        except TypeError:
            return value.model_dump()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _parse_ids(raw: Optional[str]) -> Optional[list[str]]:
    if raw is None or not raw.strip():
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def _select_tasks(domain: str, *, task_ids: Optional[list[str]], num_tasks: Optional[int]) -> list[Any]:
    tasks = get_tasks(domain, task_split_name=None)
    if task_ids is not None:
        wanted = set(str(task_id) for task_id in task_ids)
        tasks = [task for task in tasks if str(task.id) in wanted]
    if num_tasks is not None:
        tasks = tasks[:num_tasks]
    return tasks


def _task_sort_key(row: dict[str, Any]) -> tuple[int, int | str]:
    trial = int(row.get("trial") or 0)
    task_id = row.get("task_id")
    try:
        return trial, int(task_id)
    except Exception:
        return trial, str(task_id)


def _safe_path_part(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)


def _count_tool_calls(messages: list[Any]) -> int:
    count = 0
    for message in messages or []:
        count += len(getattr(message, "tool_calls", None) or [])
    return count


def _default_output_dir(args: argparse.Namespace, *, timestamp: str, domain: str) -> Path:
    parts = [
        timestamp,
        domain,
        _safe_path_part(args.agent_llm),
        f"user-{_safe_path_part(args.user_llm)}",
        f"t{args.num_trials}",
        TAUBENCH_GATS_MODE,
    ]
    if TAUBENCH_GATS_MODE != "none":
        parts.extend([TAUBENCH_CONTEXT_MODE, TAUBENCH_REFERENCE_MODE])
    return PROJECT_ROOT / "results" / "taubench" / "_".join(parts)


def _run_one(
    *,
    domain: str,
    task: Any,
    trial: int,
    seed: Optional[int],
    agent_llm: str,
    user_llm: str,
    judge_llm: str,
    max_steps: int,
    gats_mode: str,
    gats_retries: int,
    agent_timeout: int,
    gecko_url: str,
    debug: bool,
) -> dict[str, Any]:
    environment = get_environment(domain)
    user = InternalTau2UserSimulator(
        task=task,
        model_name=user_llm,
        timeout=agent_timeout,
    )
    agent = InternalTau2AssistantAgent(
        domain=domain,
        task=task,
        environment=environment,
        model_name=agent_llm,
        gats_mode=gats_mode,
        context_mode=TAUBENCH_CONTEXT_MODE,
        max_assistant_history_chars=TAUBENCH_MAX_ASSISTANT_HISTORY_CHARS,
        max_user_history_chars=TAUBENCH_MAX_USER_HISTORY_CHARS,
        max_tool_result_chars=TAUBENCH_MAX_TOOL_RESULT_CHARS,
        project_all_tool_results=TAUBENCH_PROJECT_ALL_TOOL_RESULTS,
        gecko_url=gecko_url,
        gats_retries=gats_retries,
        timeout=agent_timeout,
        debug=debug,
    )
    orchestrator = InternalTau2Orchestrator(
        task_id=str(task.id),
        user=user,
        agent=agent,
        max_steps=max_steps,
    )
    simulation = orchestrator.run(trial=trial, seed=seed)
    synthetic_action_messages = agent.get_gats_synthetic_action_messages()
    synthetic_read_calls = agent.get_gats_synthetic_read_calls()
    synthetic_action_call_count = _count_tool_calls(synthetic_action_messages)
    evaluation_error = None
    try:
        reward_info = evaluate_simulation(
            simulation,
            task,
            evaluation_type=EvaluationType.ALL,
            domain=domain,
            nl_judge_model=judge_llm,
            synthetic_action_messages=synthetic_action_messages,
        )
    except Exception as exc:
        logging.exception(
            "tau2 internal evaluation failed for domain=%s task=%s trial=%s",
            domain,
            task.id,
            trial,
        )
        evaluation_error = f"{type(exc).__name__}: {exc}"
        from benchmarks.taubench.internal.data_model.simulation import RewardInfo

        reward_info = RewardInfo(
            reward=0.0,
            info={"evaluation_error": evaluation_error},
        )
    simulation.reward_info = reward_info
    simulation_payload = _jsonable(simulation)
    if synthetic_read_calls:
        simulation_payload["gats_synthetic_read_calls"] = _jsonable(synthetic_read_calls)
        simulation_payload["gats_synthetic_read_call_count"] = len(synthetic_read_calls)
        simulation_payload["gats_synthetic_action_call_count"] = synthetic_action_call_count
    return {
        "domain": domain,
        "task_id": str(task.id),
        "trial": trial,
        "seed": seed,
        "reward": reward_info.reward,
        "reward_info": _jsonable(reward_info),
        "gats_synthetic_read_call_count": len(synthetic_read_calls),
        "gats_synthetic_action_call_count": synthetic_action_call_count,
        "gats_synthetic_read_calls": _jsonable(synthetic_read_calls),
        "evaluation_error": evaluation_error,
        "simulation": simulation_payload,
    }


def _write_outputs(
    *,
    output_dir: Path,
    domain: str,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    completed: int,
) -> None:
    rows = sorted(rows, key=_task_sort_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "runner": "run_taubench.py",
        "domain": domain,
        "agent_llm": args.agent_llm,
        "user_llm": args.user_llm,
        "judge_llm": args.judge_llm or args.agent_llm,
        "num_trials": args.num_trials,
        "workers": args.workers,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "gats_mode": TAUBENCH_GATS_MODE,
        "gats_retries": args.gats_retries,
        "completed": completed,
        "total": len(rows),
        "reward_sum": sum(float(row.get("reward") or 0.0) for row in rows),
    }
    if rows:
        summary["accuracy"] = summary["reward_sum"] / len(rows)
    (output_dir / "summary.json").write_text(
        json.dumps({**summary, "results": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    simulations = [row["simulation"] for row in rows if row.get("simulation") is not None]
    (output_dir / "results.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "info": summary,
                "tasks": [],
                "simulations": simulations,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tau-bench airline/retail benchmark")
    parser.add_argument("--domain", required=True, choices=sorted(SUPPORTED_DOMAINS))
    parser.add_argument("--task-ids", default=None, help="Comma-separated task ids")
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--agent-llm", default="haivex-gpt-5.5")
    parser.add_argument("--user-llm", default="haivex-gpt-5.5")
    parser.add_argument("--judge-llm", default=None)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gats-retries", type=int, default=2)
    parser.add_argument("--agent-timeout", type=int, default=360)
    parser.add_argument("--gecko-url", default="http://localhost:8000")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    os.environ.setdefault("GECKO_MOCK_TOOL_TIMEOUT_SECONDS", str(args.agent_timeout))

    if TAUBENCH_GATS_MODE not in TAUBENCH_SUPPORTED_GATS_MODES:
        supported = ", ".join(sorted(TAUBENCH_SUPPORTED_GATS_MODES))
        raise SystemExit(f"Error: TAUBENCH_GATS_MODE must be one of {supported}")

    if TAUBENCH_GATS_MODE != "none":
        from utils.gecko_preflight import GeckoPreflightError, require_gecko_preflight

        try:
            health = require_gecko_preflight(
                args.gecko_url,
                min_workers=args.workers,
            )
        except GeckoPreflightError as exc:
            raise SystemExit(f"Error: {exc}") from exc
        print(
            f"Gecko preflight ok: {args.gecko_url.rstrip('/')} "
            f"(workers={health.get('config', {}).get('workers')})"
        )

    domain = validate_domain(args.domain)
    tasks = _select_tasks(domain, task_ids=_parse_ids(args.task_ids), num_tasks=args.num_tasks)
    if not tasks:
        raise ValueError("No tau-bench tasks selected")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _default_output_dir(args, timestamp=timestamp, domain=domain)
    )

    jobs = []
    for trial in range(args.num_trials):
        seed = args.seed + trial if args.seed is not None else None
        for task in tasks:
            jobs.append((trial, seed, task))

    judge_llm = args.judge_llm or args.agent_llm
    rows: list[dict[str, Any]] = []

    def run_job(job: tuple[int, Optional[int], Any]) -> dict[str, Any]:
        trial, seed, task = job
        return _run_one(
            domain=domain,
            task=task,
            trial=trial,
            seed=seed,
            agent_llm=args.agent_llm,
            user_llm=args.user_llm,
            judge_llm=judge_llm,
            max_steps=args.max_steps,
            gats_mode=TAUBENCH_GATS_MODE,
            gats_retries=args.gats_retries,
            agent_timeout=args.agent_timeout,
            gecko_url=args.gecko_url,
            debug=args.debug,
        )

    if args.workers <= 1:
        for idx, job in enumerate(jobs, start=1):
            trial, _, task = job
            print(f"[{idx}/{len(jobs)}] taubench {domain} task={task.id} trial={trial}")
            rows.append(run_job(job))
            _write_outputs(output_dir=output_dir, domain=domain, args=args, rows=rows, completed=idx)
            print(f"  reward={rows[-1].get('reward')}")
    else:
        print(f"Running {len(jobs)} tau-bench jobs with {args.workers} workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {executor.submit(run_job, job): job for job in jobs}
            for idx, future in enumerate(concurrent.futures.as_completed(future_to_job), start=1):
                trial, seed, task = future_to_job[future]
                try:
                    row = future.result()
                except Exception as exc:
                    logging.exception("tau-bench task failed: task=%s trial=%s", task.id, trial)
                    row = {
                        "domain": domain,
                        "task_id": str(task.id),
                        "trial": trial,
                        "seed": seed,
                        "reward": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                rows.append(row)
                _write_outputs(output_dir=output_dir, domain=domain, args=args, rows=rows, completed=idx)
                print(f"[{idx}/{len(jobs)}] task={task.id} trial={trial} reward={row.get('reward')}")

    reward_sum = sum(float(row.get("reward") or 0.0) for row in rows)
    print(f"Results: {output_dir}")
    print(f"Reward: {reward_sum}/{len(rows)} = {reward_sum / len(rows):.2%}")


if __name__ == "__main__":
    main()

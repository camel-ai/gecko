"""Airline domain environment backed by in-repo tau2 data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from benchmarks.taubench.internal.data_model.tasks import Task
from benchmarks.taubench.internal.domains.airline.data_model import FlightDB
from benchmarks.taubench.internal.domains.airline.tools import AirlineTools
from benchmarks.taubench.internal.domains.airline.utils import (
    AIRLINE_DB_PATH,
    AIRLINE_POLICY_PATH,
    AIRLINE_TASK_SET_PATH,
)
from benchmarks.taubench.internal.environment.environment import Environment
from benchmarks.taubench.internal.utils import load_file


def get_environment(
    db: Optional[FlightDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if solo_mode:
        raise ValueError("Airline domain does not support solo mode")
    if db is None:
        db = FlightDB.load(AIRLINE_DB_PATH)
    with open(AIRLINE_POLICY_PATH, "r", encoding="utf-8") as handle:
        policy = handle.read()
    return Environment(
        domain_name="airline",
        policy=policy,
        tools=AirlineTools(db),
    )


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    tasks = [Task.model_validate(task) for task in load_file(AIRLINE_TASK_SET_PATH)]
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. "
            f"Valid splits are: {sorted(task_splits.keys())}"
        )
    return [task for task in tasks if task.id in task_splits[task_split_name]]


def get_tasks_split() -> dict[str, list[str]]:
    split_file = Path(AIRLINE_TASK_SET_PATH).parent / f"split_{Path(AIRLINE_TASK_SET_PATH).stem}.json"
    return load_file(split_file)

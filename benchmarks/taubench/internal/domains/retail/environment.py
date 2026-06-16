"""Retail domain environment backed by in-repo tau2 data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from benchmarks.taubench.internal.data_model.tasks import Task
from benchmarks.taubench.internal.domains.retail.data_model import RetailDB
from benchmarks.taubench.internal.domains.retail.tools import RetailTools
from benchmarks.taubench.internal.domains.retail.utils import (
    RETAIL_DB_PATH,
    RETAIL_POLICY_PATH,
    RETAIL_TASK_SET_PATH,
)
from benchmarks.taubench.internal.environment.environment import Environment
from benchmarks.taubench.internal.utils import load_file


def get_environment(
    db: Optional[RetailDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if solo_mode:
        raise ValueError("Retail domain does not support solo mode")
    if db is None:
        db = RetailDB.load(RETAIL_DB_PATH)
    with open(RETAIL_POLICY_PATH, "r", encoding="utf-8") as handle:
        policy = handle.read()
    return Environment(
        domain_name="retail",
        policy=policy,
        tools=RetailTools(db),
    )


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    tasks = [Task.model_validate(task) for task in load_file(RETAIL_TASK_SET_PATH)]
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
    split_file = Path(RETAIL_TASK_SET_PATH).parent / f"split_{Path(RETAIL_TASK_SET_PATH).stem}.json"
    return load_file(split_file)

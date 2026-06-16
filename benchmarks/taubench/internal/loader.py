"""In-repo tau2 airline/retail loader and registry helpers."""

from __future__ import annotations

from typing import Callable, Optional

from benchmarks.taubench.config import validate_domain
from benchmarks.taubench.internal.data_model.tasks import Task
from benchmarks.taubench.internal.environment.environment import Environment


def get_environment_constructor(domain: str) -> Callable[..., Environment]:
    domain = validate_domain(domain)
    if domain == "airline":
        from benchmarks.taubench.internal.domains.airline.environment import get_environment

        return get_environment
    if domain == "retail":
        from benchmarks.taubench.internal.domains.retail.environment import get_environment

        return get_environment
    raise AssertionError(domain)


def get_environment(domain: str, **kwargs) -> Environment:
    return get_environment_constructor(domain)(**kwargs)


def get_tasks(domain: str, task_split_name: Optional[str] = "base") -> list[Task]:
    domain = validate_domain(domain)
    if domain == "airline":
        from benchmarks.taubench.internal.domains.airline.environment import get_tasks as load

        return load(task_split_name=task_split_name)
    if domain == "retail":
        from benchmarks.taubench.internal.domains.retail.environment import get_tasks as load

        return load(task_split_name=task_split_name)
    raise AssertionError(domain)


def get_task(domain: str, task_id: str) -> Task:
    for task in get_tasks(domain, task_split_name=None):
        if str(task.id) == str(task_id):
            return task
    raise KeyError(f"Task {task_id!r} not found for tau2 domain {domain!r}")

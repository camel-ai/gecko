import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Optional

from gats.core.task import GATSTask, GATSResult, GATSTurn
from inference.client_engine import TaskInfraError

logger = logging.getLogger(__name__)


class GATSRunner:
    """Execute GATSTasks with optional parallelism, resume, and callbacks."""

    def __init__(
        self,
        solver_factory: Callable[[GATSTask], Any],
        *,
        target_score: float = 1.0,
    ):
        self._solver_factory = solver_factory
        self._target_score = target_score

    def run_one(self, task: GATSTask) -> GATSResult:
        """Execute a single task (single or multi-turn)."""
        task_start = time.time()
        turns: List[GATSTurn] = []
        all_events = []
        success = True

        try:
            solver = self._solver_factory(task)

            is_multi_turn = len(task.turns) > 1

            for turn_idx, question in enumerate(task.turns):
                gats_turn = solver.process_turn(question)
                turns.append(gats_turn)

                if gats_turn.score < self._target_score:
                    success = False
                    if not is_multi_turn:
                        break

            all_events = solver.get_events()

        except TaskInfraError:
            raise
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}", exc_info=True)
            success = False

        return GATSResult(
            task_id=task.id,
            success=success,
            turns=turns,
            total_time=time.time() - task_start,
            events=all_events,
            metadata=task.metadata,
        )

    def run(
        self,
        tasks: List[GATSTask],
        *,
        workers: int = 1,
        resume_dir: Optional[str] = None,
        on_task_done: Optional[Callable[[GATSResult], None]] = None,
    ) -> List[GATSResult]:
        """Execute multiple tasks with parallelism and resume.

        Args:
            tasks: Tasks to execute.
            workers: Parallel worker count.
            resume_dir: Directory for resume state. Completed task IDs are
                tracked via ``{task_id}.done`` marker files; on restart those
                tasks are skipped.
            on_task_done: Called after each task completes (for incremental saving).
        """
        # Resume: load completed task IDs
        completed_ids: set = set()
        if resume_dir:
            os.makedirs(resume_dir, exist_ok=True)
            for fname in os.listdir(resume_dir):
                if fname.endswith(".done"):
                    completed_ids.add(fname[:-5])
            if completed_ids:
                logger.info(f"Resuming: {len(completed_ids)} tasks already completed")

        tasks_to_run = [t for t in tasks if t.id not in completed_ids]
        if not tasks_to_run:
            logger.info("All tasks already completed")
            return []

        results: List[GATSResult] = []
        infra_retries = self._infra_task_retries()
        infra_backoff = self._infra_task_backoff()

        def _process_task(task: GATSTask) -> Optional[GATSResult]:
            last_error: Optional[TaskInfraError] = None
            for infra_attempt in range(infra_retries + 1):
                try:
                    result = self.run_one(task)
                    break
                except TaskInfraError as exc:
                    last_error = exc
                    if infra_attempt >= infra_retries:
                        logger.error(
                            "Task %s exhausted infra retries; leaving untested: %s",
                            task.id,
                            exc,
                        )
                        result = None
                        break
                    sleep_for = infra_backoff * (2 ** infra_attempt)
                    logger.warning(
                        "Retrying task %s after infra error (%s/%s, sleep=%.1fs): %s",
                        task.id,
                        infra_attempt + 1,
                        infra_retries,
                        sleep_for,
                        exc,
                    )
                    time.sleep(sleep_for)
            else:  # pragma: no cover - loop always breaks
                logger.error(
                    "Task %s exhausted infra retries; leaving untested: %s",
                    task.id,
                    last_error or "unknown infra error",
                )
                result = None
            if result is not None and resume_dir:
                done_path = os.path.join(resume_dir, f"{task.id}.done")
                with open(done_path, "w") as f:
                    f.write("")
            return result

        if workers <= 1:
            for i, task in enumerate(tasks_to_run):
                logger.info(f"[{i + 1}/{len(tasks_to_run)}] {task.id}")
                result = _process_task(task)
                if result is None:
                    continue
                results.append(result)
                if on_task_done:
                    on_task_done(result)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_task = {
                    executor.submit(_process_task, t): t for t in tasks_to_run
                }
                for i, future in enumerate(as_completed(future_to_task)):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        if result is None:
                            logger.info(
                                f"[{i + 1}/{len(tasks_to_run)}] {task.id} "
                                "(untested: infra/provider failure)"
                            )
                            continue
                        results.append(result)
                        if on_task_done:
                            on_task_done(result)
                        logger.info(
                            f"[{i + 1}/{len(tasks_to_run)}] {task.id} "
                            f"(success={result.success}, score={result.final_score:.2f})"
                        )
                    except Exception as e:
                        logger.error(f"Task {task.id} raised exception: {e}")
                        error_result = GATSResult(
                            task_id=task.id,
                            success=False,
                            turns=[],
                            total_time=0.0,
                            metadata=task.metadata,
                        )
                        results.append(error_result)
                        if on_task_done:
                            on_task_done(error_result)

        return results

    @staticmethod
    def _infra_task_retries() -> int:
        raw = os.getenv("GATS_INFRA_TASK_RETRIES", "1")
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _infra_task_backoff() -> float:
        raw = os.getenv("GATS_INFRA_TASK_RETRY_BACKOFF", "5.0")
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return 5.0

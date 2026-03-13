import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional

from gats.core.config import GATSConfig
from gats.core.task import GATSTask, GATSResult, GATSTurn
from gats.engine.solver import GATSSolver

logger = logging.getLogger(__name__)


class GATSRunner:
    """Execute GATSTasks with optional parallelism, resume, and callbacks."""

    def __init__(self, config: GATSConfig):
        self.config = config

    def run_one(self, task: GATSTask) -> GATSResult:
        """Execute a single task (single or multi-turn)."""
        task_start = time.time()
        turns: List[GATSTurn] = []
        all_events = []
        success = True

        try:
            solver = GATSSolver(task, self.config)

            for turn_idx, question in enumerate(task.turns):
                gats_turn = solver.process_turn(question)
                turns.append(gats_turn)

                if gats_turn.score < self.config.target_score:
                    success = False
                    break

            all_events = solver.get_events()

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
        workers: Optional[int] = None,
        resume_dir: Optional[str] = None,
        on_task_done: Optional[Callable[[GATSResult], None]] = None,
    ) -> List[GATSResult]:
        """Execute multiple tasks with parallelism and resume.

        Args:
            tasks: Tasks to execute.
            workers: Parallel worker count (default: config.max_workers).
            resume_dir: Directory for resume state. Completed task IDs are
                tracked via ``{task_id}.done`` marker files; on restart those
                tasks are skipped.
            on_task_done: Called after each task completes (for incremental saving).
        """
        effective_workers = workers or self.config.max_workers

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

        def _process_task(task: GATSTask) -> GATSResult:
            result = self.run_one(task)
            if resume_dir:
                done_path = os.path.join(resume_dir, f"{task.id}.done")
                with open(done_path, "w") as f:
                    f.write("")
            return result

        if effective_workers <= 1:
            for i, task in enumerate(tasks_to_run):
                logger.info(f"[{i + 1}/{len(tasks_to_run)}] {task.id}")
                result = _process_task(task)
                results.append(result)
                if on_task_done:
                    on_task_done(result)
        else:
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                future_to_task = {
                    executor.submit(_process_task, t): t for t in tasks_to_run
                }
                for i, future in enumerate(as_completed(future_to_task)):
                    task = future_to_task[future]
                    try:
                        result = future.result()
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

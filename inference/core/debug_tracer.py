import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from utils.test_case_adapter import TestCaseAdapter

logger = logging.getLogger(__name__)


class DebugTracer:
    """Manages debug trace output (files + console) for a single task."""

    def __init__(self, task_id: str, model_name: str, console_enabled: bool = False):
        self.task_id = task_id
        self.model_name = model_name
        self.console_enabled = console_enabled

    def trace_print(self, message: str) -> None:
        """Print to console if debug tracing is enabled."""
        if self.console_enabled:
            print(message)

    def write_task_debug_file(
        self,
        outcome: Any,
        attempts: list,
        checklist: List[Dict[str, Any]],
        test_case: Any,
        agent_system_prompt: str | None,
        tool_definitions: List[Dict[str, Any]],
    ) -> None:
        """Write a per-task, per-turn debug file to debug_traces/{task_id}_turn{N}_debug.txt."""
        os.makedirs("debug_traces", exist_ok=True)
        path = os.path.join("debug_traces", f"{self.task_id}_turn{outcome.turn_idx}_debug.txt")

        lines: List[str] = []

        def _section(title: str) -> None:
            lines.append("")
            lines.append(f"{'=' * 72}")
            lines.append(f"  {title}")
            lines.append(f"{'=' * 72}")

        def _subsection(title: str) -> None:
            lines.append("")
            lines.append(f"--- {title} ---")

        real_task_id = TestCaseAdapter.get_id(test_case) or "unknown"
        lines.append(f"Task Debug Report: {real_task_id}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Model: {self.model_name}")
        lines.append(
            f"Best attempt: {outcome.best_attempt}  |  Score: {outcome.score:.2f}"
            f"  |  Time: {outcome.execution_time:.1f}s"
        )
        lines.append(f"Total attempts: {len(attempts)}  |  Tool calls: {len(outcome.final_tool_calls)}")

        # ---- User request ----
        _section("USER REQUEST")
        lines.append(outcome.user_message)

        # ---- Ground truth (if available) ----
        gt = None
        if hasattr(test_case, "metadata") and isinstance(test_case.metadata, dict):
            gt = test_case.metadata.get("ground_truth")
        if gt is not None:
            _section("GROUND TRUTH")
            lines.append(json.dumps(gt, indent=2, default=str))

        # ---- Available tools ----
        _section("AVAILABLE TOOLS")
        if tool_definitions:
            for td in tool_definitions:
                name = td.get("name", "?")
                desc = td.get("description", "")[:120]
                params = td.get("parameters")
                param_summary = ""
                if isinstance(params, dict):
                    props = params.get("properties", {})
                    required = params.get("required", [])
                    if props:
                        parts = []
                        for pname in props:
                            marker = "*" if pname in required else ""
                            parts.append(f"{pname}{marker}")
                        param_summary = f"  params: {', '.join(parts)}"
                lines.append(f"  - {name}: {desc}{param_summary}")
        else:
            lines.append("  (no tool definitions available)")

        # ---- Agent system prompt ----
        _section("AGENT SYSTEM PROMPT")
        lines.append(agent_system_prompt or "<not set>")

        # ---- Checklist ----
        if checklist:
            _section("CHECKLIST ITEMS")
            for i, item in enumerate(checklist, 1):
                desc = item.get("description", str(item))
                lines.append(f"  {i}. {desc}")

        # ---- Per-attempt details ----
        for att in attempts:
            _section(f"ATTEMPT {att.attempt}  (score={att.score:.2f}, time={att.execution_time:.1f}s)")

            _subsection("Agent Input (Enhanced Message)")
            lines.append(att.agent_prompt)

            _subsection("Agent Response")
            lines.append(att.agent_response or "<empty>")

            _subsection(f"Tool Calls ({len(att.tool_calls)})")
            if att.tool_calls:
                for i, tc in enumerate(att.tool_calls, 1):
                    func = tc.get("function", "?") if isinstance(tc, dict) else str(tc)
                    args = tc.get("arguments", {}) if isinstance(tc, dict) else {}
                    result = tc.get("result", None) if isinstance(tc, dict) else None
                    lines.append(f"  [{i}] {func}")
                    lines.append(f"      args: {json.dumps(args, default=str)}")
                    if result is not None:
                        res_str = str(result)
                        if len(res_str) > 500:
                            res_str = res_str[:500] + "..."
                        lines.append(f"      result: {res_str}")
            else:
                lines.append("  (no tool calls)")

            _subsection("Judge Feedback")
            if att.feedback:
                judgment_results = att.feedback.get("judgment_results", [])
                if judgment_results:
                    for jr in judgment_results:
                        status = jr.get("status", "?")
                        desc = jr.get("description", "")[:200]
                        reasoning = jr.get("reasoning", "")[:300]
                        lines.append(f"  [{status.upper()}] {desc}")
                        if reasoning:
                            lines.append(f"    reason: {reasoning}")
                elif att.feedback.get("skipped_judge"):
                    lines.append(f"  (judge skipped: {att.feedback['skipped_judge']})")
                elif att.feedback.get("agent_failure"):
                    lines.append("  (agent failure)")
                else:
                    lines.append(f"  {json.dumps(att.feedback, indent=2, default=str)[:500]}")
            else:
                lines.append("  (no feedback)")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @staticmethod
    def dump_events_json(events: list, file_path: str) -> bool:
        """Dump the full event trace to a JSON file. Returns True on success."""
        try:
            payload = [
                {
                    "type": e.type,
                    "ts": e.ts,
                    "turn_idx": e.turn_idx,
                    "attempt": e.attempt,
                    "data": e.data,
                }
                for e in events
            ]
            with open(file_path, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to dump events JSON: {e}")
            return False

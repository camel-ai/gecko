"""GATS support for the in-repo tau-bench runner."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import requests

from benchmarks.base.test_case import TestCase
from benchmarks.taubench.config import DOMAIN_TOOLS, validate_domain
from benchmarks.taubench.context import Tau2ConversationRecorder
from benchmarks.taubench.prompts import (
    TAU2_CHECKLIST_SYSTEM_PROMPT,
    TAU2_JUDGE_SYSTEM_PROMPT,
    format_agent_system_prompt,
)
from benchmarks.taubench.tooling import create_tau2_tool_registry
from inference.core import GATSAttemptRunner, SimEvent, TurnOutcome

logger = logging.getLogger(__name__)


def preview(text: Optional[str], n: int = 200) -> str:
    if not text:
        return ""
    return text[:n]


class TauBenchGATSManager:
    """Build and run fresh GATS simulation attempts for each tau-bench user turn."""

    def __init__(
        self,
        *,
        domain: str,
        task: Any,
        policy: str,
        tau2_tools: List[Any],
        agent_model: str,
        gats_mode: str = "hybrid",
        context_mode: str = "full",
        max_assistant_history_chars: int = 800,
        max_user_history_chars: int = 2000,
        max_tool_result_chars: int = 6000,
        project_all_tool_results: bool = False,
        gecko_url: str = "http://localhost:8000",
        max_retries: int = 2,
        agent_timeout: int = 180,
        debug: bool = False,
    ):
        self.domain = validate_domain(domain)
        self.task = task
        self.policy = policy or ""
        self.tau2_tools = tau2_tools
        self.agent_model = agent_model
        self.gats_mode = gats_mode or "hybrid"
        self.context_mode = (context_mode or "full").strip().lower()
        self.max_assistant_history_chars = max_assistant_history_chars
        self.max_user_history_chars = max_user_history_chars
        self.max_tool_result_chars = max_tool_result_chars
        self.project_all_tool_results = project_all_tool_results
        self.gecko_url = gecko_url.rstrip("/")
        self.max_retries = max_retries
        self.agent_timeout = agent_timeout
        self.debug = debug
        self.session_id: Optional[str] = None
        self.last_applied_tool_call_count = 0

    @property
    def task_id(self) -> str:
        return str(getattr(self.task, "id", "unknown"))

    def _ensure_session(self) -> str:
        if self.session_id:
            return self.session_id

        response = requests.get(f"{self.gecko_url}/session-id", timeout=30)
        response.raise_for_status()
        session_id = response.json().get("session_id")
        if not session_id:
            raise RuntimeError("Gecko /session-id did not return a session_id")

        self.session_id = session_id
        response = requests.post(
            f"{self.gecko_url}/set-session-state",
            headers={"X-Session-ID": session_id},
            json={"state": {"domain": self.domain, "real_tool_calls": []}},
            timeout=30,
        )
        response.raise_for_status()
        return session_id

    def _fetch_session_state(self) -> Dict[str, Any]:
        session_id = self._ensure_session()
        response = requests.get(
            f"{self.gecko_url}/get-session-state",
            headers={"X-Session-ID": session_id},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json() if response.content else {}
        state = payload.get("state", {})
        return state if isinstance(state, dict) else {}

    def _record_sync_debug(
        self,
        *,
        session_id: str,
        tool_calls: List[Dict[str, Any]],
        updated_state: Optional[Dict[str, Any]] = None,
        sync_policy: Optional[str] = None,
    ) -> None:
        if self.debug:
            logger.debug(
                "tau-bench Gecko sync task=%s session=%s tools=%s state_keys=%s",
                self.task_id,
                session_id,
                [call.get("name") for call in tool_calls],
                sorted((updated_state or {}).keys()),
            )

    def _read_only_sync_policy(self) -> Optional[str]:
        mode = (self.gats_mode or "").strip().lower()
        if mode in {"hybrid", "real"}:
            return "upsert_observed_state"
        return None

    def _sync_observed_tool_calls(self, real_tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        session_id = self._ensure_session()
        if len(real_tool_calls) <= self.last_applied_tool_call_count:
            return self._fetch_session_state()

        new_calls = real_tool_calls[self.last_applied_tool_call_count :]
        normalized = []
        for call in new_calls:
            name = call.get("name") or call.get("function")
            if not name:
                continue
            normalized.append(
                {
                    "name": name,
                    "arguments": deepcopy(call.get("args") or call.get("arguments") or {}),
                    "result": deepcopy(call.get("result")),
                }
            )

        if normalized:
            sync_policy = self._read_only_sync_policy()
            request_payload: Dict[str, Any] = {"tool_calls": normalized}
            if sync_policy:
                request_payload["read_only_sync_policy"] = sync_policy
            response = requests.post(
                f"{self.gecko_url}/update-state-from-real",
                headers={"X-Session-ID": session_id},
                json=request_payload,
                timeout=300,
            )
            response.raise_for_status()
            payload = response.json() if response.content else {}
            state = payload.get("updated_state")
            self.last_applied_tool_call_count = len(real_tool_calls)
            if isinstance(state, dict):
                self._record_sync_debug(
                    session_id=session_id,
                    tool_calls=normalized,
                    updated_state=state,
                    sync_policy=sync_policy,
                )
                return state

        return self._fetch_session_state()

    def _task_as_dict(self) -> Dict[str, Any]:
        if hasattr(self.task, "model_dump"):
            return self.task.model_dump(mode="json", exclude_none=False)
        if isinstance(self.task, dict):
            return deepcopy(self.task)
        return {"id": self.task_id, "repr": repr(self.task)}

    def _build_attempt_runner(self, current_config: Dict[str, Any]) -> GATSAttemptRunner:
        registry = create_tau2_tool_registry(
            original_tools=self.tau2_tools,
            domain=self.domain,
            mode=self.gats_mode,
            gecko_url=self.gecko_url,
        )
        test_case = TestCase(
            id=f"tau2:{self.domain}:{self.task_id}",
            metadata={
                "benchmark": "tau2",
                "domain": self.domain,
                "task": self._task_as_dict(),
                "initial_config": deepcopy(current_config),
                "involved_classes": [self.domain],
            },
        )
        return GATSAttemptRunner(
            test_case=test_case,
            initial_config=current_config,
            model_name=self.agent_model,
            max_retries=self.max_retries,
            agent_timeout=self.agent_timeout,
            mock_server_url=self.gecko_url,
            tool_registry=registry,
            agent_system_prompt=format_agent_system_prompt(self.policy),
            judge_system_prompt=TAU2_JUDGE_SYSTEM_PROMPT,
            checklist_system_prompt=TAU2_CHECKLIST_SYSTEM_PROMPT,
            base_checklist_items=[],
            append_base_checklist_to_generated=False,
            enable_evaluation=True,
            enable_checklist=True,
            include_agent_response_in_judge=True,
            enable_tool_result_folding=True,
            enable_readonly_evidence_branch=True,
            read_only_sync_policy=self._read_only_sync_policy(),
            readonly_evidence_result_projector=self._project_retry_evidence_result,
            readonly_evidence_result_max_chars=max(400, min(1200, self.max_tool_result_chars)),
            agent_max_iteration=10,
            enable_debug=self.debug,
            verbose_debug=self.debug,
        )

    @staticmethod
    def _project_retry_evidence_result(result: Any, max_chars: int) -> Any:
        projected, _ = Tau2ConversationRecorder._project_tool_result(
            result,
            max_chars=max_chars,
            force=True,
        )
        return projected

    @staticmethod
    def _attempt_events(outcome: TurnOutcome) -> Dict[int, Dict[str, Any]]:
        attempts: Dict[int, Dict[str, Any]] = {}
        for event in outcome.events:
            if not isinstance(event, SimEvent) or event.attempt is None:
                continue
            entry = attempts.setdefault(event.attempt, {"attempt": event.attempt})
            if event.type == "agent_response":
                entry["agent_response"] = event.data.get("response", "")
            elif event.type == "tool_calls":
                entry["tool_calls"] = deepcopy(event.data.get("tool_calls", []))
            elif event.type == "inherited_evidence":
                entry["inherited_tool_calls"] = deepcopy(event.data.get("tool_calls", []))
            elif event.type == "judge":
                entry["judge"] = deepcopy(event.data)
            elif event.type == "attempt_end":
                entry["summary"] = deepcopy(event.data)
        return attempts

    @staticmethod
    def _format_tool_call(tool_call: Dict[str, Any]) -> str:
        name = tool_call.get("function") or tool_call.get("name") or "unknown"
        args = tool_call.get("arguments") or tool_call.get("args") or {}
        if isinstance(args, dict) and args:
            rendered_args = ", ".join(f"{key}={value!r}" for key, value in args.items())
        else:
            rendered_args = ""
        return f"{name}({rendered_args})"

    @staticmethod
    def _tool_call_name(tool_call: Dict[str, Any]) -> str:
        return str(
            tool_call.get("function")
            or tool_call.get("name")
            or tool_call.get("tool")
            or ""
        ).strip()

    @staticmethod
    def _tool_call_args(tool_call: Dict[str, Any]) -> Dict[str, Any]:
        args = tool_call.get("arguments") or tool_call.get("args") or {}
        return deepcopy(args) if isinstance(args, dict) else {}

    @classmethod
    def _tool_call_key(cls, tool_call: Dict[str, Any]) -> Tuple[str, str]:
        name = cls._tool_call_name(tool_call)
        args = cls._tool_call_args(tool_call)
        try:
            rendered_args = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            rendered_args = str(args)
        return name, rendered_args

    def _readonly_tool_names(self) -> set[str]:
        cfg = DOMAIN_TOOLS.get(self.domain)
        names = set(cfg.read_tools if cfg is not None else [])
        # calculate is declared read-only in the tau-bench mock schemas even
        # though it is grouped as a generic tool in domain config.
        names.add("calculate")
        return names

    def _is_safe_readonly_evidence_call(self, tool_call: Dict[str, Any]) -> bool:
        if not isinstance(tool_call, dict):
            return False
        name = self._tool_call_name(tool_call)
        if not name or name not in self._readonly_tool_names():
            return False
        if name == "transfer_to_human_agents":
            return False
        if "result" not in tool_call:
            return False
        if tool_call.get("error") or tool_call.get("exception"):
            return False
        result = tool_call.get("result")
        if isinstance(result, dict) and result.get("error"):
            return False
        if isinstance(result, str) and result.strip().lower().startswith("error:"):
            return False
        return True

    def _compact_readonly_evidence_call(
        self,
        tool_call: Dict[str, Any],
        *,
        source: str,
        source_attempt: int,
    ) -> Dict[str, Any]:
        name = self._tool_call_name(tool_call)
        result = self._project_retry_evidence_result(
            tool_call.get("result"),
            max(400, min(1200, self.max_tool_result_chars)),
        )
        return {
            "function": name,
            "name": name,
            "arguments": self._tool_call_args(tool_call),
            "result": result,
            "source": source,
            "source_attempt": source_attempt,
            "gats_readonly_evidence": True,
        }

    def _accepted_readonly_evidence(self, outcome: TurnOutcome) -> List[Dict[str, Any]]:
        if float(outcome.score or 0.0) < 1.0:
            return []

        attempts = self._attempt_events(outcome)
        best_attempt = int(outcome.best_attempt or 0)
        attempt = attempts.get(best_attempt) or {}
        evidence: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str]] = set()

        def add_call(tool_call: Dict[str, Any], *, source: str, source_attempt: int) -> bool:
            if not self._is_safe_readonly_evidence_call(tool_call):
                return False
            key = self._tool_call_key(tool_call)
            if not key[0] or key in seen:
                return True
            seen.add(key)
            evidence.append(
                self._compact_readonly_evidence_call(
                    tool_call,
                    source=source,
                    source_attempt=source_attempt,
                )
            )
            return True

        for call in attempt.get("inherited_tool_calls") or []:
            inherited_attempt = int(call.get("source_attempt", best_attempt) or best_attempt)
            add_call(call, source="inherited_readonly_evidence", source_attempt=inherited_attempt)

        for call in attempt.get("tool_calls") or outcome.final_tool_calls or []:
            if not add_call(call, source="best_attempt_readonly_prefix", source_attempt=best_attempt):
                break

        return evidence

    def _format_readonly_evidence_block(self, evidence: List[Dict[str, Any]]) -> str:
        if not evidence:
            return ""
        lines = ["Verified read-only evidence:"]
        for idx, call in enumerate(evidence, start=1):
            lines.append(f"{idx}. `{self._format_tool_call(call)}`")
            if "result" in call:
                result = call.get("result")
                if isinstance(result, (dict, list)):
                    result_text = json.dumps(result, ensure_ascii=False, default=str)
                else:
                    result_text = "" if result is None else str(result)
                lines.append(f"   result: {preview(result_text, 700)}")
        return "\n".join(lines)

    @staticmethod
    def _judgment_items(attempt: Dict[str, Any]) -> List[Dict[str, Any]]:
        feedback = (attempt.get("judge") or {}).get("feedback") or {}
        for key in ("judgment_results", "items", "checklist_results", "failed_items"):
            raw_items = feedback.get(key)
            if isinstance(raw_items, list):
                return [item for item in raw_items if isinstance(item, dict)]
        return []

    def _format_examples_plan_card(self, outcome: TurnOutcome) -> str:
        if float(outcome.score or 0.0) < 1.0:
            return ""
        attempts = self._attempt_events(outcome)
        attempt = attempts.get(int(outcome.best_attempt)) or {}
        score = float(outcome.score or 0.0)
        lines = [
            "## Reference plan card",
            "",
            f"Planner score: {score:.2f}",
            "",
        ]

        tool_calls = attempt.get("tool_calls") or outcome.final_tool_calls or []
        inherited_tool_calls = attempt.get("inherited_tool_calls") or []
        if inherited_tool_calls:
            lines.append("Verified facts from inherited read-only evidence:")
            for idx, call in enumerate(inherited_tool_calls, start=1):
                lines.append(f"{idx}. `{self._format_tool_call(call)}`")
                if "result" in call:
                    result = call.get("result")
                    if isinstance(result, (dict, list)):
                        result_text = json.dumps(result, ensure_ascii=False, default=str)
                    else:
                        result_text = "" if result is None else str(result)
                    lines.append(f"   result: {preview(result_text, 220)}")
            lines.append("")

        if tool_calls:
            lines.append("Suggested tool sequence:")
            for idx, call in enumerate(tool_calls, start=1):
                lines.append(f"{idx}. `{self._format_tool_call(call)}`")
            lines.append("")
        else:
            lines.extend(["Suggested tool sequence:", "- No tool call suggested.", ""])

        agent_response = str(attempt.get("agent_response") or "").strip()
        if agent_response:
            lines.append("Reference conclusion and target values:")
            lines.append(preview(agent_response, 700))
            lines.append("")

        judgment_items = self._judgment_items(attempt)

        failed_items = [
            item
            for item in judgment_items
            if str(item.get("status", "")).lower() == "failed"
        ]
        if not failed_items:
            feedback = (attempt.get("judge") or {}).get("feedback") or {}
            raw_failed = feedback.get("failed_items") or []
            failed_items = [item for item in raw_failed if isinstance(item, dict)]
        reasons = [
            str(item.get("reasoning") or item.get("description") or "").strip()
            for item in failed_items
        ]
        if reasons:
            lines.append("Known issue to avoid:")
            for reason in reasons[:3]:
                lines.append(f"- {reason[:240]}")
            lines.append("")

        lines.append("Use this only as planning guidance. Preserve verified ids/amounts/targets, but do not copy simulated wording into the final response.")
        return "\n".join(line for line in lines if line is not None).strip()

    def _format_examples(self, outcome: TurnOutcome) -> str:
        if float(outcome.score or 0.0) < 1.0:
            return ""
        return self._format_examples_plan_card(outcome)

    def generate_examples(
        self,
        *,
        current_message: str,
        context_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        real_tool_calls = context_payload.get("sync_real_tool_calls") or context_payload.get("real_tool_calls") or []
        current_config = self._sync_observed_tool_calls(real_tool_calls)
        solver = self._build_attempt_runner(current_config)
        history_payload = deepcopy(context_payload)
        if self.context_mode == "projected":
            history_payload["real_tool_calls"] = []
            history_payload.pop("sync_real_tool_calls", None)
        solver.set_history(history_payload)
        solver.set_initial_config(current_config)
        outcome = solver.process(current_message)
        examples_block = self._format_examples(outcome)
        readonly_evidence = self._accepted_readonly_evidence(outcome)
        return {
            "examples_block": examples_block,
            "readonly_evidence": readonly_evidence,
            "readonly_evidence_block": self._format_readonly_evidence_block(readonly_evidence),
            "score": outcome.score,
            "best_attempt": outcome.best_attempt,
        }

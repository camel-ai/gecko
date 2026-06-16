"""Internal tau2 LLM user simulator and assistant agent."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from camel.toolkits import FunctionTool

from benchmarks.taubench.config import DOMAIN_TOOLS, taubench_mock_openapi_dir
from benchmarks.taubench.adapter import TauBenchGATSManager
from benchmarks.taubench.context import Tau2ConversationRecorder
from benchmarks.taubench.prompts import (
    format_agent_system_prompt,
    format_example_injection,
    format_verified_readonly_evidence,
)
from benchmarks.taubench.internal.data_model.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from benchmarks.taubench.internal.environment.environment import Environment
from benchmarks.taubench.internal.utils import get_now
from inference.agents.chat_agent import ChatAgent

logger = logging.getLogger(__name__)

STOP_TOKENS = {"###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"}
USER_GUIDELINES_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "taubench"
    / "user_simulator"
    / "simulation_guidelines.md"
)


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
    return value


def _render_conversation_for_user(messages: list[Message], max_chars: int = 12000) -> str:
    lines: list[str] = []
    for message in messages:
        if isinstance(message, UserMessage) and message.has_text_content():
            lines.append(f"Customer: {message.content}")
        elif isinstance(message, AssistantMessage) and message.has_text_content():
            lines.append(f"Assistant: {message.content}")
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _tool_result_to_content(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(_jsonable(result), ensure_ascii=False, default=str)


@lru_cache(maxsize=8)
def _load_openapi_tool_descriptions(domain: str) -> dict[str, dict[str, Any]]:
    cfg = DOMAIN_TOOLS.get(domain)
    if cfg is None:
        return {}
    path = taubench_mock_openapi_dir() / cfg.mock_openapi_file
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Could not load tau-bench OpenAPI descriptions from %s", path, exc_info=True)
        return {}

    descriptions: dict[str, dict[str, Any]] = {}
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return descriptions

    for methods in paths.values():
        if not isinstance(methods, dict):
            continue
        for operation in methods.values():
            if not isinstance(operation, dict):
                continue
            operation_id = operation.get("operationId")
            if not isinstance(operation_id, str) or not operation_id:
                continue
            schema = (
                operation.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema", {})
            )
            properties = schema.get("properties") if isinstance(schema, dict) else None
            entry: dict[str, Any] = {}
            operation_description = operation.get("description")
            if isinstance(operation_description, str) and operation_description.strip():
                entry["description"] = " ".join(operation_description.split())
            if isinstance(properties, dict):
                entry["parameters"] = deepcopy(properties)
            if entry:
                descriptions[operation_id] = entry
    return descriptions


def _apply_openapi_parameter_descriptions(
    schema: dict[str, Any],
    *,
    domain: str,
    tool_name: str,
) -> dict[str, Any]:
    enriched = deepcopy(schema)
    openapi_tool = _load_openapi_tool_descriptions(domain).get(tool_name, {})
    if not isinstance(openapi_tool, dict) or not openapi_tool:
        return enriched

    function = enriched.get("function")
    parameters = function.get("parameters") if isinstance(function, dict) else None
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
    operation_description = openapi_tool.get("description")
    if isinstance(function, dict) and isinstance(operation_description, str):
        existing = function.get("description")
        if isinstance(existing, str) and operation_description not in existing:
            function["description"] = f"{existing.rstrip()}\n\nTool contract: {operation_description}"
        elif not existing:
            function["description"] = operation_description

    openapi_props = openapi_tool.get("parameters", {})
    if not isinstance(properties, dict) or not isinstance(openapi_props, dict):
        return enriched

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        source_schema = openapi_props.get(param_name)
        if not isinstance(source_schema, dict):
            continue
        description = source_schema.get("description")
        if isinstance(description, str) and description.strip() and not param_schema.get("description"):
            param_schema["description"] = " ".join(description.split())
    return enriched


def _make_function_tools(environment: Environment) -> list[FunctionTool]:
    tools = []
    domain = environment.get_domain_name()
    for tool in environment.get_tools():
        schema = _apply_openapi_parameter_descriptions(
            tool.openai_tool_schema,
            domain=domain,
            tool_name=tool.get_function_name(),
        )
        tools.append(FunctionTool(tool, openai_tool_schema=schema))
    return tools


def _clone_db(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_copy"):
        return value.model_copy(deep=True)
    return json.loads(json.dumps(_jsonable(value)))


def _snapshot_environment_state(environment: Environment) -> dict[str, Any]:
    return {
        "tools_db": _clone_db(environment.tools.db if environment.tools is not None else None),
        "user_tools_db": _clone_db(
            environment.user_tools.db if environment.user_tools is not None else None
        ),
    }


def _restore_environment_state(environment: Environment, snapshot: dict[str, Any]) -> None:
    if environment.tools is not None:
        environment.tools.db = _clone_db(snapshot.get("tools_db"))
    if environment.user_tools is not None:
        environment.user_tools.db = _clone_db(snapshot.get("user_tools_db"))
    environment.sync_tools()


def _make_tool_message_from_result(tool_call: ToolCall, result: Any) -> ToolMessage:
    return ToolMessage(
        id=tool_call.id,
        role="tool",
        content=_tool_result_to_content(result),
        error=False,
        timestamp=get_now(),
    )


def _replay_tool_messages_from_snapshot(
    environment: Environment,
    tool_calls: list[ToolCall],
    *,
    start_snapshot: dict[str, Any],
    final_snapshot: dict[str, Any],
) -> list[ToolMessage]:
    """Reconstruct immediate tool responses without changing final runtime state.

    CAMEL may report each tool call's result after later same-step mutations have
    already happened. tau2's official replay expects the response returned
    immediately after each call, so regenerate those responses from the pre-step
    DB snapshot and restore the actual final DB afterward.
    """

    try:
        _restore_environment_state(environment, start_snapshot)
        tool_messages: list[ToolMessage] = []
        for tool_call in tool_calls:
            tool_message = environment.get_response(tool_call)
            tool_message.timestamp = get_now()
            tool_messages.append(tool_message)
        return tool_messages
    finally:
        _restore_environment_state(environment, final_snapshot)


class InternalTau2UserSimulator:
    """LLM-backed user simulator using only in-repo task instructions."""

    def __init__(
        self,
        *,
        task: Any,
        model_name: str,
        timeout: int = 360,
    ) -> None:
        self.task = task
        self.model_name = model_name
        self.agent = ChatAgent(
            model_name=model_name,
            system_message=self._system_prompt(),
            timeout=timeout,
            max_iteration=1,
            agent_role="tau2_internal_user",
        )

    def _system_prompt(self) -> str:
        try:
            guidelines = USER_GUIDELINES_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            guidelines = (
                "# User Simulation Guidelines\n"
                "- Strictly follow the scenario instructions.\n"
                "- Never make up information not provided in the scenario.\n"
                "- Disclose information progressively.\n"
                "- Output ###STOP### when the goal is satisfied."
            )
        return (
            f"{guidelines}\n\n"
            "<scenario>\n"
            f"{self.task.user_scenario}\n"
            "</scenario>"
        )

    def next_message(self, messages: list[Message]) -> str | None:
        transcript = _render_conversation_for_user(messages)
        prompt = (
            "Conversation so far:\n"
            f"{transcript or '(none)'}\n\n"
            "Generate only the next customer message. Do not include role labels. "
            "Use the special stop/transfer/out-of-scope tokens exactly as instructed "
            "by the simulation guidelines when appropriate."
        )
        self.agent.reset()
        response = self.agent.generate_response(prompt)
        if not response.success:
            raise RuntimeError(response.error_message or "User simulator failed")
        text = (response.raw_response or "").strip()
        return text


class InternalTau2AssistantAgent:
    """Tool-calling assistant with optional GATS reference injection."""

    def __init__(
        self,
        *,
        domain: str,
        task: Any,
        environment: Environment,
        model_name: str,
        gats_mode: str = "none",
        context_mode: str = "projected",
        max_assistant_history_chars: int = 800,
        max_user_history_chars: int = 2000,
        max_tool_result_chars: int = 6000,
        project_all_tool_results: bool = False,
        gecko_url: str = "http://localhost:8000",
        gats_retries: int = 2,
        timeout: int = 360,
        debug: bool = False,
    ) -> None:
        self.domain = domain
        self.task = task
        self.environment = environment
        self.model_name = model_name
        self.recorder = Tau2ConversationRecorder(task_id=str(task.id), domain=domain)
        self.call_seq = 0
        self.gats_synthetic_read_calls: list[dict[str, Any]] = []
        self.manager: Optional[TauBenchGATSManager] = None
        mode = (gats_mode or "none").strip().lower()
        if mode not in {"none", "hybrid", "real", "mock"}:
            raise ValueError(f"Unsupported GATS mode {gats_mode!r}")
        self.gats_mode = mode
        if mode != "none":
            self.manager = TauBenchGATSManager(
                domain=domain,
                task=task,
                policy=environment.get_policy(),
                tau2_tools=environment.get_tools(),
                agent_model=model_name,
                gats_mode=mode,
                context_mode=context_mode,
                max_assistant_history_chars=max_assistant_history_chars,
                max_user_history_chars=max_user_history_chars,
                max_tool_result_chars=max_tool_result_chars,
                project_all_tool_results=project_all_tool_results,
                gecko_url=gecko_url,
                max_retries=gats_retries,
                agent_timeout=timeout,
                debug=debug,
            )

        system_prompt = format_agent_system_prompt(environment.get_policy())
        self.agent = ChatAgent(
            model_name=model_name,
            system_message=system_prompt,
            tools=_make_function_tools(environment),
            timeout=timeout,
            max_iteration=8,
            agent_role="tau2_internal_agent",
        )

    def _maybe_reference(self, user_content: str) -> tuple[str, list[dict[str, Any]]]:
        if self.manager is None:
            return "", []
        payload = self.recorder.get_solver_payload(
            current_message=user_content,
            mode=self.manager.context_mode,
            max_assistant_chars=self.manager.max_assistant_history_chars,
            max_user_chars=self.manager.max_user_history_chars,
            max_tool_result_chars=self.manager.max_tool_result_chars,
            project_all_tool_results=self.manager.project_all_tool_results,
        )
        try:
            result = self.manager.generate_examples(
                current_message=user_content,
                context_payload=payload,
            )
        except Exception as exc:
            logger.warning("Internal tau-bench GATS augmentation failed: %s", exc, exc_info=True)
            return "", []
        readonly_evidence = result.get("readonly_evidence") or []
        if not isinstance(readonly_evidence, list):
            readonly_evidence = []
        injections = [
            format_verified_readonly_evidence(result.get("readonly_evidence_block", "")),
            format_example_injection(result.get("examples_block", "")),
        ]
        return "\n\n".join(part for part in injections if part), deepcopy(readonly_evidence)

    def _next_call_id(self) -> str:
        self.call_seq += 1
        return f"call_{self.call_seq}"

    @staticmethod
    def _synthetic_read_call_key(call: dict[str, Any]) -> tuple[str, str]:
        name = str(call.get("function") or call.get("name") or "").strip()
        args = call.get("arguments") or call.get("args") or {}
        try:
            rendered_args = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            rendered_args = str(args)
        return name, rendered_args

    def get_gats_synthetic_read_calls(self) -> list[dict[str, Any]]:
        return deepcopy(self.gats_synthetic_read_calls)

    def get_gats_synthetic_action_messages(self) -> list[Message]:
        if not self.gats_synthetic_read_calls:
            return []

        tool_calls: list[ToolCall] = []
        seen: set[tuple[str, str]] = set()
        for idx, call in enumerate(self.gats_synthetic_read_calls, start=1):
            name = str(call.get("function") or call.get("name") or "").strip()
            if not name:
                continue
            key = self._synthetic_read_call_key(call)
            if key in seen:
                continue
            seen.add(key)
            args = call.get("arguments") or call.get("args") or {}
            tool_calls.append(
                ToolCall(
                    id=f"gats_read_{idx}",
                    name=name,
                    arguments=_jsonable(args) if isinstance(args, dict) else {},
                    requestor="assistant",
                    raw_data={
                        "synthetic_action_evidence": True,
                        "source": call.get("source"),
                        "source_attempt": call.get("source_attempt"),
                    },
                )
            )

        if not tool_calls:
            return []
        return [
            AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=tool_calls,
                raw_data={
                    "synthetic_action_evidence": True,
                    "source": "gats_readonly_evidence",
                },
            )
        ]

    def respond(self, user_message: UserMessage, messages: list[Message]) -> list[Message]:
        self.recorder.record_message(user_message)
        user_content = user_message.content or ""
        injection, synthetic_read_calls = self._maybe_reference(user_content)
        if synthetic_read_calls:
            self.gats_synthetic_read_calls.extend(synthetic_read_calls)
        prompt = f"{user_content}\n\n{injection}" if injection else user_content

        start_snapshot = _snapshot_environment_state(self.environment)
        response = self.agent.generate_response(prompt)
        final_snapshot = _snapshot_environment_state(self.environment)
        if not response.success:
            raise RuntimeError(response.error_message or "Assistant generation failed")

        tool_calls: list[ToolCall] = []
        raw_results: list[Any] = []
        for raw_call in response.tool_calls or []:
            name = raw_call.get("function") or raw_call.get("name")
            if not name:
                continue
            arguments = raw_call.get("arguments") or raw_call.get("args") or {}
            call_id = raw_call.get("id") or self._next_call_id()
            result = raw_call.get("result")
            tool_call = ToolCall(
                id=str(call_id),
                name=str(name),
                arguments=_jsonable(arguments) if isinstance(arguments, dict) else {},
            )
            tool_calls.append(tool_call)
            raw_results.append(result)

        tool_messages: list[ToolMessage] = []
        if tool_calls:
            try:
                tool_messages = _replay_tool_messages_from_snapshot(
                    self.environment,
                    tool_calls,
                    start_snapshot=start_snapshot,
                    final_snapshot=final_snapshot,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to reconstruct immediate tau2 tool responses; "
                    "falling back to reported CAMEL results: %s",
                    exc,
                    exc_info=True,
                )
                _restore_environment_state(self.environment, final_snapshot)
                tool_messages = [
                    _make_tool_message_from_result(tool_call, result)
                    for tool_call, result in zip(tool_calls, raw_results)
                ]

        assistant_message = AssistantMessage(
            role="assistant",
            content=response.raw_response or "",
            tool_calls=tool_calls or None,
            timestamp=get_now(),
        )
        produced: list[Message] = [assistant_message, *tool_messages]
        for message in produced:
            self.recorder.record_message(message)
        return produced

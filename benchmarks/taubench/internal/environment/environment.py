"""Minimal tau2-compatible environment replay for airline/retail."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from datetime import date, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel

from benchmarks.taubench.internal.data_model.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from benchmarks.taubench.internal.data_model.tasks import EnvAssertion, EnvFunctionCall, InitializationData
from benchmarks.taubench.internal.environment.db import DB
from benchmarks.taubench.internal.environment.toolkit import (
    ToolKitBase,
    ToolSignature,
    get_tool_signatures,
)

logger = logging.getLogger(__name__)


class Environment:
    def __init__(
        self,
        domain_name: str,
        policy: str,
        tools: Optional[ToolKitBase] = None,
        user_tools: Optional[ToolKitBase] = None,
        solo_mode: bool = False,
    ) -> None:
        self.domain_name = domain_name
        self.policy = policy
        self.tools = tools
        self.user_tools = user_tools
        self.solo_mode = solo_mode
        self.sync_tools()

    def get_domain_name(self) -> str:
        return self.domain_name

    def get_policy(self) -> str:
        return self.policy

    def get_tools(self) -> list[Any]:
        if self.tools is None:
            raise ValueError("Tools not available")
        return list(self.tools.get_tools().values())

    def get_user_tools(self, include: Optional[list[str]] = None) -> list[Any]:
        if self.user_tools is None:
            raise ValueError("User tools not available")
        return list(self.user_tools.get_tools(include=include).values())

    def _has_tool(self, tool_name: str) -> bool:
        return bool(
            (self.tools is not None and self.tools.has_tool(tool_name))
            or (self.user_tools is not None and self.user_tools.has_tool(tool_name))
        )

    def _is_mutating_tool(self, tool_name: str) -> bool:
        for toolkit in (self.tools, self.user_tools):
            if toolkit is not None and toolkit.has_tool(tool_name):
                return toolkit.tool_mutates_state(tool_name)
        return True

    def use_tool(self, tool_name: str, **kwargs: Any) -> Any:
        if self.tools is None:
            raise ValueError("Tools not available")
        return self.tools.use_tool(tool_name=tool_name, **kwargs)

    def use_user_tool(self, tool_name: str, **kwargs: Any) -> Any:
        if self.user_tools is None:
            raise ValueError("User tools not available")
        return self.user_tools.use_tool(tool_name=tool_name, **kwargs)

    def make_tool_call(
        self,
        tool_name: str,
        requestor: Literal["user", "assistant"] = "assistant",
        **kwargs: Any,
    ) -> Any:
        if requestor == "user":
            return self.use_user_tool(tool_name=tool_name, **kwargs)
        if requestor == "assistant":
            return self.use_tool(tool_name=tool_name, **kwargs)
        raise ValueError(f"Invalid requestor: {requestor}")

    def sync_tools(self) -> None:
        pass

    def run_env_function_call(self, env_function_call: EnvFunctionCall) -> Any:
        tool_kit = self.user_tools if env_function_call.env_type == "user" else self.tools
        if tool_kit is None:
            raise ValueError(f"No {env_function_call.env_type} toolkit")
        func = getattr(tool_kit, env_function_call.func_name)
        res = func(**env_function_call.arguments)
        self.sync_tools()
        return res

    def run_env_assertion(
        self,
        assertion: EnvAssertion,
        raise_assertion_error: bool = True,
    ) -> bool:
        res = self.run_env_function_call(assertion)
        if not isinstance(res, bool):
            raise ValueError(f"Assertion function returned {type(res)}")
        success = res == assertion.assert_value
        if raise_assertion_error:
            assert success, assertion.message or f"Assertion failed: {assertion}"
        return success

    def get_db_hash(self) -> Optional[str]:
        return None if self.tools is None else self.tools.get_db_hash()

    def get_user_db_hash(self) -> Optional[str]:
        return None if self.user_tools is None else self.user_tools.get_db_hash()

    def check_db(self, reference: DB) -> bool:
        return self.get_db_hash() == reference.get_hash()

    def check_user_db(self, reference: DB) -> bool:
        return self.get_user_db_hash() == reference.get_hash()

    def get_info(self, include_tool_info: bool = False) -> dict[str, Any]:
        return {
            "domain_name": self.domain_name,
            "policy": self.policy,
            "tool_defs": (
                get_tool_signatures(self.tools)
                if include_tool_info and self.tools is not None
                else None
            ),
            "user_tool_defs": (
                get_tool_signatures(self.user_tools)
                if include_tool_info and self.user_tools is not None
                else None
            ),
        }

    @staticmethod
    def _action_pairs(messages: list[Message]) -> list[tuple[ToolCall, ToolMessage]]:
        queue = deepcopy(messages)[::-1]
        actions: list[tuple[ToolCall, ToolMessage]] = []
        while queue:
            message = queue.pop()
            if isinstance(message, ToolMessage):
                raise ValueError("Tool message not expected before a tool call.")
            if isinstance(message, (AssistantMessage, UserMessage)) and message.is_tool_call():
                for tool_call in message.tool_calls or []:
                    if not queue:
                        raise ValueError("Tool message expected. Got None.")
                    tool_message = queue.pop()
                    if not isinstance(tool_message, ToolMessage):
                        raise ValueError(f"Tool message expected. Got {type(tool_message)}")
                    if tool_call.id != tool_message.id:
                        raise ValueError(
                            f"Tool call id mismatch. Got {tool_call.id} and {tool_message.id}"
                        )
                    actions.append((tool_call, tool_message))
        return actions

    def set_state(
        self,
        initialization_data: Optional[InitializationData],
        initialization_actions: Optional[list[EnvFunctionCall]],
        message_history: list[Message],
    ) -> None:
        if initialization_data is not None:
            if initialization_data.agent_data is not None and self.tools is not None:
                self.tools.update_db(initialization_data.agent_data)
                if self.user_tools is not None and self.user_tools.db is not None:
                    self.user_tools.db = self.tools.db
            if initialization_data.user_data is not None and self.user_tools is not None:
                self.user_tools.update_db(initialization_data.user_data)
                if self.tools is not None and self.tools.db is not None:
                    self.tools.db = self.user_tools.db

        for action in initialization_actions or []:
            self.run_env_function_call(action)

        for tool_call, expected_response in self._action_pairs(message_history):
            if not self._has_tool(tool_call.name):
                logger.debug("Skipping unknown replay tool %s", tool_call.name)
                continue
            if not self._is_mutating_tool(tool_call.name):
                continue
            response = self.get_response(tool_call)
            try:
                content = json.loads(response.content)
            except json.JSONDecodeError:
                content = response.content
            try:
                expected_content = json.loads(expected_response.content)
            except json.JSONDecodeError:
                expected_content = expected_response.content
            if content != expected_content:
                raise ValueError(
                    f"Tool call:\n{tool_call}\n\nReturned:\n{response}\n\n"
                    f"Expected:\n{expected_response}"
                )
        self.sync_tools()

    @classmethod
    def to_json_str(cls, resp: Any) -> str:
        def process(value: Any) -> Any:
            if isinstance(value, BaseModel):
                return value.model_dump(mode="json", exclude_none=False)
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, list):
                return [process(item) for item in value]
            if isinstance(value, tuple):
                return [process(item) for item in value]
            if isinstance(value, dict):
                return {key: process(item) for key, item in value.items()}
            if isinstance(value, (datetime, date)):
                return value.isoformat()
            return str(value)

        return resp if isinstance(resp, str) else json.dumps(process(resp), default=str)

    def get_response(self, message: ToolCall) -> ToolMessage:
        error = False
        try:
            resp = self.make_tool_call(
                message.name,
                requestor=message.requestor,
                **message.arguments,
            )
            self.sync_tools()
        except Exception as exc:
            resp = f"Error: {exc}"
            error = True
        return ToolMessage(
            id=message.id,
            content=self.to_json_str(resp),
            requestor=message.requestor,
            role="tool",
            error=error,
        )

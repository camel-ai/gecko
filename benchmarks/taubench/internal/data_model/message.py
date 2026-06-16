"""Minimal tau2 message models for in-repo airline/retail evaluation."""

from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.taubench.internal.utils import get_now

SystemRole = Literal["system"]
UserRole = Literal["user"]
AssistantRole = Literal["assistant"]
ToolRole = Literal["tool"]
ToolRequestor = UserRole | AssistantRole
ParticipantRole = UserRole | AssistantRole


class SystemMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: SystemRole = Field(description="The role of the message sender.")
    content: Optional[str] = Field(default=None)
    turn_idx: Optional[int] = None
    timestamp: Optional[str] = Field(default_factory=get_now)


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(default="")
    name: str
    arguments: dict = Field(default_factory=dict)
    requestor: ToolRequestor = "assistant"

    def __str__(self) -> str:
        return (
            f"ToolCall (from {self.requestor})\n"
            f"id: {self.id}\n"
            f"name: {self.name}\n"
            f"arguments:\n{json.dumps(self.arguments, indent=2)}"
        )


class ParticipantMessageBase(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    turn_idx: Optional[int] = None
    timestamp: Optional[str] = Field(default_factory=get_now)
    raw_data: Optional[dict] = None

    def has_content(self) -> bool:
        return self.content is not None and bool(str(self.content).strip())

    def has_text_content(self) -> bool:
        return self.has_content()

    def is_tool_call(self) -> bool:
        return self.tool_calls is not None


class AssistantMessage(ParticipantMessageBase):
    role: AssistantRole

    @classmethod
    def text(
        cls,
        content: str,
        *,
        tool_calls: Optional[list[ToolCall]] = None,
        raw_data: Optional[dict] = None,
        generation_time_seconds: Optional[float] = None,
    ) -> "AssistantMessage":
        return cls(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            raw_data=raw_data,
            generation_time_seconds=generation_time_seconds,
        )


class UserMessage(ParticipantMessageBase):
    role: UserRole

    @classmethod
    def text(
        cls,
        content: str,
        *,
        tool_calls: Optional[list[ToolCall]] = None,
        raw_data: Optional[dict] = None,
        generation_time_seconds: Optional[float] = None,
    ) -> "UserMessage":
        return cls(
            role="user",
            content=content,
            tool_calls=tool_calls,
            raw_data=raw_data,
            generation_time_seconds=generation_time_seconds,
        )


class ToolMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    role: ToolRole
    content: Optional[str] = None
    requestor: ToolRequestor = "assistant"
    error: bool = False
    turn_idx: Optional[int] = None
    timestamp: Optional[str] = Field(default_factory=get_now)


class MultiToolMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: ToolRole
    tool_messages: list[ToolMessage]


APICompatibleMessage = SystemMessage | AssistantMessage | UserMessage | ToolMessage
Message = SystemMessage | AssistantMessage | UserMessage | ToolMessage | MultiToolMessage
EnvironmentMessage = ToolMessage | MultiToolMessage
ValidInputMessage = UserMessage | AssistantMessage | EnvironmentMessage


class Tick(BaseModel):
    """Minimal full-duplex placeholder so official-style evaluator types import."""

    model_config = ConfigDict(extra="allow")

    tick_id: int
    timestamp: str
    agent_chunk: Optional[AssistantMessage] = None
    user_chunk: Optional[UserMessage] = None
    agent_tool_calls: list[ToolCall] = Field(default_factory=list)
    user_tool_calls: list[ToolCall] = Field(default_factory=list)
    agent_tool_results: list[ToolMessage] = Field(default_factory=list)
    user_tool_results: list[ToolMessage] = Field(default_factory=list)

"""Minimal tau2 simulation/reward models for local evaluation."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from benchmarks.taubench.internal.data_model.message import Message, Tick
from benchmarks.taubench.internal.data_model.tasks import Action, EnvAssertion, RewardType


class TerminationReason(str, Enum):
    AGENT_STOP = "agent_stop"
    USER_STOP = "user_stop"
    MAX_STEPS = "max_steps"
    MAX_STEPS_SECONDS = "max_steps_seconds"
    TOO_MANY_ERRORS = "too_many_errors"
    ERROR = "error"


class DBCheck(BaseModel):
    db_match: bool
    db_reward: float


class EnvAssertionCheck(BaseModel):
    env_assertion: EnvAssertion
    met: bool
    reward: float


class ActionCheck(BaseModel):
    action: Action
    action_match: bool
    action_reward: float
    tool_type: Optional[Any] = None


class CommunicateCheck(BaseModel):
    info: str
    met: bool
    justification: str


class NLAssertionCheck(BaseModel):
    nl_assertion: str
    met: bool
    justification: str


class RewardInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    reward: float
    db_check: Optional[DBCheck] = None
    env_assertions: Optional[list[EnvAssertionCheck]] = None
    action_checks: Optional[list[ActionCheck]] = None
    communicate_checks: Optional[list[CommunicateCheck]] = None
    nl_assertions: Optional[list[NLAssertionCheck]] = None
    reward_basis: Optional[list[RewardType]] = None
    reward_breakdown: Optional[dict[RewardType, float]] = None
    info: Optional[dict[str, Any]] = None


class SimulationRun(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    task_id: Optional[str] = None
    trial: Optional[int] = None
    messages: list[Message] = Field(default_factory=list)
    ticks: list[Tick] = Field(default_factory=list)
    termination_reason: TerminationReason = TerminationReason.AGENT_STOP
    reward_info: Optional[RewardInfo] = None

    @field_validator("messages", "ticks", mode="before")
    @classmethod
    def _none_to_empty_list(cls, value):
        return [] if value is None else value

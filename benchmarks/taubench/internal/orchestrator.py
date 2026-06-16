"""Self-contained half-duplex tau-bench orchestrator.

This implements the text-mode flow used by airline/retail evaluations: user message, assistant
message with optional tool calls/results, repeat until the user simulator stops
or the step cap is reached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from benchmarks.taubench.internal.data_model.message import AssistantMessage, Message, UserMessage
from benchmarks.taubench.internal.data_model.simulation import SimulationRun, TerminationReason

DEFAULT_FIRST_ASSISTANT_MESSAGE = "Hi! How can I help you today?"
USER_STOP_MARKERS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")


def _is_user_stop(content: str | None) -> bool:
    text = str(content or "")
    return any(marker in text for marker in USER_STOP_MARKERS)


class Tau2UserLike(Protocol):
    def next_message(self, messages: list[Message]) -> str | None:
        """Return the next user utterance, or None to stop."""


class Tau2AgentLike(Protocol):
    def respond(self, user_message: UserMessage, messages: list[Message]) -> list[Message]:
        """Return assistant/tool messages produced for this user turn."""


@dataclass
class InternalTau2Orchestrator:
    task_id: str
    user: Tau2UserLike
    agent: Tau2AgentLike
    max_steps: int = 200
    first_assistant_message: str | None = DEFAULT_FIRST_ASSISTANT_MESSAGE

    def run(self, *, trial: int = 0, seed: int | None = None) -> SimulationRun:
        messages: list[Message] = []
        if self.first_assistant_message:
            messages.append(
                AssistantMessage(
                    role="assistant",
                    content=self.first_assistant_message,
                )
            )
        termination_reason = TerminationReason.MAX_STEPS

        for turn_idx in range(self.max_steps):
            content = self.user.next_message(messages)
            if content is None or not str(content).strip():
                termination_reason = TerminationReason.USER_STOP
                break

            user_message = UserMessage(role="user", content=str(content).strip(), turn_idx=turn_idx)
            messages.append(user_message)
            if _is_user_stop(user_message.content):
                termination_reason = TerminationReason.USER_STOP
                break

            try:
                produced = self.agent.respond(user_message, messages)
            except Exception as exc:
                assistant_error = AssistantMessage(
                    role="assistant",
                    content=f"Internal orchestrator error: {type(exc).__name__}: {exc}",
                    turn_idx=turn_idx,
                )
                messages.append(assistant_error)
                termination_reason = TerminationReason.ERROR
                break

            messages.extend(produced)
        else:
            termination_reason = TerminationReason.MAX_STEPS

        final_messages: list[Message] = []
        for idx, message in enumerate(messages):
            copied = message.model_copy(deep=True)
            copied.turn_idx = idx
            final_messages.append(copied)

        return SimulationRun(
            task_id=self.task_id,
            trial=trial,
            seed=seed,
            messages=final_messages,
            termination_reason=termination_reason,
        )

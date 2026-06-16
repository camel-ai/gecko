"""Self-contained tau2 evaluator for the in-repo airline/retail subset."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Callable, Optional

from benchmarks.taubench.internal.data_model.message import AssistantMessage, Message, ToolCall, UserMessage
from benchmarks.taubench.internal.data_model.simulation import (
    ActionCheck,
    CommunicateCheck,
    DBCheck,
    EnvAssertionCheck,
    NLAssertionCheck,
    RewardInfo,
    SimulationRun,
    TerminationReason,
)
from benchmarks.taubench.internal.data_model.tasks import Action, RewardType, Task
from benchmarks.taubench.internal.environment.environment import Environment
from benchmarks.taubench.internal.environment.toolkit import ToolType, get_tool_types
from benchmarks.taubench.internal.loader import get_environment_constructor


class EvaluationType(str, Enum):
    ENV = "env"
    COMMUNICATE = "communicate"
    ACTION = "action"
    ALL = "all"
    NL_ASSERTIONS = "nl_assertions"
    ALL_WITH_NL_ASSERTIONS = "all_with_nl_assertions"
    ALL_IGNORE_BASIS = "all_ignore_basis"
    ALL_WITH_NL_ASSERTIONS_IGNORE_BASIS = "all_with_nl_assertions_ignore_basis"


def _extract_tool_calls(full_trajectory: list[Message]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for message in full_trajectory:
        if isinstance(message, (AssistantMessage, UserMessage)) and message.is_tool_call():
            calls.extend(message.tool_calls or [])
    return calls


def _check_actions(
    predicted_tool_calls: list[ToolCall],
    golden_actions: list[Action],
    tool_types: Optional[dict[str, ToolType]] = None,
) -> list[ActionCheck]:
    checks = []
    for action in golden_actions:
        found = any(action.compare_with_tool_call(call) for call in predicted_tool_calls)
        tool_type = tool_types.get(action.name) if tool_types is not None else None
        checks.append(
            ActionCheck(
                action=action,
                action_match=found,
                action_reward=1.0 if found else 0.0,
                tool_type=tool_type,
            )
        )
    return checks


def evaluate_actions(
    task: Task,
    full_trajectory: list[Message],
    tool_types: Optional[dict[str, ToolType]] = None,
) -> RewardInfo:
    actions = task.evaluation_criteria.actions if task.evaluation_criteria else None
    if not actions:
        return RewardInfo(
            reward=1.0,
            info={"note": "No actions to evaluate"},
            reward_breakdown={RewardType.ACTION: 1.0},
        )
    checks = _check_actions(_extract_tool_calls(full_trajectory), actions, tool_types)
    reward = 1.0 if all(check.action_match for check in checks) else 0.0
    return RewardInfo(
        reward=reward,
        action_checks=checks,
        reward_breakdown={RewardType.ACTION: reward},
    )


def evaluate_communicate(task: Task, full_trajectory: list[Message]) -> RewardInfo:
    communicate_info = (
        task.evaluation_criteria.communicate_info if task.evaluation_criteria else None
    )
    if not communicate_info:
        return RewardInfo(
            reward=1.0,
            info={"note": "No communicate_info to evaluate"},
            reward_breakdown={RewardType.COMMUNICATE: 1.0},
        )

    checks: list[CommunicateCheck] = []
    for expected in communicate_info:
        found = False
        justification = f"Information '{expected}' not communicated."
        for message in full_trajectory:
            if not isinstance(message, AssistantMessage) or not message.has_text_content():
                continue
            content = (message.content or "").lower().replace(",", "")
            if expected.lower() in content:
                found = True
                justification = (
                    f"Information '{expected}' communicated in the message:\n"
                    f" '{message.content}'"
                )
                break
        checks.append(
            CommunicateCheck(info=expected, met=found, justification=justification)
        )
    reward = 1.0 if all(check.met for check in checks) else 0.0
    return RewardInfo(
        reward=reward,
        communicate_checks=checks,
        reward_breakdown={RewardType.COMMUNICATE: reward},
    )


def evaluate_env(
    *,
    environment_constructor: Callable[..., Environment],
    task: Task,
    full_trajectory: list[Message],
    solo_mode: bool = False,
) -> RewardInfo:
    if task.evaluation_criteria is None:
        return RewardInfo(reward=1.0, info={"note": "No evaluation criteria"})
    expected_actions = task.evaluation_criteria.actions
    env_assertions = task.evaluation_criteria.env_assertions
    if expected_actions is None and env_assertions is None:
        return RewardInfo(
            reward=1.0,
            db_check=DBCheck(db_match=True, db_reward=1.0),
            info={"note": "No expected actions or env assertions"},
        )

    initial_state = task.initial_state
    initialization_data = initial_state.initialization_data if initial_state else None
    initialization_actions = initial_state.initialization_actions if initial_state else None
    initial_history = initial_state.message_history if initial_state else []

    predicted = environment_constructor(solo_mode=solo_mode)
    predicted.set_state(
        initialization_data=initialization_data,
        initialization_actions=initialization_actions,
        message_history=list(full_trajectory),
    )

    gold = environment_constructor()
    gold.set_state(
        initialization_data=initialization_data,
        initialization_actions=initialization_actions,
        message_history=list(initial_history or []),
    )
    for action in task.evaluation_criteria.actions or []:
        try:
            gold.make_tool_call(action.name, requestor=action.requestor, **action.arguments)
        except Exception:
            pass

    db_match = (
        gold.get_db_hash() == predicted.get_db_hash()
        and gold.get_user_db_hash() == predicted.get_user_db_hash()
    )
    db_reward = 1.0 if db_match else 0.0
    db_check = DBCheck(db_match=db_match, db_reward=db_reward)

    assertion_checks = []
    assertion_reward = 1.0
    for assertion in task.evaluation_criteria.env_assertions or []:
        met = predicted.run_env_assertion(assertion, raise_assertion_error=False)
        check = EnvAssertionCheck(
            env_assertion=assertion,
            met=met,
            reward=1.0 if met else 0.0,
        )
        assertion_checks.append(check)
        assertion_reward *= check.reward

    reward = 1.0
    breakdown: dict[RewardType, float] = {}
    if RewardType.DB in task.evaluation_criteria.reward_basis:
        breakdown[RewardType.DB] = db_reward
        reward *= db_reward
    if RewardType.ENV_ASSERTION in task.evaluation_criteria.reward_basis:
        breakdown[RewardType.ENV_ASSERTION] = assertion_reward
        reward *= assertion_reward
    return RewardInfo(
        reward=reward,
        db_check=db_check,
        env_assertions=assertion_checks,
        reward_basis=task.evaluation_criteria.reward_basis,
        reward_breakdown=breakdown,
    )


def evaluate_nl_assertions(
    task: Task,
    full_trajectory: list[Message],
    *,
    judge_model: Optional[str] = None,
) -> RewardInfo:
    assertions = task.evaluation_criteria.nl_assertions if task.evaluation_criteria else None
    if not assertions:
        return RewardInfo(
            reward=1.0,
            nl_assertions=[],
            info={"note": "No nl_assertions to evaluate"},
            reward_breakdown={RewardType.NL_ASSERTION: 1.0},
        )
    if not judge_model:
        raise ValueError("NL assertions require judge_model")

    from inference.agents.chat_agent import ChatAgent

    trajectory = "\n".join(
        f"{message.role}: {getattr(message, 'content', '')}"
        for message in full_trajectory
        if getattr(message, "content", None)
    )
    agent = ChatAgent(
        model_name=judge_model,
        system_message=(
            "Evaluate each expected outcome against the conversation. "
            "Return JSON with a results list; each item has expectedOutcome, "
            "reasoning, and metExpectation boolean."
        ),
        timeout=360,
    )
    response = agent.generate_response(
        "conversation:\n"
        f"{trajectory}\n\n"
        "expectedOutcomes:\n"
        f"{json.dumps(assertions, ensure_ascii=False)}"
    )
    if not response.success:
        raise RuntimeError(response.error_message or "NL assertion judge failed")
    raw_response = response.raw_response or ""
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        from utils.model_utils import sanitize_llm_json_text

        data = json.loads(sanitize_llm_json_text(raw_response))
    checks = _parse_nl_assertion_checks(data)
    reward = 1.0 if checks and all(check.met for check in checks) else 0.0
    return RewardInfo(
        reward=reward,
        nl_assertions=checks,
        reward_breakdown={RewardType.NL_ASSERTION: reward},
    )


def _parse_nl_assertion_checks(data: Any) -> list[NLAssertionCheck]:
    if isinstance(data, dict):
        raw_results = data.get("results", [])
    elif isinstance(data, list):
        raw_results = data
    else:
        raise ValueError("NL assertion judge returned JSON that is not an object or list")

    if raw_results is None:
        raw_results = []
    if not isinstance(raw_results, list):
        raise ValueError("NL assertion judge results must be a list")

    checks: list[NLAssertionCheck] = []
    for item in raw_results:
        if not isinstance(item, dict):
            raise ValueError("NL assertion judge result items must be objects")
        checks.append(
            NLAssertionCheck(
                nl_assertion=str(item.get("expectedOutcome", "")),
                met=bool(item.get("metExpectation")),
                justification=str(item.get("reasoning", "")),
            )
        )
    return checks


def evaluate_simulation(
    simulation: SimulationRun,
    task: Task,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    solo_mode: bool = False,
    domain: str = "airline",
    nl_judge_model: Optional[str] = None,
    synthetic_action_messages: Optional[list[Message]] = None,
) -> RewardInfo:
    if simulation.termination_reason not in {
        TerminationReason.AGENT_STOP,
        TerminationReason.USER_STOP,
    }:
        return RewardInfo(
            reward=0.0,
            info={
                "note": (
                    "Simulation terminated prematurely. Termination reason: "
                    f"{simulation.termination_reason.value}"
                )
            },
        )
    if task.evaluation_criteria is None:
        return RewardInfo(reward=1.0, info={"note": "No evaluation criteria"})

    env_ctor = get_environment_constructor(domain)
    tool_types = None
    try:
        env = env_ctor(solo_mode=solo_mode)
        if env.tools is not None:
            tool_types = get_tool_types(env.tools)
        if env.user_tools is not None:
            user_types = get_tool_types(env.user_tools)
            tool_types = user_types if tool_types is None else {**tool_types, **user_types}
    except Exception:
        tool_types = None

    trajectory = simulation.messages
    synthetic_action_messages = synthetic_action_messages or []
    action_trajectory = list(synthetic_action_messages) + list(trajectory)
    synthetic_action_call_count = len(_extract_tool_calls(synthetic_action_messages))

    def with_synthetic_action_metadata(info: RewardInfo, real_action_info: RewardInfo) -> RewardInfo:
        if synthetic_action_call_count <= 0:
            return info
        merged_info = dict(info.info or {})
        merged_info.update(
            {
                "synthetic_action_tool_calls_count": synthetic_action_call_count,
                "action_reward_real_only": real_action_info.reward,
                "action_reward_augmented": info.reward,
            }
        )
        return info.model_copy(
            update={
                "info": merged_info,
                "action_checks_real_only": real_action_info.action_checks,
                "action_reward_real_only": real_action_info.reward,
                "action_reward_augmented": info.reward,
                "synthetic_action_tool_calls_count": synthetic_action_call_count,
            }
        )

    if evaluation_type == EvaluationType.ENV:
        return evaluate_env(
            environment_constructor=env_ctor,
            task=task,
            full_trajectory=trajectory,
            solo_mode=solo_mode,
        )
    if evaluation_type == EvaluationType.ACTION:
        action_info = evaluate_actions(task, action_trajectory, tool_types)
        real_action_info = evaluate_actions(task, trajectory, tool_types)
        return with_synthetic_action_metadata(action_info, real_action_info)
    if evaluation_type == EvaluationType.COMMUNICATE:
        return evaluate_communicate(task, trajectory)
    if evaluation_type == EvaluationType.NL_ASSERTIONS:
        return evaluate_nl_assertions(task, trajectory, judge_model=nl_judge_model)

    env_info = evaluate_env(
        environment_constructor=env_ctor,
        task=task,
        full_trajectory=trajectory,
        solo_mode=solo_mode,
    )
    action_info = evaluate_actions(task, action_trajectory, tool_types)
    real_action_info = evaluate_actions(task, trajectory, tool_types)
    communicate_info = evaluate_communicate(task, trajectory)
    nl_info = None
    needs_nl = RewardType.NL_ASSERTION in task.evaluation_criteria.reward_basis
    if evaluation_type in {
        EvaluationType.NL_ASSERTIONS,
        EvaluationType.ALL_WITH_NL_ASSERTIONS,
        EvaluationType.ALL_WITH_NL_ASSERTIONS_IGNORE_BASIS,
    } or needs_nl:
        nl_info = evaluate_nl_assertions(task, trajectory, judge_model=nl_judge_model)

    basis = set(task.evaluation_criteria.reward_basis)
    if evaluation_type in {
        EvaluationType.ALL_IGNORE_BASIS,
        EvaluationType.ALL_WITH_NL_ASSERTIONS_IGNORE_BASIS,
    }:
        basis = {RewardType.DB, RewardType.ENV_ASSERTION, RewardType.ACTION, RewardType.COMMUNICATE}
        if nl_info is not None:
            basis.add(RewardType.NL_ASSERTION)

    reward = 1.0
    breakdown: dict[RewardType, float] = {}
    if {RewardType.DB, RewardType.ENV_ASSERTION} & basis:
        for key, value in (env_info.reward_breakdown or {}).items():
            if key in basis:
                breakdown[key] = value
                reward *= value
    if RewardType.ACTION in basis:
        value = action_info.reward
        breakdown[RewardType.ACTION] = value
        reward *= value
    if RewardType.COMMUNICATE in basis:
        value = communicate_info.reward
        breakdown[RewardType.COMMUNICATE] = value
        reward *= value
    if RewardType.NL_ASSERTION in basis and nl_info is not None:
        breakdown[RewardType.NL_ASSERTION] = nl_info.reward
        reward *= nl_info.reward

    extra_fields = {}
    info = None
    if synthetic_action_call_count > 0:
        info = {
            "synthetic_action_tool_calls_count": synthetic_action_call_count,
            "action_reward_real_only": real_action_info.reward,
            "action_reward_augmented": action_info.reward,
        }
        extra_fields = {
            "action_checks_real_only": real_action_info.action_checks,
            "action_reward_real_only": real_action_info.reward,
            "action_reward_augmented": action_info.reward,
            "synthetic_action_tool_calls_count": synthetic_action_call_count,
        }

    return RewardInfo(
        reward=reward,
        db_check=env_info.db_check,
        env_assertions=env_info.env_assertions,
        action_checks=action_info.action_checks,
        communicate_checks=communicate_info.communicate_checks,
        nl_assertions=nl_info.nl_assertions if nl_info is not None else None,
        reward_basis=task.evaluation_criteria.reward_basis,
        reward_breakdown=breakdown,
        info=info,
        **extra_fields,
    )

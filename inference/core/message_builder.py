import json
from typing import Any, Dict, List, Tuple

from utils.conversation import render_conversation
from utils.reasoning import strip_reasoning


class MessageBuilder:
    """Builds agent-facing messages, retry contexts, and state renderings."""

    @staticmethod
    def _truncate_text(value: Any, max_chars: int) -> str:
        text = str(value)
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "... [truncated]"

    @staticmethod
    def build_enhanced_message(
        user_message: str,
        attempt: int,
        previous_attempts: list,
        current_state: Dict[str, Any],
        history_items: List[Dict[str, Any]],
        inherited_tool_calls: List[Dict[str, Any]] | None = None,
    ) -> str:
        """Build the agent-facing message: History + State + [Current Task] (+ retry context).

        Args:
            user_message: Raw user request for this turn.
            attempt: Current attempt index (0-based).
            previous_attempts: List of _AttemptState from prior attempts.
            current_state: Authoritative turn-start state dict.
            history_items: Conversation history items from prior turns.
        """
        parts: List[str] = []
        inherited_tool_calls = inherited_tool_calls or []
        inherited_keys = {
            MessageBuilder._tool_call_key(tc)
            for tc in inherited_tool_calls
            if isinstance(tc, dict)
        } if attempt > 0 else set()

        if history_items:
            parts.append("=== Previous Conversation History ===")
            history_view = render_conversation(
                history_items,
                max_items=30,
                include_tool_calls=True,
                include_results=True,
                truncate_assistant=None,
                truncate_result=None,
            )
            suppressed_history_evidence = 0
            suppress_next_tool_result_for: str | None = None
            for item in history_view:
                role = item.get("role", "")
                if role == "user":
                    suppress_next_tool_result_for = None
                    parts.append(f"User: {item.get('content', '')}")
                elif role == "assistant":
                    suppress_next_tool_result_for = None
                    parts.append(f"Assistant: {item.get('content', '')}")
                elif role == "tool_call":
                    tool_key = MessageBuilder._tool_call_key(item)
                    if inherited_keys and tool_key in inherited_keys:
                        suppressed_history_evidence += 1
                        suppress_next_tool_result_for = str(
                            item.get("function") or item.get("name") or ""
                        )
                        continue
                    suppress_next_tool_result_for = None
                    func_name = item.get("function") or item.get("name", "unknown")
                    args = item.get("arguments") or item.get("args", {})
                    args_str = json.dumps(args, indent=2) if isinstance(args, dict) else str(args)
                    parts.append(f"Tool Call: {func_name}")
                    parts.append(f"Arguments: {args_str}")
                    if "result" in item:
                        result = item.get("result")
                        result_str = json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)
                        parts.append(f"Result: {result_str}")
                elif role == "tool_result":
                    func_name = item.get("function") or item.get("name", "unknown")
                    if (
                        inherited_keys
                        and suppress_next_tool_result_for
                        and str(func_name) == suppress_next_tool_result_for
                    ):
                        suppressed_history_evidence += 1
                        suppress_next_tool_result_for = None
                        continue
                    suppress_next_tool_result_for = None
                    result = item.get("result", {})
                    result_str = json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)
                    parts.append(f"Tool Result ({func_name}): {result_str}")

            if suppressed_history_evidence:
                parts.append(
                    f"(Omitted {suppressed_history_evidence} repeated read-only evidence item(s); "
                    "see the retry evidence block below.)"
                )
            parts.append("")

        parts.append("[Current Task]")
        parts.append(user_message)

        if attempt > 0 and previous_attempts:
            parts.append("")
            parts.append(
                MessageBuilder.build_retry_context(
                    attempt,
                    previous_attempts,
                    inherited_tool_calls=inherited_tool_calls,
                )
            )

        return "\n".join(parts)

    @staticmethod
    def build_retry_context(
        attempt: int,
        previous_attempts: list,
        inherited_tool_calls: List[Dict[str, Any]] | None = None,
    ) -> str:
        """Build retry context from all previous attempts, with last-attempt detail."""
        last_attempt = previous_attempts[-1]
        retry_memory = MessageBuilder.collect_retry_memory(previous_attempts)
        inherited_tool_calls = inherited_tool_calls or []
        lines = ["Problematic solution from previous attempt:"]

        if inherited_tool_calls:
            lines.append("")
            lines.append("Inherited read-only evidence from prior attempts:")
            lines.append(
                "- These calls were verified against the same turn-start state and had no state effects."
            )
            lines.append(
                "- Use them as evidence in this retry. Do not repeat them unless a needed fact is missing or conflicts."
            )
            lines.append("- State-changing effects from prior attempts are not inherited.")
            for i, tc in enumerate(inherited_tool_calls, 1):
                func = tc.get("function", "unknown")
                args = tc.get("arguments", {}) or {}
                if isinstance(args, dict) and "requestBody" in args and isinstance(args.get("requestBody"), dict):
                    args = args.get("requestBody") or {}
                if args:
                    arg_items = list(args.items())
                    args_str = ", ".join(
                        f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}"
                        for k, v in arg_items[:3]
                    )
                    if len(arg_items) > 3:
                        args_str += ", ..."
                    call_repr = f"{func}({args_str})"
                else:
                    call_repr = f"{func}()"
                lines.append(f"  {i}. {call_repr}")
                if "result" in tc:
                    lines.append(f"     result: {MessageBuilder._truncate_text(tc.get('result'), 220)}")
        inherited_keys = {
            MessageBuilder._tool_call_key(tc)
            for tc in inherited_tool_calls
            if isinstance(tc, dict)
        }

        lines.append("- Tool calls:")
        rendered_count = 0
        if last_attempt.tool_calls:
            for tc in last_attempt.tool_calls:
                if inherited_keys and MessageBuilder._tool_call_key(tc) in inherited_keys:
                    continue
                rendered_count += 1
                func = tc.get("function", "unknown")
                args = tc.get("arguments", {}) or {}
                if isinstance(args, dict) and "requestBody" in args and isinstance(args.get("requestBody"), dict):
                    args = args.get("requestBody") or {}
                result = tc.get("result", {})

                if args:
                    arg_items = list(args.items())
                    args_str = ", ".join(
                        f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}"
                        for k, v in arg_items[:3]
                    )
                    if len(arg_items) > 3:
                        args_str += ", ..."
                    call_repr = f"{func}({args_str})"
                else:
                    call_repr = f"{func}()"

                lines.append(f"  {rendered_count}. {call_repr}")

                if result:
                    result_str = MessageBuilder._truncate_text(result, 120)
                    lines.append(f"     result: {result_str}")
                else:
                    lines.append("     result: (no result)")
        if last_attempt.tool_calls and inherited_keys and rendered_count == 0:
            lines.append("   (same read-only calls as inherited evidence; no additional calls)")
        elif not last_attempt.tool_calls:
            lines.append("   (none)")

        lines.append("- Response:")
        response_text = strip_reasoning(last_attempt.agent_response or "")
        response_text = MessageBuilder._truncate_text(response_text, 1200) if response_text else "(empty)"
        lines.append(f"   {response_text}")

        lines.append("")
        lines.append("Judge findings (aggregated across previous attempts):")
        failed_items = retry_memory.get("failed_items", [])
        if failed_items:
            for idx, item in enumerate(failed_items[:5], start=1):
                desc = item.get("description", "Unknown requirement")
                reason = item.get("reasoning", "")
                from_attempt = item.get("attempt")
                suffix = f" (from attempt {from_attempt})" if from_attempt is not None else ""
                lines.append(f"  {idx}. {desc}{suffix}")
                if reason:
                    lines.append(f"     - reason: {MessageBuilder._truncate_text(reason, 700)}")
        else:
            lines.append("  1. No explicit failed checklist items were returned.")

        lines.append("")
        lines.append("Retry needed:")
        if inherited_tool_calls:
            lines.append("  1. This retry runs in a fresh execution session but may reuse the inherited read-only evidence listed above.")
            lines.append("  2. Previous turns' completed effects are included in the authoritative current state; prior attempt state-changing effects are not inherited.")
            lines.append("  3. Resolve all unresolved judge findings in this retry, preferably by using inherited evidence when it already proves the needed fact.")
            lines.append("  4. If the remaining issue is missing user-facing explanation or policy conclusion, answer directly instead of repeating the inherited reads.")
            lines.append("  5. If a judge finding says parameter values were fabricated, re-call using the user's exact words — unless the parameter truly cannot be filled from the user's message and has no default, in which case do not call the tool.")
            lines.append("  6. Do not repeat a rejected tool call unchanged. If feedback says omit/remove a key, remove that key entirely in the corrected call.")
            lines.append("  7. If the task still needs a tool call, make the corrected tool call; do not answer with prose or raw JSON instead.")
        else:
            lines.append("  1. This retry runs in a fresh session but starts from the same turn-start state shown above.")
            lines.append("  2. Previous turns' completed effects are included in the authoritative current state.")
            lines.append("  3. Resolve all unresolved judge findings in this retry.")
            lines.append("  4. If a judge finding says parameter values were fabricated, re-call using the user's exact words — unless the parameter truly cannot be filled from the user's message and has no default, in which case do not call the tool.")
            lines.append("  5. Do not repeat a rejected tool call unchanged. If feedback says omit/remove a key, remove that key entirely in the corrected call.")
            lines.append("  6. If the task still needs a tool call, make the corrected tool call; do not answer with prose or raw JSON instead.")
        lines.append("")
        lines.append("Please correct the issues above and solve the current task.")
        return "\n".join(lines)

    @staticmethod
    def _tool_call_key(tool_call: Dict[str, Any]) -> Tuple[str, str]:
        func = str(tool_call.get("function") or tool_call.get("name") or "").strip()
        args = tool_call.get("arguments") or tool_call.get("args") or {}
        try:
            args_str = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            args_str = str(args)
        return func, args_str

    @staticmethod
    def collect_retry_memory(previous_attempts: list) -> Dict[str, List[Dict[str, Any]]]:
        """Aggregate failed checklist items across attempts."""
        failed_items: List[Dict[str, Any]] = []
        seen_failed: set[Tuple[str, str]] = set()

        for attempt_state in previous_attempts or []:
            feedback = attempt_state.feedback if isinstance(attempt_state.feedback, dict) else {}
            raw_failed = feedback.get("failed_items", [])
            if isinstance(raw_failed, list):
                for item in raw_failed:
                    if not isinstance(item, dict):
                        continue
                    desc = str(item.get("description", "") or "").strip()
                    reason = str(item.get("reasoning", "") or "").strip()
                    if not desc:
                        continue
                    key = (desc, reason)
                    if key in seen_failed:
                        continue
                    seen_failed.add(key)
                    failed_items.append(
                        {
                            "description": desc,
                            "reasoning": reason,
                            "attempt": attempt_state.attempt,
                        }
                    )

        return {"failed_items": failed_items}

    @staticmethod
    def render_authoritative_state(current_state: Dict[str, Any], max_chars: int = 20000) -> str:
        """Render authoritative state in deterministic JSON with bounded size."""
        state = current_state if isinstance(current_state, dict) else {}
        try:
            text = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True)
        except Exception:
            text = str(state)

        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]
        return f"{truncated}\n... [truncated authoritative state]"

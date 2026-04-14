import json
from typing import Any, Dict, List, Optional, Tuple

from utils.conversation import render_conversation


class MessageBuilder:
    """Builds agent-facing messages, retry contexts, and state renderings."""

    @staticmethod
    def build_enhanced_message(
        user_message: str,
        attempt: int,
        previous_attempts: list,
        current_state: Dict[str, Any],
        history_items: List[Dict[str, Any]],
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
            for item in history_view:
                role = item.get("role", "")
                if role == "user":
                    parts.append(f"User: {item.get('content', '')}")
                elif role == "assistant":
                    parts.append(f"Assistant: {item.get('content', '')}")
                elif role == "tool_call":
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
                    result = item.get("result", {})
                    result_str = json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)
                    parts.append(f"Tool Result ({func_name}): {result_str}")

            parts.append("")

        parts.append("[Current Task]")
        parts.append(user_message)

        if attempt > 0 and previous_attempts:
            parts.append("")
            parts.append(MessageBuilder.build_retry_context(attempt, previous_attempts))

        return "\n".join(parts)

    @staticmethod
    def build_retry_context(attempt: int, previous_attempts: list) -> str:
        """Build retry context from all previous attempts, with last-attempt detail."""
        last_attempt = previous_attempts[-1]
        retry_memory = MessageBuilder.collect_retry_memory(previous_attempts)
        lines = ["Problematic solution from previous attempt:"]

        lines.append("- Tool calls:")
        if last_attempt.tool_calls:
            for i, tc in enumerate(last_attempt.tool_calls, 1):
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

                lines.append(f"  {i}. {call_repr}")

                if result:
                    result_str = str(result)
                    if len(result_str) > 150:
                        result_str = result_str[:150] + "... [truncated]"
                    lines.append(f"     result: {result_str}")
                else:
                    lines.append("     result: (no result)")
        else:
            lines.append("   (none)")

        lines.append("- Response:")
        response_text = (last_attempt.agent_response or "").strip()
        lines.append(f"   {response_text if response_text else '(empty)'}")

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
                    lines.append(f"     - reason: {reason}")
        else:
            lines.append("  1. No explicit failed checklist items were returned.")

        lines.append("")
        lines.append("Known invalid calls to avoid (aggregated):")
        invalid_calls = retry_memory.get("invalid_calls", [])
        if invalid_calls:
            for idx, item in enumerate(invalid_calls[:8], start=1):
                signature = item.get("signature", "<unknown_call>")
                error = item.get("error", "")
                from_attempt = item.get("attempt")
                suffix = f" (from attempt {from_attempt})" if from_attempt is not None else ""
                lines.append(f"  {idx}. {signature}{suffix}")
                if error:
                    lines.append(f"     - error: {error}")
        else:
            lines.append("  1. No invalid calls recorded.")

        lines.append("")
        lines.append("Retry needed:")
        lines.append("  1. This retry runs in a fresh session but starts from the same turn-start state shown above.")
        lines.append("  2. Previous turns' completed effects are included in the authoritative current state.")
        lines.append("  3. Resolve all unresolved judge findings in this retry.")
        lines.append("  4. Do not repeat known-invalid calls from prior attempts.")
        lines.append("  5. If a judge finding says parameter values were fabricated, re-call using the user's exact words — unless the parameter truly cannot be filled from the user's message and has no default, in which case do not call the tool.")
        lines.append("")
        lines.append("Please 改正上述问题，解决当前的task。")
        return "\n".join(lines)

    @staticmethod
    def collect_retry_memory(previous_attempts: list) -> Dict[str, List[Dict[str, Any]]]:
        """Aggregate failed checklist items and invalid tool calls across attempts."""
        failed_items: List[Dict[str, Any]] = []
        invalid_calls: List[Dict[str, Any]] = []
        seen_failed: set[Tuple[str, str]] = set()
        seen_invalid: set[Tuple[str, str]] = set()

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

            for call in attempt_state.tool_calls or []:
                if not isinstance(call, dict):
                    continue
                err = MessageBuilder.extract_error_text(call.get("result"))
                if not err:
                    continue
                signature = MessageBuilder.format_tool_call_signature(call)
                key = (signature, err)
                if key in seen_invalid:
                    continue
                seen_invalid.add(key)
                invalid_calls.append(
                    {
                        "signature": signature,
                        "error": err,
                        "attempt": attempt_state.attempt,
                    }
                )

        return {"failed_items": failed_items, "invalid_calls": invalid_calls}

    @staticmethod
    def format_tool_call_signature(call: Dict[str, Any]) -> str:
        """Format a compact, stable signature for a tool call."""
        func = str(call.get("function", "unknown"))
        args = call.get("arguments", {})
        if isinstance(args, dict) and "requestBody" in args and isinstance(args.get("requestBody"), dict):
            args = args.get("requestBody") or {}
        if not isinstance(args, dict) or not args:
            return f"{func}()"

        parts: List[str] = []
        for idx, (k, v) in enumerate(args.items()):
            if idx >= 4:
                parts.append("...")
                break
            if isinstance(v, str):
                parts.append(f"{k}='{v}'")
            else:
                parts.append(f"{k}={v}")
        return f"{func}({', '.join(parts)})"

    @staticmethod
    def extract_error_text(result: Any) -> Optional[str]:
        """Best-effort extraction of tool-call error text."""
        if isinstance(result, dict):
            if result.get("error"):
                return str(result.get("error"))
            detail = result.get("detail")
            if isinstance(detail, dict) and detail.get("error_message"):
                return str(detail.get("error_message"))
            if isinstance(detail, str) and detail:
                return detail
            if result.get("success") is False:
                return str(result.get("message") or "Tool returned success=false")
        if isinstance(result, str) and "error" in result.lower():
            return result
        return None

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

"""Conversation recording for tau2-bench messages."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple


IMPORTANT_RESULT_KEYS = {
    "id",
    "user_id",
    "reservation_id",
    "booking_id",
    "order_id",
    "flight_number",
    "flight_id",
    "origin",
    "destination",
    "departure_airport",
    "arrival_airport",
    "departure_time",
    "arrival_time",
    "date",
    "status",
    "source",
    "cabin",
    "seat",
    "passenger",
    "passengers",
    "payment",
    "payment_id",
    "price",
    "prices",
    "fare",
    "total",
    "amount",
    "balance",
    "membership",
    "insurance",
    "created_at",
    "flight_type",
    "scheduled_departure_time_est",
    "scheduled_arrival_time_est",
    "available_seats",
    "payment_methods",
    "saved_passengers",
    "total_baggages",
    "nonfree_baggages",
    "first_name",
    "last_name",
    "dob",
    "last_four",
    "brand",
    "email",
    "name",
    "phone",
    "address",
    "baggages",
    "certificate",
}


class Tau2ConversationRecorder:
    """Record tau2 messages in the compact schema consumed by GATS simulation."""

    def __init__(self, *, task_id: str, domain: str):
        self.task_id = task_id
        self.domain = domain
        self.messages: List[Dict[str, Any]] = []
        self.real_tool_calls: List[Dict[str, Any]] = []

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump(mode="json", exclude_none=False)
            except TypeError:
                return value.model_dump()
        if isinstance(value, list):
            return [Tau2ConversationRecorder._to_jsonable(item) for item in value]
        if isinstance(value, tuple):
            return [Tau2ConversationRecorder._to_jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(k): Tau2ConversationRecorder._to_jsonable(v) for k, v in value.items()}
        return value

    @staticmethod
    def _parse_tool_result(content: Any) -> Any:
        if isinstance(content, str):
            try:
                return json.loads(content)
            except Exception:
                return content
        return Tau2ConversationRecorder._to_jsonable(content)

    def record_message(self, message: Any) -> None:
        """Record a tau2 message by duck-typing its role and fields."""
        role = getattr(message, "role", None)
        if role == "tool" and hasattr(message, "tool_messages"):
            for tool_message in getattr(message, "tool_messages", []) or []:
                self._record_tool_message(tool_message)
            return

        if role == "tool":
            self._record_tool_message(message)
            return

        if role not in {"user", "assistant"}:
            return

        entry: Dict[str, Any] = {"role": role}
        for attr in ("turn_idx", "timestamp"):
            value = getattr(message, attr, None)
            if value is not None:
                entry[attr] = self._to_jsonable(value)

        content = getattr(message, "content", None)
        if content:
            entry["content"] = content

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = [
                {
                    "id": getattr(tc, "id", None),
                    "function": getattr(tc, "name", None),
                    "arguments": self._to_jsonable(getattr(tc, "arguments", {}) or {}),
                }
                for tc in tool_calls
            ]

        if "content" in entry or "tool_calls" in entry:
            self.messages.append(entry)

    def _record_tool_message(self, tool_message: Any) -> None:
        tool_call_id = getattr(tool_message, "id", None)
        if not tool_call_id:
            return

        result = self._parse_tool_result(getattr(tool_message, "content", None))
        tool_meta = {
            "turn_idx": getattr(tool_message, "turn_idx", None),
            "timestamp": getattr(tool_message, "timestamp", None),
            "requestor": getattr(tool_message, "requestor", None),
            "error": getattr(tool_message, "error", None),
        }
        matched_call: Optional[Dict[str, Any]] = None
        for message in reversed(self.messages):
            if message.get("role") != "assistant":
                continue
            for tool_call in message.get("tool_calls", []) or []:
                if tool_call.get("id") == tool_call_id:
                    tool_call["result"] = result
                    tool_call["tool_message"] = {
                        key: self._to_jsonable(value)
                        for key, value in tool_meta.items()
                        if value is not None
                    }
                    matched_call = tool_call
                    break
            if matched_call is not None:
                break

        if matched_call is None:
            return

        self.real_tool_calls.append(
            {
                "seq_no": len(self.real_tool_calls) + 1,
                "tool_id": tool_call_id,
                "name": matched_call.get("function"),
                "args": deepcopy(matched_call.get("arguments") or {}),
                "result": deepcopy(result),
                "requestor": getattr(tool_message, "requestor", None),
                "error": getattr(tool_message, "error", None),
                "turn_idx": getattr(tool_message, "turn_idx", None),
                "timestamp": getattr(tool_message, "timestamp", None),
            }
        )

    @staticmethod
    def _json_size(value: Any) -> int:
        try:
            return len(json.dumps(value, ensure_ascii=False, default=str))
        except Exception:
            return len(str(value))

    @staticmethod
    def _truncate_text(value: str, max_chars: int) -> str:
        if max_chars <= 0 or len(value) <= max_chars:
            return value
        return value[:max_chars] + f"... [truncated {len(value) - max_chars} chars]"

    @classmethod
    def _project_value(
        cls,
        value: Any,
        *,
        max_chars: int,
        max_list_items: int = 5,
        depth: int = 0,
        force: bool = False,
    ) -> Any:
        """Project large tool results while keeping IDs and action-relevant fields.

        The projection is deterministic and schema-agnostic. It avoids
        unbounded tool-result payloads in GATS simulation history.
        """
        if not force and max_chars > 0 and cls._json_size(value) <= max_chars:
            return deepcopy(value)

        if depth > 4:
            text = str(value)
            return cls._truncate_text(text, min(max_chars, 800) if max_chars > 0 else 800)

        if isinstance(value, str):
            return cls._truncate_text(value, min(max_chars, 1000) if max_chars > 0 else 1000)

        if isinstance(value, list):
            projected_items = [
                cls._project_value(
                    item,
                    max_chars=max(500, max_chars // max(1, max_list_items)),
                    max_list_items=max_list_items,
                    depth=depth + 1,
                    force=force,
                )
                for item in value[:max_list_items]
            ]
            result: Dict[str, Any] = {
                "type": "list_projection",
                "count": len(value),
                "items": projected_items,
            }
            if len(value) > max_list_items:
                result["omitted_count"] = len(value) - max_list_items
            return result

        if isinstance(value, dict):
            projected: Dict[str, Any] = {}
            scalar_overflow: Dict[str, Any] = {}
            for key, item in value.items():
                key_str = str(key)
                key_norm = key_str.lower()
                is_important = key_norm in IMPORTANT_RESULT_KEYS or key_norm.endswith("_id")
                if is_important or isinstance(item, (dict, list)):
                    projected[key_str] = cls._project_value(
                        item,
                        max_chars=max(500, max_chars // 3),
                        max_list_items=max_list_items,
                        depth=depth + 1,
                        force=force,
                    )
                elif isinstance(item, (str, int, float, bool)) or item is None:
                    scalar_overflow[key_str] = item

            remaining_budget = max_chars - cls._json_size(projected) if max_chars > 0 else 0
            if remaining_budget > 200:
                for key, item in scalar_overflow.items():
                    candidate = deepcopy(projected)
                    candidate[key] = cls._project_value(
                        item,
                        max_chars=min(remaining_budget, 500),
                        max_list_items=max_list_items,
                        depth=depth + 1,
                        force=force,
                    )
                    if cls._json_size(candidate) > max_chars:
                        break
                    projected = candidate

            omitted = max(0, len(value) - len(projected))
            if omitted:
                projected["_projection_note"] = f"omitted {omitted} low-priority fields"
            return projected

        return deepcopy(value)

    @classmethod
    def _project_tool_result(
        cls,
        result: Any,
        max_chars: int,
        *,
        force: bool = False,
    ) -> Tuple[Any, bool]:
        if max_chars <= 0 or (not force and cls._json_size(result) <= max_chars):
            return deepcopy(result), False
        projected = cls._project_value(result, max_chars=max_chars, force=force)
        return projected, True

    def _structured_conversation(
        self,
        *,
        current_message: Optional[str] = None,
        projected: bool = False,
        max_assistant_chars: int = 800,
        max_user_chars: int = 2000,
        max_tool_result_chars: int = 6000,
        project_all_tool_results: bool = False,
    ) -> List[Dict[str, Any]]:
        structured: List[Dict[str, Any]] = []
        skip_current_idx: Optional[int] = None
        if current_message:
            for idx in range(len(self.messages) - 1, -1, -1):
                message = self.messages[idx]
                if message.get("role") == "user" and str(message.get("content") or "") == current_message:
                    skip_current_idx = idx
                    break

        for message in self.messages:
            if skip_current_idx is not None and message is self.messages[skip_current_idx]:
                continue
            role = message.get("role")
            content = message.get("content")
            if role and content:
                rendered_content = content
                if projected and isinstance(rendered_content, str):
                    max_chars = max_assistant_chars if role == "assistant" else max_user_chars
                    rendered_content = self._truncate_text(rendered_content, max_chars)
                structured.append({"role": role, "content": content})
                if projected:
                    structured[-1]["content"] = rendered_content
            for tool_call in message.get("tool_calls", []) or []:
                call_entry = {
                    "role": "tool_call",
                    "function": tool_call.get("function"),
                    "arguments": deepcopy(tool_call.get("arguments") or {}),
                }
                if "result" in tool_call:
                    result = tool_call.get("result")
                    if projected:
                        projected_result, _was_projected = self._project_tool_result(
                            result,
                            max_tool_result_chars,
                            force=project_all_tool_results,
                        )
                        call_entry["result"] = projected_result
                    else:
                        call_entry["result"] = deepcopy(result)
                structured.append(call_entry)

        return structured

    def get_payload(self) -> Dict[str, Any]:
        structured = self._structured_conversation()
        return {
            "is_real_context": True,
            "structured_conversation": structured,
            "messages": deepcopy(self.messages),
            "real_tool_calls": deepcopy(self.real_tool_calls),
            "task_id": self.task_id,
            "domain": self.domain,
        }

    def get_solver_payload(
        self,
        *,
        current_message: Optional[str] = None,
        mode: str = "full",
        max_assistant_chars: int = 800,
        max_user_chars: int = 2000,
        max_tool_result_chars: int = 6000,
        project_all_tool_results: bool = False,
    ) -> Dict[str, Any]:
        """Return the conversation payload to pass into GATS simulation.

        ``full`` preserves the previous behavior. ``projected`` keeps raw sync
        inputs under ``sync_real_tool_calls`` while compacting the history that
        GATS renders in prompts.
        """
        mode = (mode or "full").strip().lower()
        if mode not in {"full", "projected"}:
            raise ValueError(f"Unsupported tau2 context mode: {mode}")
        if mode == "full":
            structured = self._structured_conversation()
            return {
                "is_real_context": True,
                "structured_conversation": structured,
                "messages": deepcopy(self.messages),
                "real_tool_calls": deepcopy(self.real_tool_calls),
                "task_id": self.task_id,
                "domain": self.domain,
            }

        structured = self._structured_conversation(
            current_message=current_message,
            projected=True,
            max_assistant_chars=max_assistant_chars,
            max_user_chars=max_user_chars,
            max_tool_result_chars=max_tool_result_chars,
            project_all_tool_results=project_all_tool_results,
        )
        return {
            "is_real_context": True,
            "context_mode": mode,
            "structured_conversation": structured,
            "messages": [],
            "real_tool_calls": deepcopy(self.real_tool_calls),
            "sync_real_tool_calls": deepcopy(self.real_tool_calls),
            "task_id": self.task_id,
            "domain": self.domain,
        }

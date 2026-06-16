from typing import Any, Dict, Iterable, List, Optional


def render_conversation(
    history: Iterable[Dict[str, Any]],
    *,
    max_items: Optional[int] = None,
    include_tool_calls: bool = True,
    include_results: bool = False,
    text_only: bool = False,
    truncate_assistant: Optional[int] = 800,
    truncate_result: Optional[int] = 500,
    drop_fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Return a filtered/truncated view of conversation history.

    Args:
        history: Full conversation items (untruncated).
        max_items: Keep only the last N items.
        include_tool_calls: Keep tool_call entries.
        include_results: Keep tool_result entries and result payloads.
        text_only: Keep only user/assistant messages.
        truncate_assistant: Max chars for assistant content (None to disable).
        truncate_result: Max chars for tool result string (None to disable).
        drop_fields: Extra keys to remove from each item.
    """
    items = list(history)
    if max_items is not None and max_items >= 0:
        items = items[-max_items:]

    rendered: List[Dict[str, Any]] = []
    for item in items:
        role = item.get("role")

        if text_only:
            if role not in {"user", "assistant"}:
                continue
        else:
            if role == "tool_call" and not include_tool_calls:
                continue
            if role == "tool_result" and not include_results:
                continue

        new_item = dict(item)

        if role == "assistant" and truncate_assistant is not None:
            content = new_item.get("content")
            if isinstance(content, str) and len(content) > truncate_assistant:
                new_item["content"] = content[:truncate_assistant] + "... [truncated]"

        if role == "tool_result" and not include_results:
            continue

        if role == "tool_result" and include_results and truncate_result is not None:
            result = new_item.get("result")
            if result is not None:
                result_str = str(result)
                if len(result_str) > truncate_result:
                    new_item["result"] = result_str[:truncate_result] + "... [truncated]"

        if drop_fields:
            for key in drop_fields:
                new_item.pop(key, None)

        rendered.append(new_item)

    return rendered

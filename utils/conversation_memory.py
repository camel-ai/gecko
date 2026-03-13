import hashlib
import json
import re
import secrets
import threading
from typing import Any, Dict, Optional, Tuple

_BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _base62_id(length: int = 9) -> str:
    return "".join(secrets.choice(_BASE62_ALPHABET) for _ in range(length))


def _json_dumps_compact(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


def _truncate_single_line(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"[\r\n\t]+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max(0, max_chars - 3)] + "..."


def _summarize_json_for_hint(json_str: str, *, sha8: str, max_chars: int = 300) -> str:
    try:
        parsed = json.loads(json_str)
    except Exception:
        return _truncate_single_line(f"type=unparsed bytes={len(json_str.encode('utf-8'))} sha={sha8}", max_chars)

    bytes_len = len(json_str.encode("utf-8"))
    if parsed is None:
        return _truncate_single_line(f"type=null bytes={bytes_len} sha={sha8}", max_chars)

    if isinstance(parsed, bool):
        return _truncate_single_line(f"type=bool value={parsed} bytes={bytes_len} sha={sha8}", max_chars)
    if isinstance(parsed, (int, float)):
        return _truncate_single_line(f"type=number value={parsed} bytes={bytes_len} sha={sha8}", max_chars)
    if isinstance(parsed, str):
        return _truncate_single_line(f"type=string len={len(parsed)} bytes={bytes_len} sha={sha8}", max_chars)

    if isinstance(parsed, list):
        elem_types = []
        for elem in parsed[:10]:
            elem_types.append(type(elem).__name__)
        hint = f"type=list len={len(parsed)} elem_types={elem_types} bytes={bytes_len} sha={sha8}"
        return _truncate_single_line(hint, max_chars)

    if isinstance(parsed, dict):
        keys = list(parsed.keys())
        keys_sorted = sorted(str(k) for k in keys)
        keys_preview = keys_sorted[:20]
        hint = f"type=object keys_total={len(keys)} keys={keys_preview} bytes={bytes_len} sha={sha8}"
        return _truncate_single_line(hint, max_chars)

    return _truncate_single_line(f"type={type(parsed).__name__} bytes={bytes_len} sha={sha8}", max_chars)


class ConversationMemoryStore:
    """Conversation-scope store for large tool call results.

    Results are stored as compact JSON strings and can be retrieved via the
    built-in get_memory tool.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_id: Dict[str, str] = {}
        self._sha_to_id: Dict[str, str] = {}

    def put_json(self, json_str: str) -> Tuple[str, int, str]:
        """Store a compact JSON string and return (memory_id, bytes, sha8)."""
        sha = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
        sha8 = sha[:8]
        bytes_len = len(json_str.encode("utf-8"))
        with self._lock:
            existing = self._sha_to_id.get(sha)
            if existing:
                return existing, bytes_len, sha8

            while True:
                memory_id = f"m_{_base62_id(9)}"
                if memory_id not in self._by_id:
                    break

            self._by_id[memory_id] = json_str
            self._sha_to_id[sha] = memory_id

        return memory_id, bytes_len, sha8

    def get_memory(self, memory_id: str) -> str:
        """Retrieve the original JSON string for a stored tool result.

        Args:
            memory_id: Memory identifier from TOOL_RESULT_REF

        Returns:
            The stored compact JSON string. If the id is unknown, returns an error string.
        """
        with self._lock:
            payload = self._by_id.get(memory_id)
        if payload is None:
            print(f"[GET_MEMORY] id={memory_id} found=false")
            return "ERROR: unknown memory_id"
        print(f"[GET_MEMORY] id={memory_id} found=true bytes={len(payload.encode('utf-8'))}")
        return payload

    def fold_result(
        self,
        *,
        function_name: str,
        result: Any,
        fold_threshold_bytes: int = 200,
        hint_max_chars: int = 300,
    ) -> Any:
        """Fold a tool result into a TOOL_RESULT_REF line if it is large enough."""
        json_str = _json_dumps_compact(result)
        bytes_len = len(json_str.encode("utf-8"))
        if bytes_len <= fold_threshold_bytes:
            return result

        memory_id, bytes_len, sha8 = self.put_json(json_str)
        hint = _summarize_json_for_hint(json_str, sha8=sha8, max_chars=hint_max_chars)
        hint_escaped = hint.replace("\\", "\\\\").replace('"', '\\"')
        return (
            f'TOOL_RESULT_REF function={function_name} id={memory_id} bytes={bytes_len} '
            f'sha={sha8} hint="{hint_escaped}"'
        )

import re

_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.S | re.I)
_UNCLOSED_THINK = re.compile(r"<think\b[^>]*>.*$", re.S | re.I)


def strip_reasoning(text: str) -> str:
    """Remove `<think>...</think>` reasoning so it does not pollute downstream prompts.

    Closed blocks are deleted. An unclosed `<think>` (typically caused by a
    max_tokens cutoff mid-reasoning) discards everything from the open tag
    onward — the model never produced a usable conclusion.
    """
    if not text:
        return text
    text = _THINK_BLOCK.sub("", text)
    text = _UNCLOSED_THINK.sub("", text)
    return text.strip()

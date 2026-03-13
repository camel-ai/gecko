"""BFCL tool filtering helpers.

The filter is intentionally conservative: it only removes tools that are
clearly irrelevant to the task, and keeps uncertain tools.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from inference.agents.chat_agent import ChatAgent
from utils.model_utils import sanitize_llm_json_text

logger = logging.getLogger(__name__)


TOOL_FILTER_SYSTEM_PROMPT = """You are a strict tool relevance filter for function-calling tasks.

Your job is to identify tools that are definitely irrelevant to the current task.

Rules:
1. Only mark a tool as irrelevant if it clearly cannot help solve the task.
2. If there is any reasonable uncertainty, keep the tool.
3. Judge by the tool's intended semantics from its name, description, and parameters.
4. Superficial parameter overlap is not enough. For example, a BMI calculator is not relevant to a triangle-area task just because both mention height.
5. Do not assume hidden transformations or repurposing beyond the tool description.
6. If a tool description or parameter description restricts the tool to a specific API, domain, dataset, or resource, and the task clearly asks for a different API, domain, dataset, or resource, mark the tool as irrelevant.
7. A generic transport method such as GET/POST/request is irrelevant when its descriptions show it is tied to a specific API/domain that does not match the task.
8. If the user message is only meta-instructions, role-setting, or non-actionable chat, concrete business tools are irrelevant.
9. If the task requires a capability that none of the tools support at all (for example: delete when tools only add/list/update; historical minimum when tools only expose current/latest/alerts), mark all such tools as irrelevant.
10. OUTPUT TYPE MATCH (critical): A tool is irrelevant if it computes or returns a DIFFERENT quantity/type of answer than what the user asks for, even if the input parameters overlap. Check: does the tool's described output match the user's question? Examples of mismatches:
    - User asks for travel TIME -> tool returns DISTANCE -> irrelevant
    - User asks for a COUNT -> tool returns DESCRIPTIONS -> irrelevant
    - User asks for rate of return -> tool calculates net PROFIT -> irrelevant
    - User asks for standard deviation -> tool calculates P-VALUE -> irrelevant
    - User asks "who won" -> tool shows BRACKETS/schedule -> irrelevant
11. CONCEPTUAL vs OPERATIONAL: If the user asks a conceptual "how to" or "explain" question, and the tool is an operational function that requires specific numeric inputs the user did not provide, the tool is irrelevant.
12. REAL-TIME DATA: If the user asks about "today", "now", or "current" and no date/time is determinable from context, a tool requiring a specific date/time input is irrelevant.
13. SCOPE BOUNDARY: A tool is scoped to exactly what its description says. Do not treat a tool designed for scope A as usable for related-but-different scope B:
    - "quadratic equation solver" is irrelevant for a linear equation (different equation type).
    - "blood cell classifier by shape/size" is irrelevant for "which cell causes clotting" (role-based lookup is a different capability).
    - "case lookup by case_id" is irrelevant for "find most impactful cases" (search/ranking is a different capability from ID-based retrieval).
    - "lawsuit search" is irrelevant for "what are traffic laws" (lawsuits are not statutes/regulations).
14. FABRICATED INPUTS: If calling a tool would require the caller to invent values that are not in the user's message and are not common knowledge (e.g., fabricating a database name, a case ID, or specific numeric inputs for a symbolic/variable question), the tool is irrelevant. Using placeholder values like 0 or 1 for missing required inputs counts as fabrication.
    - "Who won the world series in 2018?" + generic SQL query tool (no known schema) -> irrelevant (caller would have to guess table/column names).
    - "What are the roots of bx + c = 0?" + quadratic solver requiring concrete a, b, c numbers -> irrelevant (question uses symbolic variables, not specific numbers).
    - "What is the final price after a 25% discount and 10% tax?" + calculateFinalPrice(price, discount, tax) -> irrelevant (no specific price given; using price=1 as placeholder is fabrication).
    - "What is the penalty for burglary in California?" + get_penalty(crime, state) -> relevant (values come directly from the question).
15. ANSWER-AS-INPUT: If the user asks to IDENTIFY a value (e.g., "what timezone is X in?", "what type of cell is Y?") and that value is a required INPUT to the tool rather than its output, the tool is irrelevant -- it cannot tell you what you need to provide it.
16. QUERY vs GENERATE: If the user asks about existing state ("who has", "what is the current") and the tool creates/generates new state (e.g., deals a new hand, starts a new simulation) rather than querying existing state, the tool is irrelevant.
17. Return only the exact tool names that are definitely irrelevant.

Examples:
  - Task: "My son's latest goal?"
    Tools: weather lookup, news search, recipe search
    Decision: all irrelevant
    Why: the task asks for a private personal fact; tools access unrelated public domains.

  - Task: "How long would it take to travel from Boston to New York by car?"
    Tools: calculate_distance(coords, speed) -> returns distance in miles
    Decision: irrelevant
    Why: user asks for travel TIME; tool returns DISTANCE (different output type).

  - Task: "What is the rate of return for revenue $15000 and cost $22000?"
    Tools: calculate_profit(revenue, cost) -> returns net profit
    Decision: irrelevant
    Why: user asks for rate of RETURN (a ratio); tool calculates net PROFIT (a dollar amount).

  - Task: "How can I increase the population of deer in a forest?"
    Tools: calculate_population_growth(current_population, birth_rate, death_rate)
    Decision: irrelevant
    Why: user asks for strategies/methods ("how to increase"); tool only computes a projection from given rates (different capability).

Output JSON only in this format:
{"irrelevant_tools": ["tool_a", "tool_b"]}
"""


META_ONLY_PATTERNS = [
    re.compile(r"^\s*you are a helpful assistant\s*$", re.IGNORECASE),
    re.compile(
        r"^\s*don't make assumptions about what values to plug into functions\.",
        re.IGNORECASE,
    ),
]


def _is_meta_only_task(task: str) -> bool:
    text = (task or "").strip()
    if not text:
        return True
    return any(pattern.search(text) for pattern in META_ONLY_PATTERNS)


def _prefilter_definitely_irrelevant_tools(
    task: str, tools: List[Dict[str, Any]]
) -> List[str]:
    """Deterministic conservative prefilter for obviously irrelevant cases."""
    if _is_meta_only_task(task):
        return [tool.get("name", "") for tool in tools if tool.get("name")]
    return []


def _build_filter_prompt(task: str, tools: List[Dict[str, Any]]) -> str:
    tool_blocks: List[str] = []
    for tool in tools:
        name = tool.get("name", "")
        description = tool.get("description", "")
        params = tool.get("parameters", {}).get("properties", {})
        param_lines = []
        for param_name, param_schema in params.items():
            ptype = param_schema.get("type", "unknown")
            pdesc = param_schema.get("description", "")
            param_lines.append(f"- {param_name} ({ptype}): {pdesc}")
        params_text = "\n".join(param_lines) if param_lines else "- <no parameters>"
        tool_blocks.append(
            f"Tool: {name}\nDescription: {description}\nParameters:\n{params_text}"
        )

    tools_text = "\n\n".join(tool_blocks)
    return f"""Task:
{task}

Candidate tools:
{tools_text}

Return the JSON object now."""


def _parse_irrelevant_tools(raw_text: str, valid_tool_names: List[str]) -> List[str]:
    cleaned = sanitize_llm_json_text(raw_text)
    if not cleaned:
        return []

    try:
        payload = json.loads(cleaned)
    except Exception as exc:
        logger.warning("Tool filter JSON parse failed: %s", exc)
        return []

    names = payload.get("irrelevant_tools", []) if isinstance(payload, dict) else []
    if not isinstance(names, list):
        return []

    valid_set = set(valid_tool_names)
    filtered: List[str] = []
    for item in names:
        if isinstance(item, str) and item in valid_set and item not in filtered:
            filtered.append(item)
    return filtered


def filter_definitely_irrelevant_tools(
    *,
    task: str,
    tools: List[Dict[str, Any]],
    model_name: str,
    timeout: int = 60,
) -> List[str]:
    """Return tool names that are definitely irrelevant to the task."""
    if not task or not tools:
        return []

    prefiltered = _prefilter_definitely_irrelevant_tools(task, tools)
    if prefiltered:
        return prefiltered

    prompt = _build_filter_prompt(task, tools)
    agent = ChatAgent(
        model_name=model_name,
        system_message=TOOL_FILTER_SYSTEM_PROMPT,
        timeout=timeout,
        agent_role="tool_filter",
    )
    response = agent.generate_response(prompt)
    if not response.success:
        logger.warning("Tool filter failed: %s", response.error_message)
        return []
    tool_names = [tool.get("name", "") for tool in tools if tool.get("name")]
    return _parse_irrelevant_tools(response.raw_response or "", tool_names)

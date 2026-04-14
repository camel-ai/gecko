import logging
from typing import Any, Dict, List, Optional, Tuple

from inference.core.tool_loader import ToolLoader
from utils.conversation import render_conversation
from utils.conversation_memory import ConversationMemoryStore

logger = logging.getLogger(__name__)


class EvalCoordinator:
    """Coordinates checklist generation and judge evaluation for SimSolver."""

    def __init__(
        self,
        evaluator: Any,
        enable_checklist: bool = True,
        enable_tool_result_folding: bool = True,
        include_agent_response_in_judge: bool = True,
        base_checklist_items: Optional[List[str]] = None,
    ):
        self.evaluator = evaluator
        self.enable_checklist = enable_checklist
        self.enable_tool_result_folding = enable_tool_result_folding
        self.include_agent_response_in_judge = include_agent_response_in_judge
        self.base_checklist_items = base_checklist_items

    @property
    def should_generate(self) -> bool:
        return self.evaluator is not None

    def generate_checklist(
        self,
        user_message: str,
        history_items: List[Dict[str, Any]],
        tool_catalog: Optional[List[Dict[str, str]]],
        agent_system_prompt: Optional[str],
    ) -> List[Dict]:
        """Generate checklist for task evaluation.

        Returns:
            List of checklist item dicts. Also returned on failure (empty list).
        """
        try:
            if self.enable_checklist:
                previous_tasks: List[str] = []
                try:
                    text_history = render_conversation(
                        history_items,
                        text_only=True,
                        include_tool_calls=False,
                        include_results=False,
                        truncate_assistant=None,
                    )
                    for item in text_history:
                        content = item.get("content")
                        if isinstance(content, str) and content.strip():
                            previous_tasks.append(content)
                except Exception as e:
                    logger.warning(f"[CHECKLIST] Failed to build previous_tasks: {e}")
                    previous_tasks = []

                checklist = self.evaluator.generate_checklist(
                    task=user_message,
                    initial_config=None,
                    previous_tasks=previous_tasks,
                    conversation_history=history_items,
                    tool_definitions=tool_catalog or None,
                    policy_text=self.extract_policy_text(agent_system_prompt),
                )
                logger.info(f"Generated checklist with {len(checklist)} items")
                try:
                    for i, item in enumerate(checklist[:5], 1):
                        desc = item.get("description", str(item)) if isinstance(item, dict) else str(item)
                        logger.info(f"[CHECKLIST] Item {i}: {desc}")
                except Exception:
                    pass
            else:
                if isinstance(self.base_checklist_items, list) and self.base_checklist_items:
                    checklist = [
                        {"description": str(item)}
                        for item in self.base_checklist_items
                        if str(item).strip()
                    ]
                    logger.info(
                        "Using fixed checklist from base_checklist_items (generation disabled): %d items",
                        len(checklist),
                    )
                else:
                    checklist = [{"description": f"Verify the operation was executed: {user_message[:100]}"}]
                    logger.info("Using simple checklist (checklist generation disabled)")

            return checklist
        except Exception as e:
            logger.error(f"Failed to generate checklist: {e}")
            return []

    def evaluate_attempt(
        self,
        checklist: List[Dict],
        config: Dict,
        tool_calls: List[Dict],
        agent_response: str,
        history_items: List[Dict[str, Any]],
        tool_definitions: List[Dict[str, Any]],
        memory_store: Optional[ConversationMemoryStore],
        agent_system_prompt: Optional[str],
        attempt: Optional[int] = None,
        user_message: Optional[str] = None,
    ) -> Tuple[float, Dict, Optional[List]]:
        """Evaluate attempt using checklist.

        Returns:
            (score, feedback_dict, updated_judgment_results_or_None)
        """
        try:
            rendered_history = render_conversation(
                history_items,
                include_tool_calls=True,
                include_results=True,
                truncate_assistant=None,
                truncate_result=None,
            )
            if self.enable_tool_result_folding and memory_store is not None:
                judge_tool_calls = self.fold_tool_calls(tool_calls, memory_store)
                judge_history = self.fold_history(rendered_history, memory_store)
                judge_memory_store: Optional[ConversationMemoryStore] = memory_store
            else:
                judge_tool_calls = tool_calls or []
                judge_history = rendered_history
                judge_memory_store = None

            policy_text = self.extract_policy_text(agent_system_prompt)
            involved = ToolLoader.select_involved_definitions(tool_definitions, tool_calls)

            # Pass full definitions (with parameter schemas, descriptions, defaults) in both
            # branches — same as triage judge. build_tool_catalog() was converting parameters
            # to a flat string, causing _normalize_tool_definitions to drop them entirely.
            judge_tool_defs = (involved if involved else tool_definitions) or None

            judge_agent_response = agent_response if self.include_agent_response_in_judge else None
            judgment_results, critical, score = self.evaluator.judge_execution(
                checklist=checklist,
                current_config=config,
                tool_calls=judge_tool_calls,
                tool_definitions=judge_tool_defs,
                agent_response=judge_agent_response,
                conversation_history=judge_history,
                attempt=attempt,
                memory_store=judge_memory_store,
                policy_text=policy_text,
                user_request=user_message,
            )

            feedback = {
                "judgment_results": judgment_results,
                "failed_items": [item for item in judgment_results if item.get("status") == "failed"],
            }
            try:
                logger.debug(
                    f"[JUDGE] Score: {score:.2f}, Failed items: "
                    f"{len(feedback.get('failed_items', []))}"
                )
                if isinstance(judgment_results, list):
                    for i, jr in enumerate(judgment_results[:5], 1):
                        status = jr.get("status", "unknown")
                        desc = jr.get("description", "")
                        logger.debug(f"[JUDGE] Item {i}: status={status}, desc={desc[:120]}")
            except Exception:
                pass

            updated_checklist = judgment_results if isinstance(judgment_results, list) and judgment_results else None
            return score, feedback, updated_checklist

        except Exception as e:
            logger.error(f"Failed to evaluate attempt: {e}")
            failure_item = {
                "name": "judge_evaluation_failed",
                "description": "Judge evaluation failed; this attempt must be retried.",
                "reasoning": f"Judge exception: {str(e)}",
                "status": "failed",
            }
            feedback = {
                "judge_failure": True,
                "error": str(e),
                "judgment_results": [failure_item],
                "failed_items": [failure_item],
            }
            return 0.0, feedback, None

    @staticmethod
    def fold_history(
        history: List[Dict[str, Any]], memory_store: ConversationMemoryStore
    ) -> List[Dict[str, Any]]:
        """Fold large tool results in conversation history for judge input."""
        folded: List[Dict[str, Any]] = []
        for item in history or []:
            if not isinstance(item, dict):
                continue
            new_item = dict(item)
            role = new_item.get("role")
            if role in ("tool_result", "tool_call") and "result" in new_item:
                function_name = str(new_item.get("function") or new_item.get("name") or "unknown")
                new_item["result"] = memory_store.fold_result(
                    function_name=function_name,
                    result=new_item.get("result"),
                )
            folded.append(new_item)
        return folded

    @staticmethod
    def fold_tool_calls(
        tool_calls: List[Dict[str, Any]], memory_store: ConversationMemoryStore
    ) -> List[Dict[str, Any]]:
        """Fold large tool results in tool calls for judge input."""
        folded: List[Dict[str, Any]] = []
        for tc in tool_calls or []:
            if not isinstance(tc, dict):
                continue
            new_tc = dict(tc)
            if "result" in new_tc:
                function_name = str(new_tc.get("function") or "unknown")
                new_tc["result"] = memory_store.fold_result(
                    function_name=function_name,
                    result=new_tc.get("result"),
                )
            folded.append(new_tc)
        return folded

    @staticmethod
    def extract_policy_text(agent_system_prompt: Optional[str]) -> Optional[str]:
        """Extract policy text from agent system prompt, if present."""
        if not agent_system_prompt:
            return None
        if "<policy>" in agent_system_prompt and "</policy>" in agent_system_prompt:
            try:
                return agent_system_prompt.split("<policy>", 1)[1].split("</policy>", 1)[0].strip()
            except Exception:
                return agent_system_prompt.strip()
        return None

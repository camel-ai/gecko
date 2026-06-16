import copy
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from inference.utils.provider_errors import classify_provider_error


logger = logging.getLogger(__name__)

MOCK_TASK_AGENT_TIMEOUT_SECONDS = 3600


class AgentExecutor:
    """Creates and invokes ChatAgent instances, normalizes tool call results."""

    def __init__(
        self,
        model_name: str,
        agent_timeout: Optional[int],
        agent_system_prompt: Optional[str],
        agent_max_iteration: int,
        agent_summarize_threshold: Optional[int],
        tools: List[Any],
    ):
        self.model_name = model_name
        self.agent_timeout = agent_timeout
        self.agent_system_prompt = agent_system_prompt
        self.agent_max_iteration = agent_max_iteration
        self.agent_summarize_threshold = agent_summarize_threshold
        self.tools = tools

    def execute(
        self,
        message: str,
        session_id: str,
        turn_count: int,
        context: Optional[Dict] = None,
        existing_agent: Optional[Any] = None,
        attempt: Optional[int] = None,
    ) -> Tuple[List[Dict], str, int, Any, Dict[str, Any]]:
        """Execute message with agent and return tool calls, response, tools count, agent, metadata."""
        from inference.agents.chat_agent import ChatAgent

        if existing_agent:
            agent = existing_agent
            logger.debug("[AGENT] Using provided agent")
        else:
            effective_timeout = (
                self.agent_timeout
                if self.agent_timeout is not None
                else MOCK_TASK_AGENT_TIMEOUT_SECONDS
            )
            agent = ChatAgent(
                model_name=self.model_name,
                timeout=effective_timeout,
                system_message=self.agent_system_prompt,
                max_iteration=self.agent_max_iteration,
                summarize_threshold=self.agent_summarize_threshold,
                agent_role="simsolver",
            )
        if hasattr(agent, "_camel_agent") and agent._camel_agent is not None:
            agent._camel_agent.max_iteration = self.agent_max_iteration
            agent._camel_agent.summarize_threshold = self.agent_summarize_threshold
        if hasattr(agent, "set_log_context"):
            agent.set_log_context(turn_idx=turn_count, attempt=attempt)

        tools = self.tools or []
        if not tools:
            raise RuntimeError("AgentExecutor has no tools configured.")

        tools_count = len(tools)
        logger.debug(f"[AGENT] Turn {turn_count}: available tools count={tools_count}")
        logger.debug(f"[AGENT] Session ID: {session_id}")

        need_tools = True
        if hasattr(agent, "_camel_agent") and hasattr(agent._camel_agent, "tool_dict"):
            existing_tools_count = len(agent._camel_agent.tool_dict)
            if existing_tools_count > 0:
                logger.debug(f"[AGENT] Agent already has {existing_tools_count} tools")
                need_tools = False

        if need_tools and hasattr(agent, "set_tools") and tools:
            agent.set_tools(tools)
            logger.debug("[AGENT] Tools set on agent successfully")
            if hasattr(agent, "_camel_agent") and hasattr(agent._camel_agent, "tool_dict"):
                len(agent._camel_agent.tool_dict)  # verify
        elif need_tools:
            logger.warning("[AGENT] Failed to set tools on agent")

        try:
            if context is None:
                context = {}
            message_chars = len(message or "")
            context_chars = 0
            try:
                context_chars = len(json.dumps(context, ensure_ascii=False, default=str))
            except Exception:
                context_chars = len(str(context))
            logger.info(
                "Task agent input summary: turn=%s attempt=%s session=%s prompt_chars=%s "
                "context_chars=%s tools=%s max_iteration=%s timeout=%s",
                turn_count,
                attempt if attempt is not None else "?",
                session_id,
                message_chars,
                context_chars,
                tools_count,
                self.agent_max_iteration,
                self.agent_timeout if self.agent_timeout is not None else MOCK_TASK_AGENT_TIMEOUT_SECONDS,
            )
            attempt_label = attempt if attempt is not None else "?"
            provider_retry_limit = self._provider_retry_limit()
            provider_retry_backoff = self._provider_retry_backoff()
            provider_retry_count = 0
            provider_failure_type = None
            provider_error_text = ""

            for provider_attempt in range(provider_retry_limit + 1):
                response = agent.generate_response(message, context)
                provider_retry_count = provider_attempt
                agent_success_probe = bool(getattr(response, "success", True))
                agent_error_probe = getattr(response, "error_message", None)
                raw_probe = getattr(response, "raw_response", None) or ""
                if agent_error_probe or not agent_success_probe:
                    provider_error_text = str(agent_error_probe or raw_probe or "")
                    provider_failure_type = classify_provider_error(provider_error_text)
                else:
                    provider_error_text = ""
                    provider_failure_type = None

                should_retry_provider = (
                    provider_failure_type == "provider_transient_error"
                    and (not agent_success_probe or not getattr(response, "tool_calls", None))
                    and provider_attempt < provider_retry_limit
                )
                if not should_retry_provider:
                    break

                sleep_for = provider_retry_backoff * (2 ** provider_attempt)
                logger.warning(
                    "Retrying provider transient failure before consuming GATS retry "
                    "(turn=%s attempt=%s provider_try=%s/%s sleep=%.1fs): %s",
                    turn_count,
                    attempt_label,
                    provider_attempt + 1,
                    provider_retry_limit + 1,
                    sleep_for,
                    provider_error_text[:300],
                )
                time.sleep(sleep_for)
            agent_success = bool(getattr(response, "success", True))
            agent_error = getattr(response, "error_message", None)
            logger.debug("[AGENT] Response generated")
            if not agent_success:
                logger.warning(
                    "Agent execution reported failure (turn=%s attempt=%s): %s",
                    turn_count, attempt_label, agent_error,
                )

            agent_response = ""
            if hasattr(response, "raw_response"):
                agent_response = response.raw_response if response.raw_response else ""
                logger.debug(f"[AGENT] Response content: {agent_response[:200] if agent_response else 'None'}")

            tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
            logger.debug(f"[AGENT] Tool calls count: {len(tool_calls)}")
            logger.debug(f"[AGENT] Raw tool_calls type: {type(tool_calls)}")
            if tool_calls:
                logger.debug(f"[AGENT] Raw tool_calls content: {json.dumps(tool_calls, indent=2, default=str)}")

            tool_calls = [self.normalize_tool_call(tc) for tc in tool_calls if tc is not None]
            tool_calls = [self._normalize_tool_call_arguments(tc) for tc in tool_calls]

            provider_failure_type = classify_provider_error(
                agent_error or (agent_response if not agent_success else "")
            )
            if provider_failure_type and not tool_calls:
                agent_success = False
                agent_error = agent_error or agent_response or provider_failure_type

            if len(tool_calls) > 1:
                deduped = [tool_calls[0]]
                for tc in tool_calls[1:]:
                    if isinstance(tc, dict) and isinstance(deduped[-1], dict):
                        prev_key = (deduped[-1].get("function", ""), json.dumps(deduped[-1].get("arguments", {}), sort_keys=True, default=str))
                        curr_key = (tc.get("function", ""), json.dumps(tc.get("arguments", {}), sort_keys=True, default=str))
                        if prev_key == curr_key:
                            deduped[-1] = tc  # keep last of consecutive run
                            continue
                    deduped.append(tc)
                tool_calls = deduped

            try:
                for i, tc in enumerate(tool_calls, 1):
                    logger.debug(f"[AGENT] ToolCall {i} type: {type(tc)}")
                    logger.debug(f"[AGENT] ToolCall {i} raw: {tc}")
                    func = tc.get("function", "unknown") if isinstance(tc, dict) else str(tc)
                    args = tc.get("arguments", {}) if isinstance(tc, dict) else {}
                    res = tc.get("result", None) if isinstance(tc, dict) else None
                    logger.debug(f"[AGENT] ToolCall {i}: function={func}, args={args}")
                    logger.debug(f"[AGENT] ToolCall {i} args type: {type(args)}")
                    if res is not None:
                        res_preview = str(res)
                        if len(res_preview) > 300:
                            res_preview = res_preview[:300] + "..."
                        logger.debug(f"[AGENT] ToolCall {i} result: {res_preview}")
            except Exception as e:
                logger.warning(f"[AGENT] Failed to log detailed tool calls: {e}")
                import traceback
                logger.warning(f"[AGENT] Traceback: {traceback.format_exc()}")

            metadata: Dict[str, Any] = {}
            if hasattr(response, "metadata") and isinstance(response.metadata, dict):
                metadata = response.metadata
            metadata["agent_success"] = agent_success
            if agent_error:
                metadata["agent_error"] = str(agent_error)
                if "failure_type" not in metadata:
                    metadata["failure_type"] = (
                        provider_failure_type
                        or ("timeout" if "timed out" in str(agent_error).lower() else "agent_error")
                    )
            if provider_failure_type:
                metadata["provider_failure_type"] = provider_failure_type
            metadata["provider_retry_count"] = provider_retry_count

            return tool_calls, agent_response, tools_count, agent, metadata
        except Exception as e:
            logger.error(f"Failed to execute with agent: {e}")
            failure_type = classify_provider_error(e) or "exception"
            return [], str(e), tools_count, agent, {
                "agent_success": False,
                "agent_error": str(e),
                "failure_type": failure_type,
            }

    @staticmethod
    def _provider_retry_limit() -> int:
        raw = os.getenv("GECKO_PROVIDER_RETRIES", "2")
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 2

    @staticmethod
    def _provider_retry_backoff() -> float:
        raw = os.getenv("GECKO_PROVIDER_RETRY_BACKOFF", "2.0")
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return 2.0

    @staticmethod
    def normalize_tool_call(tc: Any) -> Dict[str, Any]:
        """Normalize tool call structure to ensure function name and arguments are present."""
        if not isinstance(tc, dict):
            possible: Dict[str, Any] = {}
            for key in ["function", "name", "tool", "tool_name"]:
                val = getattr(tc, key, None)
                if val:
                    possible["function"] = val if not isinstance(val, dict) else val.get("name") or val.get("function")
                    break
            args = getattr(tc, "arguments", None) or getattr(tc, "args", None)
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            if args is None:
                args = {}
            if not possible:
                logger.warning(f"Unrecognized tool_call structure (non-dict): {tc}")
                return {"function": "unknown", "arguments": args, "raw": tc}
            return {"function": possible.get("function", "unknown"), "arguments": args}

        normalized = dict(tc)

        func_field = normalized.get("function")
        if isinstance(func_field, dict):
            normalized["function"] = func_field.get("name", func_field.get("function", "unknown"))
            if "arguments" in func_field and not normalized.get("arguments"):
                normalized["arguments"] = func_field.get("arguments")
        elif not func_field and normalized.get("name"):
            normalized["function"] = normalized.get("name")

        args = normalized.get("arguments")
        if isinstance(args, str):
            try:
                normalized["arguments"] = json.loads(args)
            except Exception:
                pass
        elif args is None:
            normalized["arguments"] = {}

        if not normalized.get("function"):
            logger.warning(f"Missing function name in tool_call dict: {normalized}")
            normalized["function"] = "unknown"

        return normalized

    def _normalize_tool_call_arguments(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize recorded tool-call arguments to match schema semantics.

        Strips empty optional parameters (empty lists, empty strings, None)
        to avoid evaluation mismatches when agents include defaults, while
        preserving empty values for required parameters.
        """
        if not isinstance(tool_call, dict):
            return tool_call

        function_name = str(tool_call.get("function") or "").strip()
        arguments = tool_call.get("arguments")
        if not function_name or not isinstance(arguments, dict):
            return tool_call

        normalized = dict(tool_call)
        normalized["original_arguments"] = copy.deepcopy(arguments)
        required_params = set(self._find_tool_required_params(function_name))
        normalized["arguments"] = {
            k: v for k, v in arguments.items()
            if k in required_params or (v != [] and v != "" and v is not None)
        }
        return normalized

    def _find_tool_parameter_schema(self, function_name: str) -> Optional[Dict[str, Any]]:
        for tool in self.tools or []:
            schema = getattr(tool, "openai_tool_schema", None)
            if not isinstance(schema, dict):
                continue
            func_schema = schema.get("function", schema)
            if not isinstance(func_schema, dict):
                continue
            if str(func_schema.get("name") or "").strip() != function_name:
                continue
            params = func_schema.get("parameters", {})
            if not isinstance(params, dict):
                return None
            properties = params.get("properties", {})
            return properties if isinstance(properties, dict) else None
        return None

    def _find_tool_required_params(self, function_name: str) -> List[str]:
        """Return the 'required' list from the tool's parameter schema."""
        for tool in self.tools or []:
            schema = getattr(tool, "openai_tool_schema", None)
            if not isinstance(schema, dict):
                continue
            func_schema = schema.get("function", schema)
            if not isinstance(func_schema, dict):
                continue
            if str(func_schema.get("name") or "").strip() != function_name:
                continue
            params = func_schema.get("parameters", {})
            if not isinstance(params, dict):
                return []
            required = params.get("required", [])
            return required if isinstance(required, list) else []
        return []

    @staticmethod
    def clone_with_memory(base_agent: Any) -> Optional[Any]:
        """Clone an agent with its conversation memory using CAMEL's clone method."""
        try:
            if hasattr(base_agent, "_camel_agent"):
                cloned_camel = base_agent._camel_agent.clone(with_memory=True)
                from inference.agents.chat_agent import ChatAgent
                cloned_wrapper = ChatAgent(
                    model_name=base_agent.model_name,
                    system_message=getattr(base_agent, "system_message", None),
                    timeout=getattr(base_agent, "timeout", None),
                    agent_role=getattr(base_agent, "agent_role", "simsolver"),
                )
                cloned_wrapper._camel_agent = cloned_camel
                logger.info("Successfully cloned agent with memory")
                return cloned_wrapper
            else:
                cloned_agent = base_agent.clone(with_memory=True)
                logger.info("Successfully cloned CAMEL agent directly")
                return cloned_agent
        except Exception as e:
            logger.error(f"Failed to clone agent: {e}")
            return None

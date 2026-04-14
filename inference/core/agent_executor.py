import copy
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from inference.core.debug_tracer import DebugTracer


logger = logging.getLogger(__name__)


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
        tracer: DebugTracer,
    ):
        self.model_name = model_name
        self.agent_timeout = agent_timeout
        self.agent_system_prompt = agent_system_prompt
        self.agent_max_iteration = agent_max_iteration
        self.agent_summarize_threshold = agent_summarize_threshold
        self.tools = tools
        self.tracer = tracer

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
            logger.debug("[AGENT DEBUG] Using provided agent")
        else:
            timeout = self.agent_timeout if self.agent_timeout is not None else 60
            agent = ChatAgent(
                model_name=self.model_name,
                timeout=timeout,
                system_message=self.agent_system_prompt,
                max_iteration=self.agent_max_iteration,
                summarize_threshold=self.agent_summarize_threshold,
                agent_role="simsolver",
            )
        if hasattr(agent, "_camel_agent") and agent._camel_agent is not None:
            agent._camel_agent.max_iteration = self.agent_max_iteration
            agent._camel_agent.summarize_threshold = self.agent_summarize_threshold

        # Debug: write system prompt
        try:
            system_prompt = getattr(agent, "system_message", None) or ""
            sys_len = len(system_prompt)
            sys_preview = system_prompt[:1200] if system_prompt else "<EMPTY>"
            os.makedirs("debug_traces", exist_ok=True)
            sys_path = os.path.join(
                "debug_traces",
                f"{self.tracer.task_id}_simsolver_system_prompt_t{turn_count}_a{attempt if attempt is not None else 'na'}.txt",
            )
            with open(sys_path, "w", encoding="utf-8") as f:
                f.write(system_prompt)
            self.tracer.trace_print(f"[SYSTEM PROMPT] len={sys_len} file={sys_path}")
            self.tracer.trace_print(f"[SYSTEM PROMPT PREVIEW]\n{sys_preview}\n[END SYSTEM PROMPT PREVIEW]")
        except Exception as e:
            self.tracer.trace_print(f"[SYSTEM PROMPT] Failed to write preview: {e}")

        tools = self.tools or []
        if not tools:
            raise RuntimeError("AgentExecutor has no tools configured.")

        tools_count = len(tools)
        logger.debug(f"[AGENT DEBUG] Turn {turn_count}: Available tools count: {tools_count}")
        logger.debug(f"[AGENT DEBUG] Session ID: {session_id}")

        # Print tool names for debugging
        if tools:
            tool_names = []
            for tool in tools:
                if hasattr(tool, "__name__"):
                    tool_names.append(tool.__name__)
                elif hasattr(tool, "get_function_name"):
                    tool_names.append(tool.get_function_name())
                elif isinstance(tool, dict) and "name" in tool:
                    tool_names.append(tool["name"])
            if tool_names:
                preview_names = tool_names[:20]
                self.tracer.trace_print(f"[TOOLS] count={tools_count} sample={preview_names}")
        else:
            logger.warning("[AGENT DEBUG] No tools available!")

        # Only set tools if agent doesn't already have them (e.g., not cloned)
        need_tools = True
        if hasattr(agent, "_camel_agent") and hasattr(agent._camel_agent, "tool_dict"):
            existing_tools_count = len(agent._camel_agent.tool_dict)
            if existing_tools_count > 0:
                logger.debug(f"[AGENT DEBUG] Agent already has {existing_tools_count} tools (likely from cloning)")
                need_tools = False

        if need_tools and hasattr(agent, "set_tools") and tools:
            agent.set_tools(tools)
            logger.debug("[AGENT DEBUG] Tools set on agent successfully")
            if hasattr(agent, "_camel_agent") and hasattr(agent._camel_agent, "tool_dict"):
                len(agent._camel_agent.tool_dict)  # verify
        elif need_tools:
            logger.warning("[AGENT DEBUG] Failed to set tools on agent")

        # Generate response
        try:
            if context is None:
                context = {}
            _at0 = datetime.now()
            attempt_label = attempt if attempt is not None else "?"
            self.tracer.trace_print(
                f"[TIME] SimSolver Agent LLM START {_at0.strftime('%H:%M:%S')} "
                f"(model={self.model_name}, turn={turn_count}, attempt={attempt_label})"
            )
            response = agent.generate_response(message, context)
            _at1 = datetime.now()
            usage = self.extract_usage(response)
            agent_success = bool(getattr(response, "success", True))
            agent_error = getattr(response, "error_message", None)
            self.tracer.trace_print(
                "[TIME] SimSolver Agent LLM END   "
                f"{_at1.strftime('%H:%M:%S')} (elapsed={(_at1 - _at0).total_seconds():.3f}s, "
                f"model={self.model_name}, turn={turn_count}, attempt={attempt_label}, "
                f"tokens_in={usage['input_tokens']}, tokens_out={usage['output_tokens']}, "
                f"tokens_total={usage['total_tokens']})"
            )
            logger.debug("[AGENT DEBUG] Response generated")
            if not agent_success:
                logger.warning(
                    "Agent execution reported failure (turn=%s attempt=%s): %s",
                    turn_count, attempt_label, agent_error,
                )

            # Extract response content
            agent_response = ""
            if hasattr(response, "raw_response"):
                agent_response = response.raw_response if response.raw_response else ""
                logger.debug(f"[AGENT DEBUG] Response content: {agent_response[:200] if agent_response else 'None'}")
                resp_display = agent_response if agent_response else "<EMPTY>"
                self.tracer.trace_print(f"[SIMSOLVER AGENT RESPONSE] {resp_display}")

            # Extract and normalize tool calls
            tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
            logger.debug(f"[AGENT DEBUG] Tool calls count: {len(tool_calls)}")
            logger.debug(f"[AGENT DEBUG] Raw tool_calls type: {type(tool_calls)}")
            if tool_calls:
                logger.debug(f"[AGENT DEBUG] Raw tool_calls content: {json.dumps(tool_calls, indent=2, default=str)}")

            tool_calls = [self.normalize_tool_call(tc) for tc in tool_calls if tc is not None]
            tool_calls = [self._normalize_tool_call_arguments(tc) for tc in tool_calls]

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
                    logger.debug(f"[AGENT DEBUG] ToolCall {i} type: {type(tc)}")
                    logger.debug(f"[AGENT DEBUG] ToolCall {i} raw: {tc}")
                    func = tc.get("function", "unknown") if isinstance(tc, dict) else str(tc)
                    args = tc.get("arguments", {}) if isinstance(tc, dict) else {}
                    res = tc.get("result", None) if isinstance(tc, dict) else None
                    logger.debug(f"[AGENT DEBUG] ToolCall {i}: function={func}, args={args}")
                    logger.debug(f"[AGENT DEBUG] ToolCall {i} args type: {type(args)}")
                    if res is not None:
                        res_preview = str(res)
                        if len(res_preview) > 300:
                            res_preview = res_preview[:300] + "..."
                        logger.debug(f"[AGENT DEBUG] ToolCall {i} result: {res_preview}")
            except Exception as e:
                logger.warning(f"[AGENT DEBUG] Failed to log detailed tool calls: {e}")
                import traceback
                logger.warning(f"[AGENT DEBUG] Traceback: {traceback.format_exc()}")

            metadata: Dict[str, Any] = {}
            if hasattr(response, "metadata") and isinstance(response.metadata, dict):
                metadata = response.metadata
            metadata["agent_success"] = agent_success
            if agent_error:
                metadata["agent_error"] = str(agent_error)
                if "failure_type" not in metadata:
                    metadata["failure_type"] = (
                        "timeout" if "timed out" in str(agent_error).lower() else "agent_error"
                    )

            return tool_calls, agent_response, tools_count, agent, metadata
        except Exception as e:
            logger.error(f"Failed to execute with agent: {e}")
            return [], str(e), tools_count, agent, {
                "agent_success": False,
                "agent_error": str(e),
                "failure_type": "exception",
            }

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

        # Flatten OpenAI-style function object if present
        func_field = normalized.get("function")
        if isinstance(func_field, dict):
            normalized["function"] = func_field.get("name", func_field.get("function", "unknown"))
            if "arguments" in func_field and not normalized.get("arguments"):
                normalized["arguments"] = func_field.get("arguments")
        elif not func_field and normalized.get("name"):
            normalized["function"] = normalized.get("name")

        # Ensure arguments is a dict (parse JSON if it's a string)
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

        # Strip empty optional parameters (empty lists, empty strings)
        # to avoid evaluation mismatches when agent includes defaults
        args = normalized.get("arguments")
        if isinstance(args, dict):
            normalized["arguments"] = {
                k: v for k, v in args.items()
                if v != [] and v != "" and v is not None
            }

        return normalized

    def _normalize_tool_call_arguments(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize recorded tool-call arguments to match schema semantics."""
        if not isinstance(tool_call, dict):
            return tool_call

        function_name = str(tool_call.get("function") or "").strip()
        arguments = tool_call.get("arguments")
        if not function_name or not isinstance(arguments, dict):
            return tool_call

        params = self._find_tool_parameter_schema(function_name)
        if not params:
            return tool_call

        normalized = dict(tool_call)
        normalized["original_arguments"] = copy.deepcopy(arguments)
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
                    timeout=getattr(base_agent, "timeout", 60),
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

    @staticmethod
    def extract_usage(response: Any) -> Dict[str, int]:
        """Extract token usage from agent response, best-effort."""
        usage_info: Dict[str, Any] = {}
        if hasattr(response, "metadata") and isinstance(response.metadata, dict):
            if any(k in response.metadata for k in ("input_tokens", "output_tokens", "total_tokens")):
                usage_info = dict(response.metadata)
            else:
                usage_info = response.metadata.get("usage", {}) or {}
        if not usage_info and hasattr(response, "info") and isinstance(response.info, dict):
            usage_info = response.info.get("usage", {}) or {}

        input_tokens = usage_info.get("prompt_tokens") or usage_info.get("input_tokens") or 0
        output_tokens = usage_info.get("completion_tokens") or usage_info.get("output_tokens") or 0
        total_tokens = usage_info.get("total_tokens")
        if total_tokens is None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)

        return {
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "total_tokens": int(total_tokens or 0),
        }

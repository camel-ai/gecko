
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional
import json_repair

from .base_agent import BaseAgent, AgentResponse
from utils.model_utils import sanitize_llm_json_text

logger = logging.getLogger(__name__)


class ChatAgent(BaseAgent):
    
    def __init__(self, model_name: str = "gpt-4.1-mini", **kwargs):
        self._camel_agent = None
        self._model_backend = None
        
        # Extract tools from kwargs if provided
        self._tools = kwargs.pop('tools', None)
        self._max_iteration = kwargs.pop('max_iteration', 10)
        self._summarize_threshold = kwargs.pop('summarize_threshold', None)
        self.agent_role = kwargs.pop('agent_role', 'chat_agent')
        
        super().__init__(model_name=model_name, **kwargs)
    
    def _init_model(self):
        try:
            from camel.agents import ChatAgent as CAMELChatAgent
            from camel.messages import BaseMessage
            from camel.types import RoleType
            
            self._model_backend = self._create_model_backend()
            
            system_message = BaseMessage.make_assistant_message(
                role_name="Assistant",
                content=self.system_message
            )
            
            token_limit = self.config.get("token_limit")
            if token_limit is None:
                token_limit = 8192 if self.model_name == "ali-qwen3-14b" else 32768

            self._camel_agent = CAMELChatAgent(
                system_message=system_message,
                model=self._model_backend,
                message_window_size=50,
                token_limit=token_limit,
                tools=getattr(self, '_tools', None),
                max_iteration=self._max_iteration,
                summarize_threshold=self._summarize_threshold,
                # Use CAMEL step timeout directly; avoid external timeout threads.
                step_timeout=self.timeout,
                tool_execution_timeout=self.timeout,
            )
            
            logger.info(
                "Initialized ChatAgent(role=%s) with model: %s",
                self.agent_role,
                self.model_name,
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize ChatAgent: {e}")
            self._camel_agent = None
            self._model_backend = None
    
    def _create_model_backend(self):
        try:
            from utils.model_utils import create_model
            tools = getattr(self, '_tools', None)
            return create_model(self.model_name, tools=tools, timeout=self.timeout)
        except Exception as e:
            logger.error(f"Failed to create model backend: {e}")
            return None
    
    def set_tools(self, tools):
        self._tools = tools
        
        # If we have an existing CAMEL agent, update its tools directly
        if hasattr(self, '_camel_agent') and self._camel_agent is not None:
            try:
                # Clear existing tools if any
                if hasattr(self._camel_agent, 'tool_dict') and self._camel_agent.tool_dict:
                    existing_tool_names = list(self._camel_agent.tool_dict.keys())
                    if existing_tool_names and hasattr(self._camel_agent, 'remove_tools'):
                        self._camel_agent.remove_tools(existing_tool_names)
                        logger.debug(f"Cleared {len(existing_tool_names)} existing tools")
                
                # Add new tools
                if tools and hasattr(self._camel_agent, 'add_tools'):
                    self._camel_agent.add_tools(tools)
                    logger.debug(f"Added {len(tools)} tools to CAMEL agent")
                elif tools and hasattr(self._camel_agent, 'add_tool'):
                    # Fallback to adding tools one by one
                    for tool in tools:
                        self._camel_agent.add_tool(tool)
                    logger.debug(f"Added {len(tools)} tools individually")
                    
            except Exception as e:
                logger.warning(f"Could not update tools on existing CAMEL agent: {e} - reinitializing")
                # Only reinitialize if we can't update tools directly
                self._init_model()
        else:
            # No existing agent, initialize a new one
            self._init_model()
    
    def generate_response(self, prompt: str, 
                         context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        start_time = time.time()
        
        try:
            if not self._camel_agent:
                return AgentResponse(
                    success=False,
                    tool_calls=[],
                    error_message="Agent not initialized"
                )
            
            from camel.messages import BaseMessage
            from camel.types import RoleType
            
            user_message = BaseMessage.make_user_message(
                role_name="User",
                content=prompt
            )
            
            response = self._step_with_timeout(user_message)
            
            llm_cost = 0.0
            input_tokens = 0
            output_tokens = 0
            try:
                from utils.model_utils import calculate_cost
                llm_cost = calculate_cost(response, self.model_name)
                
                if hasattr(response, 'info') and response.info:
                    usage_info = response.info.get('usage', {})
                    input_tokens = usage_info.get('prompt_tokens', 0)
                    output_tokens = usage_info.get('completion_tokens', 0)
                    total_tokens = usage_info.get('total_tokens')
                    if total_tokens is None:
                        total_tokens = (input_tokens or 0) + (output_tokens or 0)
                    logger.debug(
                        f"LLM usage - Input: {input_tokens}, Output: {output_tokens}, "
                        f"Total: {total_tokens}, Cost: ${llm_cost:.4f}"
                    )
                else:
                    total_tokens = input_tokens + output_tokens
            except Exception as e:
                logger.warning(f"Failed to calculate LLM cost: {e}")
                total_tokens = (input_tokens or 0) + (output_tokens or 0)
            
            response_text, tool_calls = self._handle_multi_message_response(response)
            
            logger.info(f"Final extracted tool calls: {tool_calls}")
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                success=True,
                tool_calls=tool_calls,
                raw_response=response_text,
                execution_time=execution_time,
                metadata={
                    "model": self.model_name,
                    "prompt_length": len(prompt),
                    "response_length": len(response_text),
                    "llm_cost": llm_cost,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            logger.error(
                "Agent(role=%s) response generation failed: %s",
                self.agent_role,
                error_msg,
            )
            
            return AgentResponse(
                success=False,
                tool_calls=[],
                error_message=error_msg,
                execution_time=execution_time
            )

    def _step_with_timeout(self, user_message: Any) -> Any:
        """Run CAMEL step in-process to avoid timeout zombie workers."""
        return self._camel_agent.step(user_message)
    
    def _serialize_tool_result(self, result: Any) -> Any:
        """Serialize tool execution result for JSON compatibility.

        Handles Pydantic models, dataclasses, and other complex types.

        Args:
            result: Tool execution result

        Returns:
            JSON-serializable result
        """
        # Handle None
        if result is None:
            return None

        # Primitive types
        if isinstance(result, (int, str, float, bool)):
            return result

        # Pydantic v2 models
        if hasattr(result, "model_dump"):
            try:
                return result.model_dump()
            except Exception as e:
                logger.warning(f"model_dump() failed: {e}")

        # Pydantic v1 models
        if hasattr(result, "dict") and callable(getattr(result, "dict")):
            try:
                return result.dict()
            except Exception as e:
                logger.warning(f"dict() method failed: {e}")

        # Dataclass
        if hasattr(result, "__dataclass_fields__"):
            try:
                from dataclasses import asdict
                return asdict(result)
            except Exception as e:
                logger.warning(f"asdict() failed: {e}")

        # List - recursively serialize
        if isinstance(result, list):
            return [self._serialize_tool_result(item) for item in result]

        # Tuple - convert to list
        if isinstance(result, tuple):
            return [self._serialize_tool_result(item) for item in result]

        # Dict - recursively serialize values
        if isinstance(result, dict):
            return {k: self._serialize_tool_result(v) for k, v in result.items()}

        # Fallback: convert to string
        logger.warning(f"Unknown result type {type(result).__name__}, converting to string")
        return str(result)

    def _extract_camel_tool_calls(self, camel_response) -> List[Dict[str, Any]]:
        tool_calls = []

        try:
            # DEBUG: Log what we're extracting from
            logger.info(f"[EXTRACT DEBUG] camel_response type: {type(camel_response)}")
            logger.info(f"[EXTRACT DEBUG] Has info: {hasattr(camel_response, 'info')}")

            if hasattr(camel_response, 'info') and camel_response.info:
                info = camel_response.info
                logger.info(f"[EXTRACT DEBUG] info keys: {list(info.keys()) if isinstance(info, dict) else 'not a dict'}")

                if 'tool_calls' in info and info['tool_calls']:
                    logger.info(f"[EXTRACT DEBUG] Found {len(info['tool_calls'])} tool calls in info")
                    for idx, tool_call in enumerate(info['tool_calls']):
                        logger.info(f"[EXTRACT DEBUG] tool_call {idx} type: {type(tool_call)}")

                        if hasattr(tool_call, 'tool_name') and hasattr(tool_call, 'args'):
                            logger.info(f"[EXTRACT DEBUG] tool_call {idx}: name={tool_call.tool_name}, args={tool_call.args}")
                            tool_call_data = {
                                "function": tool_call.tool_name,
                                "arguments": tool_call.args
                            }
                            if hasattr(tool_call, 'result') and tool_call.result is not None:
                                tool_call_data["result"] = self._serialize_tool_result(tool_call.result)
                                logger.info(f"[EXTRACT DEBUG] tool_call {idx} has result")

                            tool_calls.append(tool_call_data)
                        elif 'function' in tool_call:
                            func_info = tool_call['function']
                            logger.info(f"[EXTRACT DEBUG] tool_call {idx} has function key: {func_info}")
                            tool_calls.append({
                                "function": func_info.get('name', ''),
                                "arguments": json.loads(func_info.get('arguments', '{}'))
                            })
                
                elif 'external_tool_call_requests' in info and info['external_tool_call_requests']:
                    for tool_call in info['external_tool_call_requests']:
                        if 'function' in tool_call:
                            func_info = tool_call['function']
                            tool_calls.append({
                                "function": func_info.get('name', ''),
                                "arguments": json.loads(func_info.get('arguments', '{}'))
                            })
            
            if hasattr(camel_response, 'msgs') and camel_response.msgs:
                for msg in camel_response.msgs:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            if hasattr(tool_call, 'function'):
                                func_info = tool_call.function
                                tool_calls.append({
                                    "function": func_info.name if hasattr(func_info, 'name') else '',
                                    "arguments": json.loads(func_info.arguments if hasattr(func_info, 'arguments') else '{}')
                                })
        
        except Exception as e:
            logger.warning(f"Failed to extract CAMEL tool calls: {e}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
        
        return tool_calls
    
    def _handle_multi_message_response(self, response):
        response_texts = []
        all_tool_calls = []

        try:
            all_tool_calls = self._extract_camel_tool_calls(response)

            # --- Extract text and record messages ---
            if hasattr(response, 'msgs') and response.msgs:
                if len(response.msgs) > 1:
                    logger.info(f"Handling multiple messages: {len(response.msgs)} messages returned")

                    for i, msg in enumerate(response.msgs):
                        if hasattr(msg, 'content') and msg.content and msg.content.strip():
                            response_texts.append(msg.content.strip())
                            logger.debug(f"Message {i+1} content: {msg.content[:100]}...")

                        if ((hasattr(msg, 'tool_calls') and msg.tool_calls) or
                            (hasattr(msg, 'content') and msg.content and msg.content.strip())):
                            try:
                                self._camel_agent.record_message(msg)
                                logger.debug(f"Recorded message {i+1} to chat history")
                            except Exception as e:
                                logger.warning(f"Failed to record message {i+1}: {e}")

                else:
                    msg = response.msgs[0]
                    if hasattr(msg, 'content') and msg.content:
                        response_texts.append(msg.content)

            # --- Fallback: msg-based extraction (no results) if info path failed ---
            if not all_tool_calls and hasattr(response, 'msgs') and response.msgs:
                for i, msg in enumerate(response.msgs):
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            try:
                                if hasattr(tool_call, 'function'):
                                    func_info = tool_call.function
                                    all_tool_calls.append({
                                        "function": func_info.name if hasattr(func_info, 'name') else '',
                                        "arguments": json.loads(func_info.arguments if hasattr(func_info, 'arguments') else '{}')
                                    })
                            except Exception as e:
                                logger.warning(f"Failed to extract tool call from message {i+1}: {e}")

            response_text = " ".join(response_texts) if response_texts else ""

            if not response_text:
                if hasattr(response, 'content'):
                    response_text = response.content
                elif hasattr(response, 'msg') and hasattr(response.msg, 'content'):
                    response_text = response.msg.content
                else:
                    response_text = str(response)

            if not all_tool_calls and response_text:
                all_tool_calls = self.parse_tool_calls(response_text)

            logger.debug(f"Multi-message handling result: {len(all_tool_calls)} tool calls, text length: {len(response_text)}")

        except Exception as e:
            logger.error(f"Error in _handle_multi_message_response: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

            response_text = response.content if hasattr(response, 'content') else str(response)
            if hasattr(response, 'msg') and hasattr(response.msg, 'content'):
                response_text = response.msg.content
            all_tool_calls = self._extract_camel_tool_calls(response)
            if not all_tool_calls:
                all_tool_calls = self.parse_tool_calls(response_text)

        return response_text, all_tool_calls
    
    def _parse_tool_calls_impl(self, response_text: str) -> List[Dict[str, Any]]:
        tool_calls = []
        
        try:
            response_text = sanitize_llm_json_text(response_text)
            tool_calls = self._parse_json_tool_calls(response_text)
            if tool_calls:
                return tool_calls
            
            tool_calls = self._parse_function_call_format(response_text)
            if tool_calls:
                return tool_calls
            
            tool_calls = self._parse_markdown_code_blocks(response_text)
            if tool_calls:
                return tool_calls
            
            tool_calls = self._parse_with_regex(response_text)
            
        except Exception as e:
            logger.warning(f"Tool call parsing failed: {e}")
        
        return tool_calls
    
    def _parse_json_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        tool_calls = []
        
        json_pattern = r'\\[.*?\\]'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "function" in item:
                            tool_calls.append(item)
                elif isinstance(data, dict) and "function" in data:
                    tool_calls.append(data)
            except json.JSONDecodeError:
                try:
                    data = json_repair.loads(match)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "function" in item:
                                tool_calls.append(item)
                    elif isinstance(data, dict) and "function" in data:
                        tool_calls.append(data)
                except:
                    continue
        
        return tool_calls
    
    def _parse_function_call_format(self, text: str) -> List[Dict[str, Any]]:
        tool_calls = []
        
        pattern = r'(\\w+)\\s*\\((.*?)\\)'
        matches = re.findall(pattern, text)
        
        for func_name, args_str in matches:
            try:
                arguments = {}
                if args_str.strip():
                    arg_pairs = args_str.split(',')
                    for pair in arg_pairs:
                        if '=' in pair:
                            key, value = pair.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"\'')
                            
                            try:
                                if value.lower() in ('true', 'false'):
                                    value = value.lower() == 'true'
                                elif value.isdigit():
                                    value = int(value)
                                elif '.' in value and value.replace('.', '').isdigit():
                                    value = float(value)
                            except:
                                pass
                            
                            arguments[key] = value
                
                tool_calls.append({
                    "function": func_name,
                    "arguments": arguments
                })
                
            except Exception as e:
                logger.warning(f"Failed to parse function call {func_name}: {e}")
        
        return tool_calls
    
    def _parse_markdown_code_blocks(self, text: str) -> List[Dict[str, Any]]:
        tool_calls = []
        
        code_block_pattern = r'```(?:json|python)?\\n?(.*?)\\n?```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match.strip())
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "function" in item:
                            tool_calls.append(item)
                elif isinstance(data, dict) and "function" in data:
                    tool_calls.append(data)
            except json.JSONDecodeError:
                try:
                    tool_calls.extend(self._parse_function_call_format(match))
                except:
                    continue
        
        return tool_calls
    
    def _parse_with_regex(self, text: str) -> List[Dict[str, Any]]:
        tool_calls = []
        
        patterns = [
            r'function:\\s*(\\w+).*?arguments:\\s*({.*?})',
            r'"function":\\s*"(\\w+)".*?"arguments":\\s*({.*?})',
            r'call\\s+(\\w+)\\s*\\((.*?)\\)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    if len(match) == 2:
                        func_name, args_str = match
                        
                        arguments = {}
                        if args_str.strip().startswith('{'):
                            arguments = json.loads(args_str)
                        else:
                            arguments = self._parse_arguments_string(args_str)
                        
                        tool_calls.append({
                            "function": func_name,
                            "arguments": arguments
                        })
                        
                except Exception as e:
                    logger.warning(f"Regex parsing failed for match {match}: {e}")
        
        return tool_calls
    
    def _parse_arguments_string(self, args_str: str) -> Dict[str, Any]:
        arguments = {}
        
        try:
            args_str = args_str.strip('()')
            
            pairs = []
            current = ""
            paren_count = 0
            quote_char = None
            
            for char in args_str:
                if quote_char:
                    current += char
                    if char == quote_char and (not current or current[-2] != '\\\\'):
                        quote_char = None
                elif char in ('"', "'"):
                    quote_char = char
                    current += char
                elif char == '(':
                    paren_count += 1
                    current += char
                elif char == ')':
                    paren_count -= 1
                    current += char
                elif char == ',' and paren_count == 0:
                    pairs.append(current.strip())
                    current = ""
                else:
                    current += char
            
            if current.strip():
                pairs.append(current.strip())
            
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    
                    try:
                        if value.lower() in ('true', 'false'):
                            value = value.lower() == 'true'
                        elif value.lower() == 'none':
                            value = None
                        elif value.isdigit():
                            value = int(value)
                        elif '.' in value and value.replace('.', '').replace('-', '').isdigit():
                            value = float(value)
                        elif value.startswith('[') and value.endswith(']'):
                            value = json.loads(value)
                        elif value.startswith('{') and value.endswith('}'):
                            value = json.loads(value)
                    except:
                        pass
                    
                    arguments[key] = value
        
        except Exception as e:
            logger.warning(f"Failed to parse arguments string: {e}")
        
        return arguments
    
    def reset(self):
        if self._camel_agent:
            self._camel_agent.reset()

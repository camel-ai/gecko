"""
Reusable components for inference engines
Extracted from UniversalInferenceEngine for shared use
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from inference.client_engine import MockServerClient
from inference.agents.chat_agent import ChatAgent
from inference.evaluation import TaskEvaluationResult
from utils.test_case_adapter import TestCaseAdapter

logger = logging.getLogger(__name__)


def create_session_with_state(state: Dict, mock_client: MockServerClient) -> str:
    """
    Create a new session with initial state/config
    
    Args:
        state: Initial state/config dictionary
        mock_client: Mock server client instance
        
    Returns:
        Session ID
    """
    try:
        # Create a minimal test case for session creation
        # The test_case is required by the API but not used
        from benchmarks.base.test_case import TestCase
        dummy_test_case = TestCase(
            id="example_gen",
            metadata={},
            content={},
            expected_outputs=None,
            evaluation_config={}
        )
        
        session_id = mock_client.create_session(dummy_test_case)
        
        if state:
            # Set initial config/state in the session
            mock_client.set_session_state(session_id, state, bootstrap_mode="auto")
            logger.debug(f"Created session {session_id} with initial state")
        else:
            logger.debug(f"Created session {session_id} with empty state")
            
        return session_id
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise


def create_agent_with_tools(tools: List[Dict], 
                           session_id: str,
                           model_name: str,
                           mock_server_url: str,
                           temperature: float = 0.1,
                           retry_context: str = None,
                           system_prompt: str = None) -> Any:
    """
    Create an agent with tools bound to a session.
    Supports both OpenAPI schemas and BFCL function definitions.
    
    Args:
        tools: Tool/function definitions (OpenAPI schemas or BFCL format)
        session_id: Mock server session ID
        model_name: Model to use for agent
        mock_server_url: Mock server URL
        temperature: Model temperature
        retry_context: Optional retry context for failed attempts
        system_prompt: Optional custom system prompt to override default
        
    Returns:
        Agent instance
    """
    try:
        from camel.toolkits import FunctionTool
        import requests
        import json
        
        function_tools = []
        
        # Helper function to create mock function
        def make_mock_func(original_name, description, domain=None):
            def mock_func(**kwargs):
                """Mock function that sends to mock server"""
                # Determine the domain from the schema if available
                if domain:
                    # tau2 style with domain prefix
                    endpoint = f"{mock_server_url}/{domain}{original_name}"
                    response = requests.post(
                        endpoint,
                        json=kwargs,
                        headers={"X-Session-ID": session_id}
                    )
                else:
                    # BFCL style with execute endpoint
                    response = requests.post(
                        f"{mock_server_url}/execute",
                        json={
                            "function_name": original_name.strip('/'),
                            "arguments": kwargs
                        },
                        headers={"X-Session-ID": session_id}
                    )
                if response.status_code == 200:
                    return response.json().get("result", "Success")
                else:
                    return f"Error: {response.text}"
            
            mock_func.__doc__ = description
            return mock_func
        
        # Process each tool
        for tool in tools:
            if isinstance(tool, dict):
                # Check if this is an OpenAPI schema
                if 'openapi' in tool and 'paths' in tool:
                    # Try to detect domain from schema info
                    domain = None
                    if 'info' in tool and 'title' in tool['info']:
                        title = tool['info']['title'].lower()
                        if 'airline' in title:
                            domain = 'airline'
                        elif 'retail' in title:
                            domain = 'retail'
                        elif 'telecom' in title:
                            domain = 'telecom'
                    
                    # Extract functions from OpenAPI schema
                    for path, methods in tool.get('paths', {}).items():
                        for method, spec in methods.items():
                            # Extract function name from path
                            func_name = path.strip('/').replace('/', '_')
                            
                            # Get description
                            description = spec.get('description', spec.get('summary', ''))
                            
                            # Get parameters from requestBody
                            params_schema = {}
                            if 'requestBody' in spec:
                                content = spec['requestBody'].get('content', {})
                                if 'application/json' in content:
                                    params_schema = content['application/json'].get('schema', {})
                            
                            # Clean tool name for OpenAI API
                            clean_tool_name = func_name.replace('.', '_')
                            
                            # Create mock function with domain if tau2
                            mock_func = make_mock_func(path, description, domain=domain)
                            mock_func.__name__ = clean_tool_name
                            
                            # Create OpenAI schema
                            openai_schema = {
                                'type': 'function',
                                'function': {
                                    'name': clean_tool_name,
                                    'description': description,
                                    'parameters': params_schema
                                }
                            }
                            
                            # Create FunctionTool
                            function_tool = FunctionTool(
                                func=mock_func,
                                openai_tool_schema=openai_schema
                            )
                            function_tools.append(function_tool)
                            logger.debug(f"Added OpenAPI function tool: {clean_tool_name}")
                
                else:
                    # Regular BFCL function definition
                    # Extract tool info
                    if 'function' in tool:
                        tool_info = tool['function']
                    else:
                        tool_info = tool
                    
                    tool_name = tool_info.get('name', '')
                    tool_params = tool_info.get('parameters', {})
                    description = tool_info.get('description', '')
                    
                    # Clean tool name for OpenAI API
                    clean_tool_name = tool_name.replace('.', '_')
                    
                    # Create mock function
                    mock_func = make_mock_func(tool_name, description)
                    mock_func.__name__ = clean_tool_name
                    
                    # Create OpenAI schema
                    openai_schema = {
                        'type': 'function',
                        'function': {
                            'name': clean_tool_name,
                            'description': description,
                            'parameters': tool_params
                        }
                    }
                    
                    # Create FunctionTool
                    function_tool = FunctionTool(
                        func=mock_func,
                        openai_tool_schema=openai_schema
                    )
                    function_tools.append(function_tool)
                    logger.debug(f"Added BFCL function tool: {clean_tool_name}")
        
        logger.info(f"Created {len(function_tools)} function tools from {len(tools)} tool definitions")
        
        # Create agent with proper system message
        if system_prompt:
            # Use custom system prompt if provided
            system_message = system_prompt
        else:
            # Use default system message
            system_message = "You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\nPay attention to the description of the functions, which decide whether the function can be used to solve the task. If none of the functions can be used, point it out by saying \"irrelevant\". If the given question lacks the parameters required by the function, also point it out by saying \"missing parameters\".\n\nAt each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.\nPay attention to the tool call results. If a tool call fails, you will see the error message in the tool call result. Use the error message to help you solve the task better.\nFor the files mentioned in the task, you can assume that files are exist in the current directory."
        
        # Add retry context if provided
        if retry_context:
            system_message = f"{system_message}\n\n--- Retry Context ---\n{retry_context}"
        
        # Create agent
        agent = ChatAgent(
            model_name=model_name,
            temperature=temperature,
            system_message=system_message
        )
        
        # Set tools using set_tools method
        if function_tools:
            agent.set_tools(function_tools)
            logger.debug(f"Created agent with {len(function_tools)} tools for session {session_id}")
        else:
            logger.warning("No tools to set for agent")
        
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        raise


def execute_with_retries(agent: Any,
                         question: str,
                         evaluator: Any,
                         max_retries: int = 2,
                         context: List[str] = None,
                         session_id: str = None,
                         mock_client: MockServerClient = None) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Execute a question with retry logic, collecting success and failure examples
    
    Args:
        agent: Agent instance
        question: Question/task to execute
        evaluator: Task evaluator for scoring
        max_retries: Maximum retry attempts
        context: Previous completed tasks for context
        session_id: Current session ID
        mock_client: Mock server client
        
    Returns:
        Tuple of (success_examples, failure_examples, final_tool_calls)
    """
    success_examples = []
    failure_examples = []
    final_tool_calls = []
    
    previous_evaluation = None
    previous_tool_calls = None
    best_attempt = None  # Track the best attempt so far
    best_score = -1.0  # Track the best score
    
    for attempt in range(max_retries + 1):
        try:
            # Build full prompt with context
            full_prompt = _build_prompt_with_context(question, context, previous_evaluation, previous_tool_calls, attempt)
            
            # Generate response and execute tools
            response = agent.generate_response(full_prompt)
            
            # Extract tool calls
            tool_calls = []
            if response.success and response.tool_calls:
                tool_calls = response.tool_calls
            
            # Get updated state from mock server
            current_state = {}
            if mock_client and session_id:
                try:
                    current_state = mock_client.get_session_state(session_id)
                except:
                    pass
            
            # Evaluate completion
            if evaluator:
                evaluation = evaluator.evaluate_completion(
                    task=question,
                    tool_calls=tool_calls,
                    current_state=current_state,
                    initial_state=None,
                    tool_definitions=agent.tools if hasattr(agent, 'tools') else []
                )
            else:
                # No evaluator - simple pass/fail based on tool calls
                evaluation = TaskEvaluationResult(
                    score=1.0 if tool_calls else 0.0,
                    passed=bool(tool_calls),
                    retry_reason="No tool calls generated" if not tool_calls else None,
                    checklist=[],
                    judgment=[],
                    metadata={}
                )
            
            # Create example
            example = {
                'question': question,
                'tool_calls': tool_calls,
                'score': evaluation.score,
                'attempt': attempt,
                'state': current_state.copy() if current_state else {},
                'evaluation': evaluation.to_dict()
            }
            
            # Track best attempt (higher score, or same score but later attempt)
            # Since we iterate attempts in order, always update when score >= best_score to get latest
            if evaluation.score >= best_score:
                best_score = evaluation.score
                best_attempt = example
            
            # Categorize example
            if evaluation.passed or evaluation.score >= 1.0:
                success_examples.append(example)
                final_tool_calls = tool_calls
                logger.info(f"✅ Task completed on attempt {attempt} (score: {evaluation.score:.2f})")
                break
            else:
                failure_examples.append(example)
                previous_evaluation = evaluation
                previous_tool_calls = tool_calls
                logger.info(f"🔄 Attempt {attempt} failed (score: {evaluation.score:.2f}): {evaluation.retry_reason}")
                
                if attempt >= max_retries:
                    # Use best attempt's tool calls when max retries reached
                    if best_attempt:
                        final_tool_calls = best_attempt.get('tool_calls', [])
                        logger.warning(f"❌ Max retries reached. Using best attempt with score {best_score:.2f}")
                    else:
                        final_tool_calls = tool_calls  # Fallback to last attempt if no best found
                        logger.warning(f"❌ Max retries reached")
                    
        except Exception as e:
            logger.error(f"Error in attempt {attempt}: {e}")
            failure_examples.append({
                'question': question,
                'tool_calls': [],
                'score': 0.0,
                'attempt': attempt,
                'error': str(e)
            })
            
            if attempt >= max_retries:
                break
    
    return success_examples, failure_examples, final_tool_calls


def _build_prompt_with_context(question: str,
                               context: List[str] = None,
                               previous_evaluation: Any = None,
                               previous_tool_calls: List[Dict] = None,
                               attempt: int = 0) -> str:
    """
    Build prompt with context and retry information
    
    Args:
        question: Main question/task
        context: Previous completed tasks
        previous_evaluation: Previous attempt's evaluation
        previous_tool_calls: Previous attempt's tool calls
        attempt: Current attempt number
        
    Returns:
        Formatted prompt string
    """
    parts = []
    
    # Add context from previous tasks
    if context and attempt == 0:  # Only add context on first attempt
        parts.append("Previous completed tasks for reference:")
        for i, task in enumerate(context, 1):
            parts.append(f"{i}. {task}")
        parts.append("")
    
    # Add retry context
    if attempt > 0 and previous_evaluation:
        parts.append(f"--- Previous Attempt {attempt} Failed ---")
        
        if hasattr(previous_evaluation, 'score'):
            parts.append(f"Score: {previous_evaluation.score:.2f}")
        
        if hasattr(previous_evaluation, 'retry_reason') and previous_evaluation.retry_reason:
            parts.append(f"Reason: {previous_evaluation.retry_reason}")
        
        if hasattr(previous_evaluation, 'checklist') and previous_evaluation.checklist:
            failed_checks = [
                check for check in previous_evaluation.checklist 
                if not check.get('passed', False)
            ]
            if failed_checks:
                parts.append("\nFailed requirements:")
                for check in failed_checks[:3]:  # Limit to top 3
                    parts.append(f"  - {check.get('description', 'Unknown check')}: {check.get('reason', 'Failed')}")
        
        if previous_tool_calls:
            parts.append(f"\nPrevious tool calls ({len(previous_tool_calls)}):")
            for tc in previous_tool_calls[:3]:  # Show first 3
                tool_name = tc.get('name', tc.get('function', 'unknown'))
                parts.append(f"  - {tool_name}(...)")
        
        parts.append("\nPlease address the issues above and complete the task correctly.")
        parts.append("---\n")
    
    # Add main question
    parts.append(question)
    
    return "\n".join(parts)


def evaluate_task_completion(task: str,
                             tool_calls: List[Dict],
                             current_state: Dict,
                             evaluator: Any,
                             tool_definitions: List[Dict] = None) -> TaskEvaluationResult:
    """
    Evaluate task completion using the provided evaluator
    
    Args:
        task: Task description
        tool_calls: Tool calls made
        current_state: Current state after execution
        evaluator: Task evaluator instance
        tool_definitions: Available tool definitions
        
    Returns:
        TaskEvaluationResult
    """
    if not evaluator:
        # Default evaluation without evaluator
        return TaskEvaluationResult(
            score=1.0 if tool_calls else 0.0,
            passed=bool(tool_calls),
            retry_reason="No tool calls generated" if not tool_calls else None,
            checklist=[],
            judgment=[],
            metadata={}
        )
    
    # Use evaluator
    return evaluator.evaluate_completion(
        task=task,
        tool_calls=tool_calls,
        current_state=current_state,
        initial_state=None,
        tool_definitions=tool_definitions
    )


def extract_tool_definitions(test_case: Any) -> List[Dict]:
    """
    Extract tool definitions from various test case formats
    
    Args:
        test_case: Test case in various formats
        
    Returns:
        List of tool definitions
    """
    tools = []
    
    if isinstance(test_case, dict):
        # Direct dictionary format
        if 'function' in test_case:
            tools = test_case.get('function', [])
        elif 'functions' in test_case:
            tools = test_case.get('functions', [])
        elif 'tools' in test_case:
            tools = test_case.get('tools', [])
    elif hasattr(test_case, 'content'):
        # Wrapped TestCase format
        if hasattr(test_case.content, 'function'):
            tools = test_case.content.function
        elif isinstance(test_case.content, dict):
            tools = test_case.content.get('function', [])
    
    return tools if isinstance(tools, list) else []


def calculate_new_state(current_state: Dict,
                        tool_calls: List[Dict],
                        execution_results: List[Dict]) -> Dict:
    """
    Calculate new state based on tool execution results
    
    Args:
        current_state: Current state/config
        tool_calls: Tool calls that were made
        execution_results: Results from tool execution
        
    Returns:
        Updated state dictionary
    """
    new_state = current_state.copy() if current_state else {}
    
    # Simple state update based on results
    # This is a placeholder - actual implementation depends on the tool semantics
    for call, result in zip(tool_calls, execution_results):
        if result and isinstance(result, dict):
            # Update state based on result
            if 'state_update' in result:
                new_state.update(result['state_update'])
            elif 'output' in result and isinstance(result['output'], dict):
                # Try to extract state changes from output
                for key, value in result['output'].items():
                    if key in new_state:
                        new_state[key] = value
    
    return new_state


def extract_learning_patterns(success_examples: List[Dict],
                              failure_examples: List[Dict]) -> List[str]:
    """
    Extract learning patterns from examples
    
    Args:
        success_examples: List of successful examples
        failure_examples: List of failed examples
        
    Returns:
        List of pattern strings
    """
    patterns = []
    
    # Analyze failures for common issues
    if failure_examples:
        missing_tools = sum(1 for ex in failure_examples if not ex.get('tool_calls'))
        if missing_tools > 0:
            patterns.append(f"Ensure to use available tools ({missing_tools} attempts had no tool calls)")
        
        # Check for low scores
        low_scores = [ex for ex in failure_examples if ex.get('score', 0) < 0.5]
        if low_scores:
            patterns.append("Pay attention to task requirements and parameters")
    
    # Analyze success patterns
    if success_examples:
        # Check what made them successful
        for ex in success_examples[:1]:  # Look at first success
            if ex.get('attempt', 0) > 0:
                patterns.append(f"Success achieved after {ex['attempt']} retries - persistence is key")
            
            tool_count = len(ex.get('tool_calls', []))
            if tool_count == 1:
                patterns.append("Use only the necessary tools, avoid redundant calls")
            elif tool_count > 1:
                patterns.append(f"Multiple tools may be needed ({tool_count} used successfully)")
    
    # Default patterns if none found
    if not patterns:
        patterns.append("Follow the function specifications carefully")
        patterns.append("Provide all required parameters")
    
    return patterns[:5]  # Limit to 5 patterns

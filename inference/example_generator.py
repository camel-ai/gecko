"""
Dynamic Example Generator for Function Calling Tasks
Unified interface for single-turn and multi-turn example generation
"""

import logging
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

from inference.universal.config import InferenceConfig
from inference.universal.components import (
    create_session_with_state,
    create_agent_with_tools,
    execute_with_retries,
    evaluate_task_completion,
    calculate_new_state,
    extract_learning_patterns
)
from inference.client_engine import MockServerClient
from inference.evaluation import LLMTaskEvaluator, RuleBasedTaskEvaluator

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ExampleGenerator:
    """
    Dynamic example generator with unified interface for single and multi-turn tasks.
    
    Single-turn usage:
        generator = ExampleGenerator(tool_definitions=tools)
        examples = generator.generate("Calculate triangle area")
        
    Multi-turn usage:
        generator = ExampleGenerator(tools, initial_state={'balance': 1000})
        ex1 = generator.generate("Deposit $500")
        generator.update_state(ex1['final_tool_calls'], real_results)
        ex2 = generator.generate("Withdraw $200")  # Uses context automatically
    """
    
    def __init__(self,
                 tool_definitions: List[Dict] = None,
                 initial_state: Dict = None,
                 config: InferenceConfig = None,
                 system_prompt: str = None):
        """
        Initialize example generator
        
        Args:
            tool_definitions: Function/tool definitions (can be None/empty)
            initial_state: Initial config/state (can be None/empty)
            config: Inference configuration for model, retries, evaluator, etc.
            system_prompt: Optional system prompt to use for agent creation
        """
        # Store tools and state
        self.tool_definitions = tool_definitions or []
        self.current_state = initial_state.copy() if initial_state else {}
        self.state_history = [self.current_state.copy()]
        self.system_prompt = system_prompt
        
        # Track completed tasks for context
        self.completed_tasks = []
        self.turn_count = 0
        
        # Configuration
        self.config = config or InferenceConfig()
        
        # Initialize mock server client
        self.mock_client = MockServerClient(self.config.mock_server_url)
        
        # Initialize task evaluator
        self.task_evaluator = self._init_task_evaluator()
        
        # Track all examples for summary
        self.all_examples = []
        
        logger.info(f"ExampleGenerator initialized with {len(self.tool_definitions)} tools, "
                   f"model: {self.config.model_name}")
    
    def _init_task_evaluator(self):
        """Initialize task evaluator based on configuration"""
        if self.config.max_retries == 0:
            # No retries, no evaluator needed
            return None
            
        if self.config.task_evaluator_type == "rule_based":
            evaluator = RuleBasedTaskEvaluator(mode=self.config.rule_evaluator_mode)
            logger.info(f"Using RuleBasedTaskEvaluator (mode: {self.config.rule_evaluator_mode})")
        else:
            # Default to LLM evaluator
            evaluator = LLMTaskEvaluator(
                model_name=self.config.task_evaluator_model_name,
                enable_checklist=self.config.enable_checklist
            )
            logger.info(f"Using LLMTaskEvaluator (model: {self.config.task_evaluator_model_name})")
        
        return evaluator
    
    def generate(self, question: str, max_attempts: Optional[int] = None) -> Dict:
        """
        Generate examples for a single question/turn
        
        Args:
            question: The task/question to generate examples for
            max_attempts: Maximum retry attempts (overrides config if provided)
            
        Returns:
            Dictionary containing:
                - question: The input question
                - success_examples: List of successful attempts
                - failure_examples: List of failed attempts  
                - final_tool_calls: The final tool calls (from success or last attempt)
                - final_score: Final evaluation score
                - patterns: Extracted learning patterns
                - turn_index: Index of this turn
                - state_after: State after this turn's execution
        """
        max_attempts = max_attempts if max_attempts is not None else self.config.max_retries
        
        logger.info(f"Generating examples for turn {self.turn_count}: {question[:100]}...")
        
        try:
            # Create session with current state
            session_id = create_session_with_state(self.current_state, self.mock_client)
            
            # Create agent with tools
            agent = create_agent_with_tools(
                tools=self.tool_definitions,
                session_id=session_id,
                model_name=self.config.model_name,
                mock_server_url=self.config.mock_server_url,
                temperature=self.config.model_config.get('temperature', 0.1),
                system_prompt=self.system_prompt  # Pass system prompt if provided
            )
            
            # Execute with retries to collect examples
            success_examples, failure_examples, final_tool_calls = execute_with_retries(
                agent=agent,
                question=question,
                evaluator=self.task_evaluator,
                max_retries=max_attempts,
                context=self.completed_tasks,
                session_id=session_id,
                mock_client=self.mock_client
            )
            
            # Get final state from mock server
            state_after = {}
            try:
                state_after = self.mock_client.get_session_state(session_id)
            except Exception as e:
                logger.warning(f"Could not get final state from mock server: {e}")
                state_after = self.current_state.copy()
            
            # Calculate final score
            if success_examples:
                final_score = success_examples[-1].get('score', 1.0)
            elif failure_examples:
                final_score = failure_examples[-1].get('score', 0.0)
            else:
                final_score = 0.0
            
            # Extract patterns
            patterns = extract_learning_patterns(success_examples, failure_examples)
            
            # Mark task as completed (add to context for future turns)
            self.completed_tasks.append(question)
            
            # Create result
            result = {
                'question': question,
                'success_examples': success_examples,
                'failure_examples': failure_examples,
                'final_tool_calls': final_tool_calls,
                'final_score': final_score,
                'patterns': patterns,
                'turn_index': self.turn_count,
                'state_before': self.current_state.copy(),
                'state_after': state_after,
                'session_id': session_id
            }
            
            # Store for later retrieval
            self.all_examples.append(result)
            
            # Increment turn counter
            self.turn_count += 1
            
            # Update internal state with mock server's estimate
            # (can be overridden by update_state() if real execution is done)
            if state_after and state_after != self.current_state:
                self.current_state = state_after
                self.state_history.append(state_after.copy())
                logger.debug(f"Updated internal state from mock server estimate")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating examples: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return empty result on error
            return {
                'question': question,
                'success_examples': [],
                'failure_examples': [{
                    'question': question,
                    'tool_calls': [],
                    'score': 0.0,
                    'error': str(e)
                }],
                'final_tool_calls': [],
                'final_score': 0.0,
                'patterns': ["Error occurred during generation"],
                'turn_index': self.turn_count,
                'state_before': self.current_state.copy(),
                'state_after': self.current_state.copy(),
                'error': str(e)
            }
    
    def update_state(self, tool_calls: List[Dict], execution_results: List[Dict]) -> Dict:
        """
        Update internal state based on real tool execution results.
        Call this after executing real tools externally to calibrate the state.
        
        Args:
            tool_calls: The actual tool calls that were made
            execution_results: Results from real tool execution
            
        Returns:
            Updated state dictionary
        """
        # Calculate new state based on current state + real execution results
        new_state = calculate_new_state(self.current_state, tool_calls, execution_results)
        
        # Update internal state
        self.current_state = new_state
        self.state_history.append(new_state.copy())
        
        logger.info(f"State updated from real execution. State keys: {list(new_state.keys())}")
        
        return new_state
    
    def get_context(self) -> Dict:
        """
        Get current context information
        
        Returns:
            Dictionary with completed tasks, state history, and metadata
        """
        return {
            'completed_tasks': self.completed_tasks,
            'current_state': self.current_state.copy(),
            'state_history': self.state_history,
            'turn_count': self.turn_count,
            'is_multi_turn': self.turn_count > 1
        }
    
    def get_all_examples(self) -> Dict:
        """
        Get all examples generated so far
        
        Returns:
            Dictionary with all generated examples and summary statistics
        """
        total_success = sum(len(ex.get('success_examples', [])) for ex in self.all_examples)
        total_failure = sum(len(ex.get('failure_examples', [])) for ex in self.all_examples)
        
        return {
            'examples': self.all_examples,
            'total_turns': self.turn_count,
            'completed_tasks': self.completed_tasks,
            'state_history': self.state_history,
            'is_multi_turn': self.turn_count > 1,
            'summary': {
                'total_success_examples': total_success,
                'total_failure_examples': total_failure,
                'success_rate': total_success / (total_success + total_failure) if (total_success + total_failure) > 0 else 0
            }
        }
    
    def reset(self):
        """Reset the generator to initial state (useful for reusing the same instance)"""
        self.current_state = self.state_history[0].copy() if self.state_history else {}
        self.state_history = [self.current_state.copy()]
        self.completed_tasks = []
        self.turn_count = 0
        self.all_examples = []
        logger.info("ExampleGenerator reset to initial state")
    
    def format_for_injection(self, mode: str = 'structured') -> str:
        """
        Format all collected examples for injection into prompts
        
        Args:
            mode: Format mode ('structured', 'minimal', 'detailed')
            
        Returns:
            Formatted string ready for prompt injection
        """
        if not self.all_examples:
            return ""
        
        parts = []
        
        if mode == 'structured':
            parts.append("## Learning from Previous Attempts\n")
            
            # Add failure examples
            all_failures = []
            for ex in self.all_examples:
                all_failures.extend(ex.get('failure_examples', []))
            
            if all_failures:
                parts.append("### Failed Attempts (Avoid These Mistakes):")
                for i, fail in enumerate(all_failures[:3], 1):  # Limit to 3
                    parts.append(f"\n**Attempt {i}:**")
                    parts.append(f"- Question: {fail.get('question', 'N/A')[:100]}")
                    parts.append(f"- Score: {fail.get('score', 0):.2f}")
                    if fail.get('tool_calls'):
                        parts.append(f"- Tool calls: {len(fail['tool_calls'])} made")
                    else:
                        parts.append("- Tool calls: None (this was the problem)")
                    if 'error' in fail:
                        parts.append(f"- Error: {fail['error']}")
            
            # Add success examples
            all_successes = []
            for ex in self.all_examples:
                all_successes.extend(ex.get('success_examples', []))
            
            if all_successes:
                parts.append("\n### Successful Attempts (Follow These Patterns):")
                for i, success in enumerate(all_successes[:2], 1):  # Limit to 2
                    parts.append(f"\n**Success {i}:**")
                    parts.append(f"- Question: {success.get('question', 'N/A')[:100]}")
                    parts.append(f"- Score: {success.get('score', 1.0):.2f}")
                    parts.append(f"- Tool calls: {len(success.get('tool_calls', []))}")
                    if success.get('attempt', 0) > 0:
                        parts.append(f"- Succeeded after {success['attempt']} retries")
            
            # Add patterns
            all_patterns = []
            for ex in self.all_examples:
                all_patterns.extend(ex.get('patterns', []))
            
            if all_patterns:
                unique_patterns = list(dict.fromkeys(all_patterns))  # Remove duplicates
                parts.append("\n### Key Insights:")
                for pattern in unique_patterns[:5]:
                    parts.append(f"• {pattern}")
        
        elif mode == 'minimal':
            # Just the essential info
            if self.all_examples:
                latest = self.all_examples[-1]
                if latest.get('patterns'):
                    parts.append("Key insights from analysis:")
                    for pattern in latest['patterns'][:3]:
                        parts.append(f"• {pattern}")
        
        elif mode == 'detailed':
            # Include everything
            parts.append("## Complete Example History\n")
            for i, ex in enumerate(self.all_examples, 1):
                parts.append(f"### Turn {i}: {ex['question'][:50]}...")
                parts.append(f"- Final score: {ex['final_score']:.2f}")
                parts.append(f"- Success examples: {len(ex['success_examples'])}")
                parts.append(f"- Failure examples: {len(ex['failure_examples'])}")
                if ex['patterns']:
                    parts.append("- Patterns: " + ", ".join(ex['patterns'][:3]))
        
        return "\n".join(parts)
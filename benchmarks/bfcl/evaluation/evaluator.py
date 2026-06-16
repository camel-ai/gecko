
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import copy

from ..data.loader import BFCLDataLoader, TestCase, GroundTruth, get_bfcl_data_loader
from ..tools.executor import BFCLToolExecutor, ExecutionResult, get_bfcl_tool_executor

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    test_id: str
    passed: bool
    score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_results: List[ExecutionResult] = field(default_factory=list)
    state_comparison: Optional[Dict[str, Any]] = None


@dataclass
class BatchEvaluationSummary:
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float
    average_score: float
    results: List[EvaluationResult] = field(default_factory=list)
    passed_ids: List[str] = field(default_factory=list)
    failed_ids: List[str] = field(default_factory=list)


class BFCLEvaluator:
    
    def __init__(self, 
                 data_loader: Optional[BFCLDataLoader] = None,
                 tool_executor: Optional[BFCLToolExecutor] = None):
        self.data_loader = data_loader or get_bfcl_data_loader()
        self.tool_executor = tool_executor or get_bfcl_tool_executor()
        
        self._official_evaluator = self._load_official_evaluator()
    
    def _load_official_evaluator(self):
        try:
            from benchmarks.bfcl.official_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
                state_checker, response_checker
            )
            return {
                "state_checker": state_checker,
                "response_checker": response_checker
            }
        except ImportError:
            logger.warning("Official BFCL evaluator not available")
            return None
    
    def evaluate_single_test(self, test_id: str, 
                           agent_tool_calls: List[Dict[str, Any]],
                           final_state_only: bool = False) -> EvaluationResult:
        try:
            test_case = self.data_loader.load_test_case(test_id)
            ground_truth = self.data_loader.load_ground_truth(test_id)
            
            if not ground_truth:
                logger.warning(f"No ground truth found for test {test_id}")
                return EvaluationResult(
                    test_id=test_id,
                    passed=False,
                    error_message="No ground truth available"
                )
            
            agent_execution_results = self.tool_executor.execute_tool_calls(
                agent_tool_calls, reset_instances=True
            )
            
            ground_truth_execution_results = self.tool_executor.execute_tool_calls(
                ground_truth.ground_truth, reset_instances=True
            )
            
            if self._official_evaluator:
                passed, details = self._evaluate_with_official_checker(
                    test_case, agent_tool_calls, ground_truth.ground_truth, final_state_only
                )
            else:
                passed, details = self._evaluate_with_builtin_checker(
                    agent_execution_results, ground_truth_execution_results, final_state_only
                )
            
            score = 1.0 if passed else 0.0
            
            return EvaluationResult(
                test_id=test_id,
                passed=passed,
                score=score,
                details=details,
                execution_results=agent_execution_results
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed for test {test_id}: {e}")
            return EvaluationResult(
                test_id=test_id,
                passed=False,
                error_message=str(e)
            )

    def evaluate(self, test_case, execution_result) -> Dict[str, Any]:
        """
        Compatibility adapter for the BaseEvaluator interface used by the benchmark plugin.
        """
        tool_calls = []
        for call in getattr(execution_result, "tool_calls", []) or []:
            if isinstance(call, dict):
                tool_calls.append(
                    {
                        "function": call.get("name") or call.get("function"),
                        "arguments": call.get("arguments", {}),
                    }
                )

        result = self.evaluate_single_test(
            test_id=getattr(test_case, "id", None),
            agent_tool_calls=tool_calls,
            final_state_only=False,
        )
        return {
            "score": result.score,
            "passed": result.passed,
            "details": result.details,
            "error_message": result.error_message,
        }
    
    def _evaluate_with_official_checker(self, test_case: TestCase,
                                      agent_calls: List[Dict],
                                      ground_truth_calls: List[Dict],
                                      final_state_only: bool) -> Tuple[bool, Dict]:
        try:
            return self._evaluate_with_builtin_checker(
                [], [], final_state_only
            )
        except Exception as e:
            logger.error(f"Official evaluator failed: {e}")
            return False, {"error": str(e)}
    
    def _evaluate_with_builtin_checker(self, agent_results: List[ExecutionResult],
                                     ground_truth_results: List[ExecutionResult],
                                     final_state_only: bool) -> Tuple[bool, Dict]:
        details = {
            "agent_executions": len(agent_results),
            "ground_truth_executions": len(ground_truth_results),
            "agent_successes": sum(1 for r in agent_results if r.success),
            "ground_truth_successes": sum(1 for r in ground_truth_results if r.success)
        }
        
        if not all(r.success for r in agent_results):
            details["failure_reason"] = "Agent execution failed"
            return False, details
        
        if not all(r.success for r in ground_truth_results):
            details["failure_reason"] = "Ground truth execution failed"
            return False, details
        
        if len(agent_results) != len(ground_truth_results):
            details["failure_reason"] = "Different number of executions"
            return False, details
        
        for i, (agent_result, gt_result) in enumerate(zip(agent_results, ground_truth_results)):
            if not self._compare_execution_results(agent_result, gt_result):
                details["failure_reason"] = f"Result mismatch at step {i}"
                details[f"step_{i}_agent"] = agent_result.result
                details[f"step_{i}_ground_truth"] = gt_result.result
                return False, details
        
        details["success_reason"] = "All executions match"
        return True, details
    
    def _compare_execution_results(self, result1: ExecutionResult, result2: ExecutionResult) -> bool:
        return result1.result == result2.result
    
    def batch_evaluate(self, test_ids: List[str],
                      agent_results: Dict[str, List[Dict[str, Any]]],
                      final_state_only: bool = False) -> BatchEvaluationSummary:
        results = []
        passed_ids = []
        failed_ids = []
        
        for test_id in test_ids:
            if test_id not in agent_results:
                logger.warning(f"No agent result for test {test_id}")
                result = EvaluationResult(
                    test_id=test_id,
                    passed=False,
                    error_message="No agent result provided"
                )
            else:
                result = self.evaluate_single_test(
                    test_id, agent_results[test_id], final_state_only
                )
            
            results.append(result)
            
            if result.passed:
                passed_ids.append(test_id)
            else:
                failed_ids.append(test_id)
        
        total_tests = len(test_ids)
        passed_tests = len(passed_ids)
        failed_tests = len(failed_ids)
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        average_score = sum(r.score for r in results) / total_tests if total_tests > 0 else 0.0
        
        return BatchEvaluationSummary(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            pass_rate=pass_rate,
            average_score=average_score,
            results=results,
            passed_ids=self._sort_ids_numerically(passed_ids),
            failed_ids=self._sort_ids_numerically(failed_ids)
        )
    
    def evaluate_from_results_file(self, results_file_path: str,
                                 final_state_only: bool = False) -> BatchEvaluationSummary:
        try:
            with open(results_file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if content.startswith('['):
                results_data = json.loads(content)
            else:
                results_data = []
                for line in content.split('\n'):
                    if line.strip():
                        results_data.append(json.loads(line))
            
            agent_results = {}
            test_ids = []
            
            for result_item in results_data:
                test_id = result_item.get('test_id')
                if not test_id:
                    continue
                
                test_ids.append(test_id)
                
                tool_calls = []
                
                if 'all_real_tool_calls' in result_item:
                    tool_calls = result_item['all_real_tool_calls']
                elif 'turns' in result_item:
                    for turn in result_item['turns']:
                        if 'real_tool_calls' in turn:
                            tool_calls.extend(turn['real_tool_calls'])
                
                agent_results[test_id] = tool_calls
            
            return self.batch_evaluate(test_ids, agent_results, final_state_only)
            
        except Exception as e:
            logger.error(f"Failed to evaluate from results file {results_file_path}: {e}")
            return BatchEvaluationSummary(0, 0, 0, 0.0, 0.0)
    
    def evaluate_category(self, category: str, agent_results: Dict[str, List[Dict]],
                         final_state_only: bool = False) -> BatchEvaluationSummary:
        test_ids = self.data_loader.list_test_ids(category)
        return self.batch_evaluate(test_ids, agent_results, final_state_only)
    
    def _sort_ids_numerically(self, test_ids: List[str]) -> List[str]:
        def extract_numeric_id(test_id: str) -> int:
            try:
                parts = test_id.split('_')
                for part in reversed(parts):
                    if part.isdigit():
                        return int(part)
                
                import re
                numbers = re.findall(r'\d+', test_id)
                if numbers:
                    return int(numbers[0])
                
                return 0
            except:
                return 0
        
        return sorted(test_ids, key=extract_numeric_id)
    
    def get_evaluation_stats(self, summary: BatchEvaluationSummary) -> Dict[str, Any]:
        stats = {
            "total_tests": summary.total_tests,
            "passed_tests": summary.passed_tests,
            "failed_tests": summary.failed_tests,
            "pass_rate": summary.pass_rate,
            "average_score": summary.average_score,
            "passed_ids": summary.passed_ids,
            "failed_ids": summary.failed_ids
        }
        
        if summary.results:
            execution_times = [
                sum(er.execution_time for er in result.execution_results)
                for result in summary.results
                if result.execution_results
            ]
            
            if execution_times:
                stats.update({
                    "average_execution_time": sum(execution_times) / len(execution_times),
                    "total_execution_time": sum(execution_times)
                })
        
        return stats


_evaluator_instance = None

def get_bfcl_evaluator() -> BFCLEvaluator:
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = BFCLEvaluator()
    return _evaluator_instance


import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ResultFormatter:
    
    def __init__(self):
        self.supported_formats = ['json', 'jsonl', 'csv', 'report', 'bfcl_eval']
    
    def format_single_result(self, 
                            test_id: str,
                            execution_result: Dict[str, Any],
                            evaluation_result: Dict[str, Any],
                            benchmark_name: str,
                            format_type: str = 'json') -> Dict[str, Any]:
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}")
        
        base_result = {
            'test_id': test_id,
            'benchmark': benchmark_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'execution': self._clean_execution_result(execution_result),
            'evaluation': evaluation_result
        }
        
        if format_type == 'json':
            return self._format_json(base_result)
        elif format_type == 'jsonl':
            return self._format_jsonl(base_result)
        elif format_type == 'csv':
            return self._format_csv_row(base_result)
        elif format_type == 'report':
            return self._format_report_entry(base_result)
        elif format_type == 'bfcl_eval':
            return self._format_bfcl_eval(base_result)
        
        return base_result
    
    def format_batch_results(self, 
                           results: List[Dict[str, Any]], 
                           format_type: str = 'json',
                           include_summary: bool = True,
                           config: Dict[str, Any] = None) -> Dict[str, Any]:
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}")
        
        cleaned_results = []
        for result in results:
            cleaned = self._clean_result_redundancy(result)
            cleaned_results.append(cleaned)
        
        batch_result = {}
        
        if config:
            batch_result['config'] = config
        
        if include_summary:
            summary = self._generate_summary(cleaned_results)
            summary['retry_stats'] = self._generate_retry_stats(cleaned_results)
            batch_result['summary'] = summary
        
        batch_result['results'] = cleaned_results
        
        return batch_result
    
    def _clean_result_redundancy(self, result: Dict[str, Any]) -> Dict[str, Any]:
        
        if 'turns' in result and 'execution_summary' in result:
            return result
        
        execution_data = result.get('execution', {})
        if not execution_data:
            execution_data = {
                'success': result.get('success', False),
                'execution_time': result.get('execution_time', 0),
                'outputs_count': len(result.get('tool_calls', [])),
                'error': result.get('error', None)
            }
        
        execution_info = {
            'success': execution_data.get('success', False),
            'execution_time': execution_data.get('execution_time', 0),
            'outputs_count': execution_data.get('outputs_count', 0),
            'error': execution_data.get('error', None)
        }
        
        if 'test_time_scaling' in execution_data:
            execution_info['test_time_scaling'] = execution_data['test_time_scaling']
        
        cleaned = {
            'id': result.get('id') or result.get('test_id'),
            'category': result.get('category'),
            'question': result.get('question'),
            'execution': execution_info,
            'evaluation': result.get('evaluation', {}),
            'bfcl_data': result.get('bfcl_data', {})
        }
        
        if 'test_time_scaling' in result:
            cleaned['test_time_scaling'] = result['test_time_scaling']
        
        if 'function_definitions' in result:
            cleaned['function_definitions'] = result['function_definitions']
        
        if 'llm_costs' in result:
            cleaned['llm_costs'] = result['llm_costs']
        if 'total_llm_cost' in result:
            cleaned['total_llm_cost'] = result['total_llm_cost']
        
        if 'tool_calls' in result:
            tool_calls = result['tool_calls']
            if isinstance(tool_calls, list):
                enhanced_calls = []
                for call in tool_calls:
                    enhanced_call = call.copy()
                    if 'result' not in enhanced_call and 'response' in call:
                        enhanced_call['result'] = call['response']
                    enhanced_calls.append(enhanced_call)
                cleaned['tool_calls'] = enhanced_calls
            else:
                cleaned['tool_calls'] = tool_calls
        
        if 'execution' in result and 'tool_results' in result['execution']:
            cleaned['execution']['tool_results'] = result['execution']['tool_results']
        
        if 'execution' in result and 'outputs' in result['execution']:
            cleaned['execution']['outputs'] = result['execution']['outputs']
        
        if 'execution' in result and 'outputs' in result['execution']:
            outputs = result['execution']['outputs']
            if isinstance(outputs, list):
                for output in outputs:
                    if output.get('type') == 'function_call' and 'result' in output:
                        if 'tool_calls' not in cleaned:
                            cleaned['tool_calls'] = []
                        for tool_call in cleaned.get('tool_calls', []):
                            if (tool_call.get('function') == output.get('name') or 
                                tool_call.get('name') == output.get('name')):
                                tool_call['execution_result'] = output.get('result')
                                break
        
        return cleaned
    
    def _clean_execution_result(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = execution_result.copy()
        
        internal_fields = ['_internal', '__cache__', '_temp']
        for field in internal_fields:
            cleaned.pop(field, None)
        
        return cleaned
    
    def _format_json(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return result
    
    def _format_jsonl(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'id': result['test_id'],
            'benchmark': result['benchmark'],
            'success': result['execution'].get('success', False),
            'score': result['evaluation'].get('score', 0.0),
            'execution_time': result['execution'].get('execution_time', 0.0),
            'outputs_count': len(result['execution'].get('outputs', [])),
            'timestamp': result['timestamp']
        }
    
    def _format_csv_row(self, result: Dict[str, Any]) -> Dict[str, str]:
        return {
            'test_id': result['test_id'],
            'benchmark': result['benchmark'],
            'success': str(result['execution'].get('success', False)),
            'score': str(result['evaluation'].get('score', 0.0)),
            'execution_time': str(result['execution'].get('execution_time', 0.0)),
            'outputs_count': str(len(result['execution'].get('outputs', []))),
            'has_error': str(result['execution'].get('error') is not None),
            'timestamp': result['timestamp']
        }
    
    def _format_report_entry(self, result: Dict[str, Any]) -> Dict[str, Any]:
        execution = result['execution']
        evaluation = result['evaluation']
        
        return {
            'test_id': result['test_id'],
            'benchmark': result['benchmark'],
            'status': 'PASS' if execution.get('success', False) else 'FAIL',
            'score': evaluation.get('score', 0.0),
            'details': {
                'execution_time': execution.get('execution_time', 0.0),
                'outputs_count': len(execution.get('outputs', [])),
                'error': execution.get('error'),
                'evaluation_details': evaluation.get('details', {})
            },
            'timestamp': result['timestamp']
        }
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {'total': 0}
        
        total = len(results)
        
        success_count = 0
        total_score = 0.0
        total_time = 0.0
        total_llm_cost = 0.0
        cost_by_category = {}
        
        for result in results:
            execution = result.get('execution', {})
            execution_summary = result.get('execution_summary', {})
            evaluation = result.get('evaluation', {})
            
            if execution_summary.get('success', False) or execution.get('success', False):
                success_count += 1
            
            total_score += evaluation.get('score', 0.0)
            total_time += execution_summary.get('total_execution_time', 0.0) or execution.get('execution_time', 0.0)
            
            if 'total_llm_cost' in result:
                total_llm_cost += result['total_llm_cost']
            
            if 'llm_costs' in result:
                for cost_key, cost_value in result['llm_costs'].items():
                    # mock_server_response_generation, mock_server_config_update, mock_server_request_validation
                    category = cost_key
                    
                    if category not in cost_by_category:
                        cost_by_category[category] = 0.0
                    cost_by_category[category] += cost_value
        
        summary = {
            'total': total,
            'success_count': success_count,
            'failure_count': total - success_count,
            'success_rate': success_count / total if total > 0 else 0.0,
            'average_score': total_score / total if total > 0 else 0.0,
            'total_time': total_time,
            'average_time': total_time / total if total > 0 else 0.0,
            'total_llm_cost': total_llm_cost,
            'average_llm_cost': total_llm_cost / total if total > 0 else 0.0,
            'llm_cost_by_category': cost_by_category
        }
        
        benchmark_stats = {}
        for result in results:
            benchmark = result.get('benchmark') or result.get('benchmark_name') or 'unknown'
            if benchmark not in benchmark_stats:
                benchmark_stats[benchmark] = {'count': 0, 'success': 0}
            
            benchmark_stats[benchmark]['count'] += 1
            success_flag = (
                result.get('execution_summary', {}).get('success', False)
                or result.get('execution', {}).get('success', False)
            )
            if success_flag:
                benchmark_stats[benchmark]['success'] += 1
        
        summary['by_benchmark'] = benchmark_stats
        
        return summary
    
    def _generate_retry_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_retries = 0
        successful_after_retry = 0
        results_with_retry = 0
        
        for result in results:
            if 'execution_summary' in result:
                execution_summary = result.get('execution_summary', {})
                total_attempts = execution_summary.get('total_attempts', 1)
                executed_turns = execution_summary.get('executed_turns', 1) or 1
                baseline_attempts = max(1, int(executed_turns))
                retries_for_result = max(0, int(total_attempts) - baseline_attempts)
                
                if retries_for_result > 0:
                    results_with_retry += 1
                    total_retries += retries_for_result
                    
                    if execution_summary.get('success', False):
                        successful_after_retry += 1
            else:
                turns = result.get('turns', [])
                has_retry = False
                is_successful = result.get('execution', {}).get('success', False)
                
                for turn in turns:
                    retries = turn.get('retries', [])
                    if len(retries) > 1:
                        has_retry = True
                        total_retries += (len(retries) - 1)
                
                if has_retry:
                    results_with_retry += 1
                    if is_successful:
                        successful_after_retry += 1
        
        retry_success_rate = 0.0
        if results_with_retry > 0:
            retry_success_rate = successful_after_retry / results_with_retry
        
        return {
            'total_retries': total_retries,
            'successful_after_retry': successful_after_retry,
            'retry_success_rate': retry_success_rate
        }
    
    def export_results(self, 
                      results: Union[Dict[str, Any], List[Dict[str, Any]]], 
                      output_path: Union[str, Path],
                      format_type: str = 'json',
                      benchmark=None):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == 'json':
            self._export_json(results, output_path)
        elif format_type == 'jsonl':
            self._export_jsonl(results, output_path)
        elif format_type == 'csv':
            self._export_csv(results, output_path)
        elif format_type == 'report':
            self._export_report(results, output_path)
        elif format_type == 'bfcl_eval':
            self._export_bfcl_eval(results, output_path, benchmark)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        logger.info(f"Exported results to {output_path} in {format_type} format")
    
    def _export_json(self, results: Union[Dict, List], output_path: Path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    def _export_jsonl(self, results: Union[Dict, List], output_path: Path):
        if isinstance(results, dict) and 'results' in results:
            results_list = results['results']
        elif isinstance(results, list):
            results_list = results
        else:
            results_list = [results]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results_list:
                f.write(json.dumps(result, ensure_ascii=False, default=str) + '\n')
    
    def _export_csv(self, results: Union[Dict, List], output_path: Path):
        import csv
        
        if isinstance(results, dict) and 'results' in results:
            results_list = results['results']
        elif isinstance(results, list):
            results_list = results
        else:
            results_list = [results]
        
        if not results_list:
            return
        
        first_result = results_list[0]
        if 'execution' in first_result:
            csv_results = [self._format_csv_row(r) for r in results_list]
        else:
            csv_results = results_list
        
        fieldnames = csv_results[0].keys()
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_results)
    
    def _export_report(self, results: Union[Dict, List], output_path: Path):
        if isinstance(results, dict) and 'results' in results:
            results_list = results['results']
            summary = results.get('summary', {})
        elif isinstance(results, list):
            results_list = results
            summary = self._generate_summary(results_list)
        else:
            results_list = [results]
            summary = self._generate_summary(results_list)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Test Results Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Summary:\n")
            f.write(f"  Total Tests: {summary.get('total', 0)}\n")
            f.write(f"  Success: {summary.get('success_count', 0)}\n")
            f.write(f"  Failed: {summary.get('failure_count', 0)}\n")
            f.write(f"  Success Rate: {summary.get('success_rate', 0):.1%}\n")
            f.write(f"  Average Score: {summary.get('average_score', 0):.3f}\n")
            f.write(f"  Total Time: {summary.get('total_time', 0):.2f}s\n\n")
            
            f.write("Detailed Results:\n")
            f.write("-" * 50 + "\n")
            
            for result in results_list:
                if 'execution' in result:
                    report_result = self._format_report_entry(result)
                else:
                    report_result = result
                
                f.write(f"Test ID: {report_result['test_id']}\n")
                f.write(f"Status: {report_result.get('status', 'UNKNOWN')}\n")
                f.write(f"Score: {report_result.get('score', 0):.3f}\n")
                
                details = report_result.get('details', {})
                if details.get('error'):
                    f.write(f"Error: {details['error']}\n")
                
                f.write(f"Execution Time: {details.get('execution_time', 0):.2f}s\n")
                f.write("\n")
    
    def create_comparison_report(self, 
                               results_groups: Dict[str, List[Dict[str, Any]]], 
                               output_path: Union[str, Path]):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summaries = {}
        for group_name, results in results_groups.items():
            summaries[group_name] = self._generate_summary(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Summary Comparison:\n")
            f.write(f"{'Group':<20} {'Total':<8} {'Success':<8} {'Rate':<8} {'Avg Score':<10} {'Avg Time':<10}\n")
            f.write("-" * 70 + "\n")
            
            for group_name, summary in summaries.items():
                f.write(f"{group_name:<20} "
                       f"{summary.get('total', 0):<8} "
                       f"{summary.get('success_count', 0):<8} "
                       f"{summary.get('success_rate', 0):<8.1%} "
                       f"{summary.get('average_score', 0):<10.3f} "
                       f"{summary.get('average_time', 0):<10.2f}\n")
            
            f.write("\n")
            
        
        logger.info(f"Created comparison report: {output_path}")
    
    def _format_bfcl_eval(self, result: Dict[str, Any], benchmark=None) -> Dict[str, Any]:
        test_id = result.get('test_id') or result.get('id', '')
        execution = result.get('execution', {})
        
        def _read(obj: Any, key: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        category = result.get('category', '')
        is_formatted_multi_turn = (
            'turns' in result
            and result['turns']
            and (
                category.startswith('multi_turn')
                or result.get('benchmark') == 'tau2'
                or result.get('execution_summary', {}).get('total_turns', 1) > 1
            )
        )

        if is_formatted_multi_turn:
            multi_turn_result = []
            
            for turn in result['turns']:
                turn_calls = []
                source_tool_calls = []

                # For BFCL multi-turn, official answer should come from real task-agent calls.
                real_tool_calls = _read(turn, 'real_tool_calls', [])
                if isinstance(real_tool_calls, list) and real_tool_calls:
                    source_tool_calls = real_tool_calls
                else:
                    retries = _read(turn, 'retries', None)
                    if not retries:
                        # InferenceResult(turn dataclass) path
                        retries = _read(turn, 'all_attempts', [])
                    if isinstance(retries, list) and retries:
                        best_retry = None
                        for retry in retries:
                            if isinstance(retry, dict) and 'judgment' in retry and retry['judgment'].get('passed', False):
                                best_retry = retry
                                break
                        if not best_retry and retries:
                            best_retry = max(
                                retries,
                                key=lambda r: (
                                    (r.get('judgment', {}) if isinstance(r, dict) else {}).get('score', 0),
                                    retries.index(r),
                                ),
                            )

                        if isinstance(best_retry, dict) and 'tool_calls' in best_retry:
                            source_tool_calls = best_retry.get('tool_calls', [])

                for tool_call in source_tool_calls:
                    func_str = self._format_tool_call_as_string(tool_call, test_id, benchmark)
                    if func_str:
                        turn_calls.append(func_str)
                
                multi_turn_result.append(turn_calls)
            
            latency = result.get('execution_summary', {}).get('total_execution_time', 0)
            
            return {
                "id": test_id,
                "result": multi_turn_result,
                "input_token_count": 0,
                "output_token_count": 0,
                "latency": latency,
                "reasoning_content": ""
            }

        # Formatted single-turn BFCL result path:
        # read from the same formatted turn structure used by the main result file.
        if 'turns' in result and isinstance(result.get('turns'), list) and result['turns']:
            first_turn = result['turns'][0] if isinstance(result['turns'][0], dict) else {}
            source_tool_calls = first_turn.get('final_tool_calls')

            if not isinstance(source_tool_calls, list):
                source_tool_calls = []
                retries = first_turn.get('retries', [])
                if isinstance(retries, list) and retries:
                    best_retry = None
                    for retry in retries:
                        if isinstance(retry, dict) and 'judgment' in retry and retry['judgment'].get('passed', False):
                            best_retry = retry
                            break
                    if not best_retry:
                        best_retry = max(
                            retries,
                            key=lambda r: (
                                (r.get('judgment', {}) if isinstance(r, dict) else {}).get('score', 0),
                                retries.index(r),
                            ),
                        )
                    if isinstance(best_retry, dict):
                        source_tool_calls = best_retry.get('tool_calls', [])

            bfcl_result = []
            for tool_call in source_tool_calls:
                raw_function_name = tool_call.get('name', '') or tool_call.get('function', '')

                if benchmark and hasattr(benchmark, 'map_tool_call_to_original_function'):
                    function_name = benchmark.map_tool_call_to_original_function(test_id, raw_function_name)
                else:
                    function_name = raw_function_name.replace('post_', '').replace('_0_0', '')
                    if function_name.startswith('/'):
                        function_name = function_name[1:].replace('/', '_')
                    if function_name.startswith('simple_0_0'):
                        function_name = function_name.replace('simple_0_0', '').lstrip('_')

                arguments = tool_call.get('arguments', {})
                if isinstance(arguments, str):
                    try:
                        parsed_args = json.loads(arguments)
                        if isinstance(parsed_args, dict) and 'requestBody' in parsed_args:
                            args_json = json.dumps(parsed_args['requestBody'])
                        else:
                            args_json = arguments
                    except Exception:
                        args_json = arguments
                elif isinstance(arguments, dict):
                    if 'requestBody' in arguments:
                        args_json = json.dumps(arguments['requestBody'])
                    else:
                        args_json = json.dumps(arguments)
                else:
                    args_json = json.dumps(arguments)

                bfcl_result.append({function_name: args_json})

            latency = result.get('execution_summary', {}).get('total_execution_time', 0)
            return {
                "id": test_id,
                "result": bfcl_result,
                "input_token_count": 0,
                "output_token_count": 0,
                "latency": latency,
                "reasoning_content": ""
            }

        # ExecutionResult (pre-format) path:
        # reconstruct multi-turn answers from outputs[type=turn_result]
        outputs = result.get('outputs', []) if isinstance(result, dict) else []
        turn_events = []
        if isinstance(outputs, list):
            for item in outputs:
                if isinstance(item, dict) and item.get('type') == 'turn_result':
                    turn_events.append(item)

        if turn_events:
            turn_events.sort(key=lambda x: x.get('turn_index', 0))
            multi_turn_result = []
            for turn_event in turn_events:
                source_tool_calls = []
                real_calls = turn_event.get('real_tool_calls')
                if isinstance(real_calls, list) and real_calls:
                    source_tool_calls = real_calls
                else:
                    generic_calls = turn_event.get('tool_calls')
                    if isinstance(generic_calls, list):
                        source_tool_calls = generic_calls

                turn_calls = []
                for tool_call in source_tool_calls:
                    func_str = self._format_tool_call_as_string(tool_call, test_id, benchmark)
                    if func_str:
                        turn_calls.append(func_str)
                multi_turn_result.append(turn_calls)

            latency = execution.get('execution_time', 0)
            return {
                "id": test_id,
                "result": multi_turn_result,
                "input_token_count": 0,
                "output_token_count": 0,
                "latency": latency,
                "reasoning_content": ""
            }
        
        else:
            tool_calls = result.get('tool_calls', [])
            bfcl_result = []
            
            for tool_call in tool_calls:
                raw_function_name = tool_call.get('name', '') or tool_call.get('function', '')
                
                if benchmark and hasattr(benchmark, 'map_tool_call_to_original_function'):
                    function_name = benchmark.map_tool_call_to_original_function(test_id, raw_function_name)
                else:
                    function_name = raw_function_name.replace('post_', '').replace('_0_0', '')
                    if function_name.startswith('/'):
                        function_name = function_name[1:].replace('/', '_')
                    if function_name.startswith('simple_0_0'):
                        function_name = function_name.replace('simple_0_0', '').lstrip('_')
                
                arguments = tool_call.get('arguments', {})
                if isinstance(arguments, str):
                    try:
                        parsed_args = json.loads(arguments)
                        if isinstance(parsed_args, dict) and 'requestBody' in parsed_args:
                            args_json = json.dumps(parsed_args['requestBody'])
                        else:
                            args_json = arguments
                    except:
                        args_json = arguments
                elif isinstance(arguments, dict):
                    if 'requestBody' in arguments:
                        args_json = json.dumps(arguments['requestBody'])
                    else:
                        args_json = json.dumps(arguments)
                else:
                    args_json = json.dumps(arguments)
                
                bfcl_result.append({function_name: args_json})
            
            latency = execution.get('execution_time', 0)
            
            return {
                "id": test_id,
                "result": bfcl_result,
                "input_token_count": 0,
                "output_token_count": 0,
                "latency": latency,
                "reasoning_content": ""
            }
    
    def _format_tool_call_as_string(self, tool_call: Dict[str, Any], test_id: str, benchmark=None) -> str:
        raw_function_name = tool_call.get('function', '') or tool_call.get('name', '')
        
        function_name = raw_function_name
        
        multi_turn_mappings = {
            'gorillafilesystem_cd': 'cd',
            'gorillafilesystem_mkdir': 'mkdir',
            'gorillafilesystem_mv': 'mv',
            'gorillafilesystem_cp': 'cp',
            'gorillafilesystem_rm': 'rm',
            'gorillafilesystem_rmdir': 'rmdir',
            'gorillafilesystem_ls': 'ls',
            'gorillafilesystem_cat': 'cat',
            'gorillafilesystem_touch': 'touch',
            'gorillafilesystem_echo': 'echo',
            'gorillafilesystem_grep': 'grep',
            'gorillafilesystem_find': 'find',
            'gorillafilesystem_wc': 'wc',
            'gorillafilesystem_sort': 'sort',
            'gorillafilesystem_tail': 'tail',
            'gorillafilesystem_diff': 'diff',
            'gorillafilesystem_du': 'du',
            'gorillafilesystem_head': 'head',
            'gorillafilesystem_pwd': 'pwd',
            # Math API
            'mathapi_mean': 'mean',
            'mathapi_standard_deviation': 'standard_deviation',
            'mathapi_logarithm': 'logarithm',
            # Message API
            'messageapi_message_login': 'message_login',
            'messageapi_get_user_id': 'get_user_id',
            'messageapi_send_message': 'send_message',
            'messageapi_delete_message': 'delete_message',
            'messageapi_view_messages_sent': 'view_messages_sent',
            'messageapi_add_contact': 'add_contact',
            # Twitter API
            'twitterapi_authenticate_twitter': 'authenticate_twitter',
            'twitterapi_post_tweet': 'post_tweet',
            'twitterapi_comment': 'comment',
            # Ticket API
            'ticketapi_ticket_login': 'ticket_login',
            'ticketapi_create_ticket': 'create_ticket',
            'ticketapi_get_ticket': 'get_ticket',
            'ticketapi_edit_ticket': 'edit_ticket',
            'ticketapi_resolve_ticket': 'resolve_ticket',
        }
        
        if function_name in multi_turn_mappings:
            function_name = multi_turn_mappings[function_name]
        elif benchmark and hasattr(benchmark, 'map_tool_call_to_original_function'):
            function_name = benchmark.map_tool_call_to_original_function(test_id, raw_function_name)
        
        arguments = tool_call.get('arguments', {})
        
        params = {}
        if isinstance(arguments, str):
            try:
                params = json.loads(arguments)
            except:
                params = {'value': arguments}
        elif isinstance(arguments, dict):
            params = arguments
        
        if 'requestBody' in params:
            params = params['requestBody']
        
        param_strs = []
        for key, value in params.items():
            if isinstance(value, str):
                escaped = value.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                param_strs.append(f"{key}='{escaped}'")
            elif isinstance(value, bool):
                param_strs.append(f"{key}={value}")
            elif isinstance(value, (int, float)):
                param_strs.append(f"{key}={value}")
            elif isinstance(value, list):
                param_strs.append(f"{key}={value}")
            elif isinstance(value, dict):
                param_strs.append(f"{key}={json.dumps(value)}")
            else:
                param_strs.append(f"{key}={repr(value)}")
        
        if param_strs:
            return f"{function_name}({', '.join(param_strs)})"
        else:
            return f"{function_name}()"
    
    def _export_bfcl_eval(self, results: Union[Dict, List], output_path: Path, benchmark=None):
        with open(output_path, 'w', encoding='utf-8') as f:
            if isinstance(results, dict):
                if 'results' in results:
                    for result in results['results']:
                        bfcl_result = self._format_bfcl_eval(result, benchmark)
                        f.write(json.dumps(bfcl_result) + '\n')
                else:
                    bfcl_result = self._format_bfcl_eval(results, benchmark)
                    f.write(json.dumps(bfcl_result) + '\n')
            elif isinstance(results, list):
                for result in results:
                    bfcl_result = self._format_bfcl_eval(result, benchmark)
                    f.write(json.dumps(bfcl_result) + '\n')
    
    def append_single_result_to_bfcl_file(self, result: Dict[str, Any], output_path: Path, benchmark=None):
        bfcl_result = self._format_bfcl_eval(result, benchmark)
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(bfcl_result) + '\n')


import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from benchmarks.base.benchmark import BaseBenchmark
from benchmarks.base.execution_result import ExecutionResult
from utils.file_manager import FileManager
from utils.result_formatter import ResultFormatter
from utils.test_selector import TestSelector
from .config import InferenceConfig
from .engine import UniversalInferenceEngine

logger = logging.getLogger(__name__)


class UniversalOrchestrator:
    
    def __init__(self, 
                 benchmark: BaseBenchmark,
                 config: InferenceConfig,
                 output_dir: str = "results"):
        self.benchmark = benchmark
        self.config = config
        self.output_dir = Path(output_dir)
        
        self.inference_engine = UniversalInferenceEngine(config)
        self.inference_engine.set_benchmark(benchmark)
        
        self.file_manager = FileManager(self.output_dir)
        self.result_formatter = ResultFormatter()
        self.test_selector = TestSelector(benchmark)
        
        self.execution_stats = {
            "total_tests": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "start_time": None,
            "end_time": None,
            "total_time": 0
        }
        
        logger.info(f"Universal Orchestrator initialized for {benchmark.name}")
    
    def _extract_path_params_from_selection(self, test_selection: Dict[str, Any]) -> Dict[str, Any]:
        category = None
        if 'filters' in test_selection and 'category' in test_selection['filters']:
            category = test_selection['filters']['category']
        
        is_all = test_selection.get('all_tests', False)
        
        return {
            'category': category,
            'is_all': is_all,
            'benchmark_name': self.benchmark.name,
            'model_name': self.config.model_name
        }
    
    def run_benchmark_suite(self, 
                           test_selection: Dict[str, Any],
                           resume: bool = False) -> Dict[str, Any]:
        logger.info(f"🚀 Starting benchmark suite: {self.benchmark.name}")
        
        self._current_test_selection = test_selection
        self._current_resume_mode = resume
        
        try:
            target_test_ids = self._get_target_test_ids(test_selection)
            
            if resume:
                test_ids_to_run = self._filter_for_resume(target_test_ids, test_selection)
            else:
                test_ids_to_run = target_test_ids
            
            execution_results = self._execute_tests(test_ids_to_run, test_selection)
            
            formatted_results = self._format_execution_results(execution_results)
            
            final_results = self._format_and_save_results(formatted_results, test_selection, target_test_ids if resume else None)
            
            summary = self._generate_summary(final_results)
            
            logger.info(f"✅ Benchmark suite completed: {len(final_results)} tests executed")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Benchmark suite failed: {e}")
            raise
    
    def _get_target_test_ids(self, test_selection: Dict[str, Any]) -> List[str]:
        logger.info("📋 Selecting target tests...")
        
        all_test_ids = self.benchmark.list_test_ids(**test_selection.get('filters', {}))
        
        target_test_ids = self.test_selector.get_target_test_ids(
            all_test_ids=all_test_ids,
            **test_selection
        )
        
        validated_test_ids = self.test_selector.validate_test_ids(
            target_test_ids, all_test_ids
        )
        
        logger.info(f"Selected {len(validated_test_ids)} tests for execution")
        return validated_test_ids
    
    def _filter_for_resume(self, target_test_ids: List[str], test_selection: Dict[str, Any]) -> List[str]:
        logger.info("🔄 Applying resume filter...")
        
        path_params = self._extract_path_params_from_selection(test_selection)
        temp_file = self.file_manager.generate_output_path(
            path_params['benchmark_name'],
            path_params['model_name'],
            category=path_params['category'],
            is_all=path_params['is_all'],
            is_temp=True,
            extension='.jsonl'
        )
        
        completed_ids = set()
        invalid_ids = set()
        if temp_file.exists():
            try:
                import json
                with open(temp_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            result = json.loads(line)
                            test_id = result['id']
                            
                            result_data = result.get('result', [])
                            if not result_data or len(result_data) == 0:
                                invalid_ids.add(test_id)
                            else:
                                completed_ids.add(test_id)
                                
                logger.info(f"📋 Found {len(completed_ids)} completed tests and {len(invalid_ids)} invalid tests in temp file")
            except Exception as e:
                logger.warning(f"Failed to read temp file: {e}")
        
        target_ids_set = set(target_test_ids)
        missing_ids = target_ids_set - completed_ids
        invalid_in_target = invalid_ids & target_ids_set
        
        tests_to_run = missing_ids | invalid_in_target
        tests_to_run_list = sorted(list(tests_to_run))
        
        logger.info(f"📊 Resume analysis:")
        logger.info(f"   Target tests: {len(target_test_ids)}")  
        logger.info(f"   Completed tests: {len(completed_ids)}")
        logger.info(f"   Invalid tests: {len(invalid_in_target)}")
        logger.info(f"   Missing tests: {len(missing_ids)}")
        logger.info(f"   Total to run: {len(tests_to_run_list)}")
        
        return tests_to_run_list
    
    def _execute_tests(self, test_ids: List[str], test_selection: Dict[str, Any]) -> List[ExecutionResult]:
        if not test_ids:
            logger.info("No tests to execute")
            return []
        
        logger.info(f"🔧 Executing {len(test_ids)} tests with {self.config.max_workers} workers")
        
        self.execution_stats["total_tests"] = len(test_ids)
        self.execution_stats["start_time"] = time.time()
        
        if self.config.max_workers == 1:
            results = self._execute_sequential(test_ids, test_selection)
        else:
            results = self._execute_parallel(test_ids, test_selection)
        
        self.execution_stats["end_time"] = time.time()
        self.execution_stats["total_time"] = (
            self.execution_stats["end_time"] - self.execution_stats["start_time"]
        )
        
        return results
    
    def _execute_sequential(self, test_ids: List[str], test_selection: Dict[str, Any]) -> List[ExecutionResult]:
        results = []
        path_params = self._extract_path_params_from_selection(test_selection)
        temp_bfcl_file = self.file_manager.generate_output_path(
            path_params['benchmark_name'],
            path_params['model_name'],
            category=path_params['category'],
            is_all=path_params['is_all'],
            is_temp=True,
            extension='.jsonl'
        )
        
        for i, test_id in enumerate(test_ids, 1):
            logger.info(f"📊 Progress: {i}/{len(test_ids)} - {test_id}")
            
            try:
                result = self._execute_single_test(test_id)
                results.append(result)
                
                self._update_stats(result)

                try:
                    self.result_formatter.append_single_result_to_bfcl_file(
                        result.__dict__, temp_bfcl_file, self.benchmark
                    )
                except Exception as e:
                    logger.warning(f"Failed to write temp result for {test_id}: {e}")
                
                if self.config.save_intermediate_results:
                    self._save_intermediate_result(result, getattr(self, '_current_test_selection', None))
                    
            except Exception as e:
                logger.error(f"Failed to execute {test_id}: {e}")
                if self._is_untested_error(e):
                    logger.warning(f"Skipping untested task (not exported): {test_id}")
                    self.execution_stats["skipped_tests"] += 1
                    continue
                error_result = ExecutionResult(test_id=test_id)
                error_result.mark_completed(success=False, error=str(e))
                results.append(error_result)
                self._update_stats(error_result)
        
        return results
    
    def _execute_parallel(self, test_ids: List[str], test_selection: Dict[str, Any]) -> List[ExecutionResult]:
        results = []
        
        progress_bar = None
        if TQDM_AVAILABLE:
            progress_bar = tqdm(
                total=len(test_ids),
                desc="🚀 执行测试",
                unit="tests",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
        
        path_params = self._extract_path_params_from_selection(test_selection)
        output_file = self.file_manager.generate_output_path(
            path_params['benchmark_name'],
            path_params['model_name'],
            category=path_params['category'],
            is_all=path_params['is_all']
        )
        temp_bfcl_file = self.file_manager.generate_output_path(
            path_params['benchmark_name'],
            path_params['model_name'],
            category=path_params['category'],
            is_all=path_params['is_all'],
            is_temp=True,
            extension='.jsonl'
        )
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_test = {
                executor.submit(self._execute_single_test, test_id): test_id 
                for test_id in test_ids
            }
            
            completed = 0
            for future in as_completed(future_to_test):
                test_id = future_to_test[future]
                completed += 1
                
                if progress_bar:
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'current': test_id,
                        'workers': self.config.max_workers
                    })
                else:
                    logger.info(f"📊 Progress: {completed}/{len(test_ids)} - {test_id}")
                
                try:
                    result = future.result()
                    results.append(result)
                    self._update_stats(result)
                    
                    try:
                        self.result_formatter.append_single_result_to_bfcl_file(
                            result.__dict__, temp_bfcl_file, self.benchmark
                        )
                    except Exception as e:
                        logger.warning(f"Failed to write temp result for {test_id}: {e}")
                    
                    if self.config.save_intermediate_results:
                        self._save_intermediate_result(result, getattr(self, '_current_test_selection', None))
                        
                except Exception as e:
                    logger.error(f"Failed to execute {test_id}: {e}")
                    if self._is_untested_error(e):
                        logger.warning(f"Skipping untested task (not exported): {test_id}")
                        self.execution_stats["skipped_tests"] += 1
                        continue
                    error_result = ExecutionResult(test_id=test_id)
                    error_result.mark_completed(success=False, error=str(e))
                    results.append(error_result)
                    self._update_stats(error_result)
        
        if progress_bar:
            progress_bar.close()
        
        return results
    
    def _execute_single_test(self, test_id: str) -> ExecutionResult:
        logger.debug(f"Executing test: {test_id}")
        
        test_case = self.benchmark.load_test_case(test_id)
        
        execution_result = self.inference_engine.execute_test_case(test_case)
        
        return execution_result
    
    def _format_execution_results(self, execution_results: List[ExecutionResult]) -> List[Dict[str, Any]]:
        logger.info(f"📝 Formatting {len(execution_results)} execution results...")
        
        formatted_results = []
        
        for execution_result in execution_results:
            try:
                test_case = self.benchmark.load_test_case(execution_result.test_id)
                
                empty_evaluation = {}
                formatted_result = self.benchmark.format_result(test_case, execution_result, empty_evaluation)
                
                formatted_result['model'] = self.config.model_name
                formatted_result['benchmark'] = self.benchmark.name
                
                if 'final_score' in execution_result.metadata:
                    formatted_result['task_evaluation_score'] = execution_result.metadata['final_score']
                
                if hasattr(execution_result, 'llm_costs') and execution_result.llm_costs:
                    formatted_result['llm_costs'] = execution_result.llm_costs
                    total_cost = sum(execution_result.llm_costs.values())
                    formatted_result['total_llm_cost'] = total_cost
                    logger.debug(f"Test {execution_result.test_id} total LLM cost: ${total_cost:.4f}")
                
                formatted_results.append(formatted_result)
                
            except Exception as e:
                logger.error(f"Failed to format result for {execution_result.test_id}: {e}")
                error_result = {
                    'id': execution_result.test_id,
                    'format_error': str(e),
                    'execution_success': execution_result.success
                }
                formatted_results.append(error_result)
        
        return formatted_results
    
    def _merge_with_existing_results(self, new_results: List[Dict[str, Any]], target_test_ids: List[str], test_selection: Dict[str, Any]) -> List[Dict[str, Any]]:
        logger.info(f"🔄 Merging {len(new_results)} new results with existing results...")
        
        path_params = self._extract_path_params_from_selection(test_selection)
        output_file = self.file_manager.generate_output_path(
            path_params['benchmark_name'],
            path_params['model_name'],
            category=path_params['category'],
            is_all=path_params['is_all']
        )
        
        existing_results = []
        if output_file.exists():
            try:
                import json
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    existing_results = data.get('results', [])
                logger.info(f"📄 Loaded {len(existing_results)} existing results from {output_file}")
            except Exception as e:
                logger.warning(f"Failed to load existing results: {e}")
        
        new_results_dict = {result['id']: result for result in new_results}
        
        merged_results = []
        processed_ids = set()
        
        for existing_result in existing_results:
            test_id = existing_result['id']
            if test_id in new_results_dict:
                merged_results.append(new_results_dict[test_id])
                processed_ids.add(test_id)
                logger.debug(f"Updated result for {test_id}")
            else:
                merged_results.append(existing_result)
                processed_ids.add(test_id)
        
        for test_id, new_result in new_results_dict.items():
            if test_id not in processed_ids:
                merged_results.append(new_result)
                processed_ids.add(test_id)
                logger.debug(f"Added new result for {test_id}")
        
        logger.info(f"✅ Merged results: {len(merged_results)} total ({len(new_results)} updated/new)")
        return merged_results
    
    def _load_all_results_from_temp(self, test_selection: Dict[str, Any]) -> List[Dict[str, Any]]:
        path_params = self._extract_path_params_from_selection(test_selection)
        temp_bfcl_file = self.file_manager.generate_output_path(
            path_params['benchmark_name'],
            path_params['model_name'],
            category=path_params['category'],
            is_all=path_params['is_all'],
            is_temp=True,
            extension='.jsonl'
        )
        
        all_results = []
        if temp_bfcl_file.exists():
            import json
            
            unique_bfcl_results = {}
            with open(temp_bfcl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        bfcl_result = json.loads(line)
                        test_id = bfcl_result.get('id')
                        if test_id:
                            unique_bfcl_results[test_id] = bfcl_result
            
            for test_id, bfcl_result in unique_bfcl_results.items():
                execution_result = ExecutionResult(test_id=test_id)
                execution_result.tool_calls = bfcl_result.get('result', [])
                execution_result.mark_completed(success=True)
                execution_result.metrics['execution_time'] = bfcl_result.get('latency', 0)
                
                try:
                    test_case = self.benchmark.load_test_case(test_id)
                    empty_evaluation = {}
                    formatted_result = self.benchmark.format_result(test_case, execution_result, empty_evaluation)
                    all_results.append(formatted_result)
                except Exception as e:
                    logger.warning(f"Failed to format {test_id}: {e}")
        
        logger.info(f"📋 从temp文件加载并重新评估了 {len(all_results)} 个结果")
        return all_results
    
    def _finalize_eval_file_from_temp(self, test_selection: Dict[str, Any], eval_file: str):
        path_params = self._extract_path_params_from_selection(test_selection)
        temp_file = self.file_manager.generate_output_path(
            path_params['benchmark_name'],
            path_params['model_name'],
            category=path_params['category'],
            is_all=path_params['is_all'],
            is_temp=True,
            extension='.jsonl'
        )
        
        if not temp_file.exists():
            logger.warning(f"Temp file not found: {temp_file}")
            return
        
        import json
        
        results_dict = {}
        with open(temp_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    result = json.loads(line)
                    test_id = result.get('id')
                    if test_id:
                        results_dict[test_id] = result
        
        def extract_number(test_id):
            import re
            match = re.search(r'(\d+)', test_id)
            return int(match.group(1)) if match else 0
        
        sorted_results = sorted(results_dict.values(), key=lambda x: extract_number(x.get('id', '')))
        
        with open(eval_file, 'w', encoding='utf-8') as f:
            for result in sorted_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info(f"📝 生成BFCL评估文件: {len(sorted_results)}个唯一结果 -> {eval_file}")
    
    def _merge_bfcl_eval_results(self, new_results: List[Dict[str, Any]], bfcl_eval_file: str):
        from pathlib import Path
        import json
        
        bfcl_eval_path = Path(bfcl_eval_file)
        existing_bfcl_results = []
        
        if bfcl_eval_path.exists():
            try:
                with open(bfcl_eval_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            existing_bfcl_results.append(json.loads(line))
                logger.info(f"📄 Loaded {len(existing_bfcl_results)} existing BFCL results")
            except Exception as e:
                logger.warning(f"Failed to load existing BFCL results: {e}")
        
        new_results_dict = {}
        for result in new_results:
            if 'tool_calls' in result:
                new_results_dict[result['id']] = {
                    'id': result['id'],
                    'result': result.get('tool_calls', []),
                    'input_token_count': 0,
                    'output_token_count': 0,
                    'latency': result.get('execution', {}).get('execution_time', 0),
                    'reasoning_content': ""
                }
        
        merged_bfcl_results = []
        processed_ids = set()
        
        for existing_result in existing_bfcl_results:
            test_id = existing_result['id']
            if test_id in new_results_dict:
                merged_bfcl_results.append(new_results_dict[test_id])
                processed_ids.add(test_id)
            else:
                merged_bfcl_results.append(existing_result)
                processed_ids.add(test_id)
        
        for test_id, new_result in new_results_dict.items():
            if test_id not in processed_ids:
                merged_bfcl_results.append(new_result)
                processed_ids.add(test_id)
        
        def extract_number(test_id):
            import re
            match = re.search(r'(\d+)', test_id)
            return int(match.group(1)) if match else 0
        
        merged_bfcl_results.sort(key=lambda x: extract_number(x['id']))
        
        with open(bfcl_eval_path, 'w', encoding='utf-8') as f:
            for result in merged_bfcl_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ Merged BFCL results: {len(merged_bfcl_results)} total")
    
    def _format_and_save_results(self, evaluation_results: List[Dict[str, Any]], test_selection: Dict[str, Any], target_test_ids: List[str] = None) -> List[Dict[str, Any]]:
        logger.info(f"💾 Formatting and saving results...")
        
        if hasattr(self, '_current_resume_mode') and self._current_resume_mode:
            evaluation_results = self._load_all_results_from_temp(test_selection)
        
        def extract_number(test_id):
            import re
            match = re.search(r'(\d+)', test_id)
            return int(match.group(1)) if match else 0
        
        evaluation_results.sort(key=lambda x: extract_number(x['id']))
        logger.info(f"💾 Final result count: {len(evaluation_results)}")
        
        path_params = self._extract_path_params_from_selection(test_selection)
        output_file = self.file_manager.generate_output_path(
            path_params['benchmark_name'],
            path_params['model_name'],
            category=path_params['category'],
            is_all=path_params['is_all']
        )
        
        global_config = {
            'model': self.config.model_name,
            'evaluator_type': self.config.task_evaluator_type,
            'enable_checklist': self.config.enable_checklist,
            'max_retries': self.config.max_retries,
            'target_score': self.config.target_score,
            'enable_real_execution': self.config.enable_real_execution,
            'benchmark': self.benchmark.name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        formatted_batch = self.result_formatter.format_batch_results(
            evaluation_results,
            format_type='json',
            include_summary=True,
            config=global_config
        )

        self._export_process_traces(formatted_batch, output_file)
        
        self.result_formatter.export_results(
            formatted_batch,
            output_file,
            format_type='json'
        )
        
        eval_file = str(output_file).replace('.json', '_bfcl_eval.jsonl')
        
        if hasattr(self, '_current_resume_mode') and self._current_resume_mode:
            self._finalize_eval_file_from_temp(test_selection, eval_file)
        else:
            self.result_formatter.export_results(
                formatted_batch,
                eval_file,
                format_type='bfcl_eval',
                benchmark=self.benchmark,
            )

            # Temp BFCL file is still useful during execution/progress reporting,
            # but the finalized eval file should come from formatted_batch only.
            temp_file = self.file_manager.generate_output_path(
                path_params['benchmark_name'],
                path_params['model_name'],
                category=path_params['category'],
                is_all=path_params['is_all'],
                is_temp=True,
                extension='.jsonl'
            )
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove temp BFCL file {temp_file}: {e}")
        
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Evaluation format saved to: {eval_file}")
        
        self._saved_files = {
            'main_results': str(output_file),
            'eval_results': str(eval_file)
        }
        
        return evaluation_results
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        stats = self.execution_stats
        
        summary = {
            'benchmark': self.benchmark.name,
            'model': self.config.model_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            
            'execution_stats': stats,
            'total_tests': len(results),
            
            'success_rate': stats['successful_tests'] / max(1, stats['total_tests']),
            'average_time': stats['total_time'] / max(1, stats['total_tests']),
            
            'config': self.config.to_dict(),
            
            'benchmark_info': self.benchmark.get_benchmark_info()
        }
        
        if results:
            summary['result_analysis'] = self._analyze_results(results)
        
        summary['saved_files'] = getattr(self, '_saved_files', {})
        
        return summary
    
    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        analysis = {
            'total_results': len(results),
            'by_category': {},
            'score_distribution': {},
            'common_errors': {}
        }
        
        for result in results:
            category = result.get('category', 'unknown')
            if category not in analysis['by_category']:
                analysis['by_category'][category] = {'count': 0, 'success': 0}
            
            analysis['by_category'][category]['count'] += 1
            
            evaluation = result.get('evaluation', {})
            success_flag = (
                result.get('execution_summary', {}).get('success', False)
                or evaluation.get('passed', False)
            )
            if success_flag:
                analysis['by_category'][category]['success'] += 1
        
        return analysis

    def _export_process_traces(self, formatted_batch: Dict[str, Any], output_file: Path) -> None:
        import json

        results = formatted_batch.get('results', []) if isinstance(formatted_batch, dict) else []
        if not isinstance(results, list):
            return

        trace_dir = output_file.parent / f"{output_file.stem}_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            if not isinstance(result, dict):
                continue

            process_trace = result.get('process_trace')
            if not isinstance(process_trace, dict) or not process_trace:
                continue

            test_id = result.get('id') or result.get('test_id')
            if not test_id:
                continue

            trace_payload = {
                'id': test_id,
                'benchmark': self.benchmark.name,
                'model': self.config.model_name,
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'process_trace': process_trace,
            }

            trace_file = trace_dir / f"{test_id}.timeline.json"
            with open(trace_file, 'w', encoding='utf-8') as f:
                json.dump(trace_payload, f, indent=2, ensure_ascii=False)

            result['process_trace_file'] = str(trace_file)
            result.pop('process_trace', None)
    
    def _sort_results_by_id(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        import re
        
        def extract_numeric_id(test_id: str) -> int:
            if '-' in test_id:
                match = re.search(r'(\w+)_(\d+)-', test_id)
                if match:
                    return int(match.group(2))
            else:
                match = re.search(r'(\w+)_(\d+)', test_id)
                if match:
                    return int(match.group(2))
            
            return 999999
        
        try:
            results.sort(key=lambda x: extract_numeric_id(x.get('test_id', x.get('id', ''))))
            logger.debug("Results sorted by numeric ID")
        except Exception as e:
            logger.warning(f"Failed to sort results by ID: {e}")
        
        return results
    
    def _finalize_sorted_bfcl_file(self, temp_file: Path, final_file: Path):
        import json
        import re
        
        if not temp_file.exists():
            logger.warning(f"Temp BFCL file not found: {temp_file}")
            return
        
        def extract_numeric_id(test_id: str) -> int:
            if '-' in test_id:
                match = re.search(r'(\w+)_(\d+)-', test_id)
                if match:
                    return int(match.group(2))
            else:
                match = re.search(r'(\w+)_(\d+)', test_id)
                if match:
                    return int(match.group(2))
            return 999999
        
        try:
            bfcl_results = []
            with open(temp_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        bfcl_results.append(json.loads(line.strip()))
            
            bfcl_results.sort(key=lambda x: extract_numeric_id(x.get('id', '')))
            
            with open(final_file, 'w', encoding='utf-8') as f:
                for result in bfcl_results:
                    f.write(json.dumps(result) + '\n')
            
            logger.info(f"BFCL results sorted and saved to: {final_file}")
            
            temp_file.unlink()
            
        except Exception as e:
            logger.error(f"Failed to finalize sorted BFCL file: {e}")
            try:
                import shutil
                shutil.copy2(temp_file, final_file)
                logger.info(f"Copied unsorted temp file to: {final_file}")
            except Exception as copy_error:
                logger.error(f"Failed to copy temp file: {copy_error}")
    
    def _update_stats(self, result: ExecutionResult):
        if result.success:
            self.execution_stats["successful_tests"] += 1
        else:
            self.execution_stats["failed_tests"] += 1

    @staticmethod
    def _is_untested_error(error: Exception) -> bool:
        message = str(error)
        return "UNTESTED_TASK:" in message
    
    def _save_intermediate_result(self, result: ExecutionResult, test_selection: Dict[str, Any] = None):
        if test_selection:
            path_params = self._extract_path_params_from_selection(test_selection)
            temp_file = self.file_manager.generate_output_path(
                path_params['benchmark_name'],
                path_params['model_name'],
                category=path_params['category'],
                is_all=path_params['is_all'],
                is_temp=True
            )
        else:
            temp_file = self.file_manager.generate_output_path(
                self.benchmark.name,
                self.config.model_name,
                "temp"
            ).with_suffix('.tmp')
        
        simplified_result = {
            'test_id': result.test_id,
            'success': result.success,
            'error': result.error,
            'execution_time': result.get_execution_time(),
            'outputs_count': result.get_total_outputs()
        }
        
        self.file_manager.save_result(simplified_result, temp_file)
    
    def get_orchestrator_info(self) -> Dict[str, Any]:
        return {
            'orchestrator_type': 'UniversalOrchestrator',
            'benchmark': self.benchmark.name,
            'config': self.config.to_dict(),
            'output_dir': str(self.output_dir),
            'execution_stats': self.execution_stats,
            'components': {
                'inference_engine': self.inference_engine.get_engine_info(),
                'benchmark_info': self.benchmark.get_benchmark_info()
            }
        }

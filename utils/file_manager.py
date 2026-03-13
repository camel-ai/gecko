
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable

logger = logging.getLogger(__name__)


class FileManager:
    
    def __init__(self, base_dir: Union[str, Path] = "results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def load_existing_results(self, 
                            file_path: Union[str, Path], 
                            success_checker: Optional[Callable[[Dict], bool]] = None) -> Dict[str, Set[str]]:
        completed_tests = set()
        failed_tests = set()
        
        file_path = Path(file_path)
        if not file_path.exists():
            logger.info(f"No existing results file found: {file_path}")
            return {"completed_tests": completed_tests, "failed_tests": failed_tests}
        
        is_successful = success_checker or self._default_success_checker
        
        try:
            last_results = {}  # test_id -> result
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        result = json.loads(line)
                        test_id = self._extract_test_id(result)
                        if not test_id:
                            continue
                        
                        last_results[test_id] = result
                            
                    except json.JSONDecodeError:
                        continue
            
            for test_id, result in last_results.items():
                if is_successful(result):
                    completed_tests.add(test_id)
                else:
                    failed_tests.add(test_id)
            
            logger.info(f"📄 Resume: {len(completed_tests)} completed, {len(failed_tests)} failed tests loaded")
            
        except Exception as e:
            logger.warning(f"Failed to load existing results: {e}")
        
        return {"completed_tests": completed_tests, "failed_tests": failed_tests}
    
    def _extract_test_id(self, result: Dict) -> Optional[str]:
        for field in ['test_id', 'id', 'test_case_id', 'case_id']:
            if field in result and result[field]:
                return str(result[field])
        return None
    
    def _default_success_checker(self, result: Dict) -> bool:
        if 'success' in result:
            return bool(result['success'])
        
        if result.get('error') or result.get('failed'):
            return False
        
        if 'outputs' in result:
            outputs = result['outputs']
            if isinstance(outputs, list):
                return len(outputs) > 0
        
        execution_time = result.get('execution_time', result.get('latency', 0))
        if execution_time > 300:
            return False
        
        if 'result' in result:
            test_result = result['result']
            if isinstance(test_result, list):
                return len(test_result) > 0
            return bool(test_result)
        
        return True
    
    def save_result(self, result: Dict, file_path: Union[str, Path], mode: str = 'a'):
        file_path = Path(file_path)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, mode, encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False, default=str) + '\n')
        except Exception as e:
            logger.error(f"Failed to save result to {file_path}: {e}")
            raise
    
    def save_results_batch(self, results: List[Dict], file_path: Union[str, Path]):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False, default=str) + '\n')
            
            logger.info(f"Saved {len(results)} results to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save batch results: {e}")
            raise
    
    def load_all_results(self, file_path: Union[str, Path]) -> List[Dict]:
        results = []
        file_path = Path(file_path)
        
        if not file_path.exists():
            return results
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Failed to load results from {file_path}: {e}")
        
        return results
    
    def generate_output_path(self, 
                           benchmark_name: str,
                           model: str, 
                           mode: str = "batch",
                           category: str = None,
                           is_all: bool = False,
                           is_temp: bool = False,
                           is_score: bool = False,
                           extension: str = ".json") -> Path:
        # Category is required for new file structure
        if not category:
            raise ValueError("Category is required for new file structure")
        
        clean_model = model.replace(".", "_").replace("-", "_")
        clean_benchmark = benchmark_name.replace(".", "_").replace("-", "_")
        clean_category = category.replace(".", "_").replace("-", "_")
        
        dir_path = self.base_dir / clean_benchmark / clean_model
        
        dir_path.mkdir(parents=True, exist_ok=True)
        
        file_type = "all" if is_all else "batch"
        filename = f"{clean_category}_{file_type}"
        
        if is_score:
            filename += "_score"
        
        filename += extension
        
        if is_temp:
            filename += ".tmp"
        
        return dir_path / filename
    
    def create_backup(self, file_path: Union[str, Path]) -> Optional[Path]:
        file_path = Path(file_path)
        if not file_path.exists():
            return None
        
        timestamp = int(os.path.getmtime(file_path))
        backup_name = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
        backup_path = file_path.parent / backup_name
        
        try:
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None
    
    def cleanup_temp_files(self, pattern: str = "temp_*"):
        try:
            deleted_count = 0
            for temp_file in self.base_dir.glob(pattern):
                temp_file.unlink()
                deleted_count += 1
                logger.debug(f"Deleted temp file: {temp_file}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} temp files")
                
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")
    
    def get_file_stats(self, 
                      file_path: Union[str, Path], 
                      success_checker: Optional[Callable[[Dict], bool]] = None) -> Dict[str, Any]:
        file_path = Path(file_path)
        if not file_path.exists():
            return {"exists": False}
        
        try:
            results = self.load_all_results(file_path)
            is_successful = success_checker or self._default_success_checker
            successful = sum(1 for r in results if is_successful(r))
            
            return {
                "exists": True,
                "size": file_path.stat().st_size,
                "total_results": len(results),
                "successful_results": successful,
                "failed_results": len(results) - successful,
                "success_rate": successful / len(results) if results else 0
            }
        except Exception as e:
            logger.error(f"Failed to get file stats: {e}")
            return {"exists": True, "error": str(e)}
    
    def merge_result_files(self, 
                          input_files: List[Union[str, Path]], 
                          output_file: Union[str, Path],
                          dedup_key: str = "test_id") -> int:
        all_results = {}
        
        for input_file in input_files:
            results = self.load_all_results(input_file)
            for result in results:
                key = result.get(dedup_key)
                if key:
                    all_results[key] = result
        
        merged_results = list(all_results.values())
        self.save_results_batch(merged_results, output_file)
        
        logger.info(f"Merged {len(merged_results)} unique results to {output_file}")
        return len(merged_results)
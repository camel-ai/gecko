
import json
import os
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    test_id: str
    category: str
    question: List[Dict[str, Any]]
    function: List[Dict[str, Any]]
    initial_config: Optional[Dict[str, Any]] = None
    involved_classes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundTruth:
    test_id: str
    ground_truth: List[Dict[str, Any]]
    execution_result: Optional[Dict[str, Any]] = None
    expected_state: Optional[Dict[str, Any]] = None


class BFCLDataLoader:
    
    def __init__(self, data_dir: Optional[str] = None, ground_truth_dir: Optional[str] = None):
        self.data_dir = self._resolve_data_dir(data_dir)
        self.ground_truth_dir = self._resolve_ground_truth_dir(ground_truth_dir)
        
        self._test_cases_cache: Dict[str, TestCase] = {}
        self._ground_truth_cache: Dict[str, GroundTruth] = {}
        self._categories_cache: Dict[str, List[str]] = {}
        
        logger.info(f"BFCL Data Loader initialized with data_dir: {self.data_dir}")
    
    def _resolve_data_dir(self, data_dir: Optional[str]) -> str:
        if data_dir:
            return data_dir

        return os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "bfcl_v4", "task")

    def _resolve_ground_truth_dir(self, ground_truth_dir: Optional[str]) -> str:
        if ground_truth_dir:
            return ground_truth_dir

        return os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "bfcl_v4", "possible_answer")
    
    def list_categories(self) -> List[str]:
        categories = set()
        
        if os.path.exists(self.data_dir):
            for file_name in os.listdir(self.data_dir):
                if file_name.endswith('.json'):
                    category = self._extract_category_from_filename(file_name)
                    if category:
                        categories.add(category)
        
        return sorted(list(categories))
    
    def _extract_category_from_filename(self, filename: str) -> Optional[str]:
        name = filename.replace('.json', '').lower()
        
        if 'bfcl_v4_' in name:
            category_part = name.replace('bfcl_v4_', '')
            return category_part
        
        elif 'bfcl_v3_' in name:
            category_part = name.replace('bfcl_v3_', '')
            
            return category_part
        
        elif 'multi_turn' in name:
            return 'multi_turn'
        elif 'single_turn' in name:
            return 'single_turn'
        elif 'live_simple' in name:
            return 'live_simple'
        elif 'live_multiple' in name:
            return 'live_multiple'
        elif 'live_parallel' in name:
            return 'live_parallel'
        elif 'live_irrelevance' in name:
            return 'live_irrelevance'
        elif 'simple' in name:
            return 'simple'
        elif 'multiple' in name:
            return 'multiple'
        elif 'parallel' in name:
            return 'parallel'
        elif 'irrelevance' in name:
            return 'irrelevance'
        elif 'java' in name:
            return 'java'
        else:
            return 'unknown'
    
    def list_test_ids(self, category: Optional[str] = None) -> List[str]:
        if category:
            return self._load_category_test_ids(category)
        else:
            all_ids = []
            for cat in self.list_categories():
                all_ids.extend(self._load_category_test_ids(cat))
            return sorted(list(set(all_ids)))
    
    def _load_category_test_ids(self, category: str) -> List[str]:
        if category in self._categories_cache:
            return self._categories_cache[category]
        
        test_ids = []
        
        data_files = self._find_data_files(category)
        
        for data_file in data_files:
            try:
                ids = self._extract_ids_from_file(data_file)
                test_ids.extend(ids)
            except Exception as e:
                logger.error(f"Failed to load test IDs from {data_file}: {e}")
        
        self._categories_cache[category] = test_ids
        return test_ids
    
    def _find_data_files(self, category: str) -> List[str]:
        data_files = []
        
        if not os.path.exists(self.data_dir):
            return data_files
        
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.json'):
                file_category = self._extract_category_from_filename(file_name)
                if file_category == category:
                    data_files.append(os.path.join(self.data_dir, file_name))
        
        return data_files
    
    def _extract_ids_from_file(self, file_path: str) -> List[str]:
        test_ids = []
        
        # Skip format_sensitivity.json - it's a mapping file, not test data
        if 'format_sensitivity.json' in file_path:
            return test_ids
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    test_id = data.get('id')
                    if test_id:
                        test_ids.append(test_id)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num} in {file_path}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num} in {file_path}: {e}")
        
        return test_ids
    
    def load_test_case(self, test_id: str) -> TestCase:
        if test_id in self._test_cases_cache:
            return self._test_cases_cache[test_id]
        
        test_case = None
        for category in self.list_categories():
            data_files = self._find_data_files(category)
            
            for data_file in data_files:
                test_case = self._search_test_case_in_file(test_id, data_file, category)
                if test_case:
                    break
            
            if test_case:
                break
        
        if not test_case:
            raise ValueError(f"Test case '{test_id}' not found")
        
        self._test_cases_cache[test_id] = test_case
        return test_case
    
    def _search_test_case_in_file(self, test_id: str, file_path: str, category: str) -> Optional[TestCase]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get('id') == test_id:
                            return TestCase(
                                test_id=test_id,
                                category=category,
                                question=data.get('question', []),
                                function=data.get('function', []),
                                initial_config=data.get('initial_config'),
                                involved_classes=data.get('involved_classes', []),
                                metadata={
                                    'source_file': file_path,
                                    'raw_data': data
                                }
                            )
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error searching in file {file_path}: {e}")
        
        return None
    
    def load_ground_truth(self, test_id: str) -> Optional[GroundTruth]:
        if test_id in self._ground_truth_cache:
            return self._ground_truth_cache[test_id]
        
        if not os.path.exists(self.ground_truth_dir):
            logger.warning(f"Ground truth directory not found: {self.ground_truth_dir}")
            return None
        
        ground_truth = None
        for file_name in os.listdir(self.ground_truth_dir):
            if file_name.endswith('.json'):
                file_path = os.path.join(self.ground_truth_dir, file_name)
                ground_truth = self._search_ground_truth_in_file(test_id, file_path)
                if ground_truth:
                    break
        
        if ground_truth:
            self._ground_truth_cache[test_id] = ground_truth
        
        return ground_truth
    
    def _search_ground_truth_in_file(self, test_id: str, file_path: str) -> Optional[GroundTruth]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get('id') == test_id:
                            return GroundTruth(
                                test_id=test_id,
                                ground_truth=data.get('ground_truth', []),
                                execution_result=data.get('execution_result'),
                                expected_state=data.get('expected_state')
                            )
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error searching ground truth in file {file_path}: {e}")
        
        return None
    
    def get_test_case_info(self, test_id: str) -> Dict[str, Any]:
        try:
            test_case = self.load_test_case(test_id)
            return {
                "test_id": test_case.test_id,
                "category": test_case.category,
                "total_turns": len(test_case.question),
                "involved_classes": test_case.involved_classes,
                "has_initial_config": test_case.initial_config is not None,
                "function_count": len(test_case.function),
                "first_turn_content": test_case.question[0].get("content", "") if test_case.question else ""
            }
        except Exception as e:
            logger.error(f"Failed to get test case info for {test_id}: {e}")
            return {"test_id": test_id, "error": str(e)}
    
    def batch_load_test_cases(self, test_ids: List[str]) -> Dict[str, TestCase]:
        results = {}
        
        for test_id in test_ids:
            try:
                results[test_id] = self.load_test_case(test_id)
            except Exception as e:
                logger.error(f"Failed to load test case {test_id}: {e}")
        
        return results
    
    def get_category_stats(self, category: str) -> Dict[str, Any]:
        test_ids = self.list_test_ids(category)
        
        stats = {
            "category": category,
            "total_tests": len(test_ids),
            "data_files": self._find_data_files(category)
        }
        
        sample_size = min(10, len(test_ids))
        if sample_size > 0:
            sample_ids = test_ids[:sample_size]
            turn_counts = []
            class_counts = []
            
            for test_id in sample_ids:
                try:
                    info = self.get_test_case_info(test_id)
                    turn_counts.append(info.get("total_turns", 0))
                    class_counts.append(len(info.get("involved_classes", [])))
                except:
                    continue
            
            if turn_counts:
                stats.update({
                    "avg_turns": sum(turn_counts) / len(turn_counts),
                    "max_turns": max(turn_counts),
                    "min_turns": min(turn_counts)
                })
            
            if class_counts:
                stats.update({
                    "avg_classes": sum(class_counts) / len(class_counts),
                    "max_classes": max(class_counts),
                    "min_classes": min(class_counts)
                })
        
        return stats


_data_loader_instance = None

def get_bfcl_data_loader() -> BFCLDataLoader:
    global _data_loader_instance
    if _data_loader_instance is None:
        _data_loader_instance = BFCLDataLoader()
    return _data_loader_instance
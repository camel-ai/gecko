
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class TestSelector:
    
    def __init__(self, benchmark=None):
        self.benchmark = benchmark
    
    def get_target_test_ids(self, 
                           all_test_ids: Optional[List[str]] = None,
                           ids: Optional[str] = None,
                           ids_file: Optional[str] = None, 
                           all_tests: bool = False,
                           pattern: Optional[str] = None,
                           **filters) -> List[str]:
        if all_test_ids is None:
            if self.benchmark is None:
                raise ValueError("Either all_test_ids or benchmark must be provided")
            all_test_ids = self.benchmark.list_test_ids(**filters)
        
        if all_tests:
            target_ids = all_test_ids.copy()
        elif ids:
            target_ids = self._parse_ids_string(ids)
        elif ids_file:
            target_ids = self._load_ids_from_file(ids_file)
        elif pattern:
            target_ids = self._filter_by_pattern(all_test_ids, pattern)
        else:
            raise ValueError("Must specify one of: all_tests, ids, ids_file, or pattern")
        
        return target_ids
    
    def _parse_ids_string(self, ids_string: str) -> List[str]:
        if not ids_string:
            return []
        
        separators = [',', ';', ' ', '\\n']
        ids = [ids_string]
        
        for sep in separators:
            new_ids = []
            for id_part in ids:
                new_ids.extend(id_part.split(sep))
            ids = new_ids
        
        cleaned_ids = []
        for test_id in ids:
            test_id = test_id.strip()
            if test_id:
                cleaned_ids.append(test_id)
        
        return cleaned_ids
    
    def _load_ids_from_file(self, ids_file: str) -> List[str]:
        ids_path = Path(ids_file)
        if not ids_path.exists():
            raise FileNotFoundError(f"IDs file not found: {ids_file}")
        
        ids = []
        try:
            with open(ids_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        line_ids = self._parse_ids_string(line)
                        ids.extend(line_ids)
            
            logger.info(f"Loaded {len(ids)} test IDs from {ids_file}")
            
        except Exception as e:
            logger.error(f"Failed to load IDs from {ids_file}: {e}")
            raise
        
        return ids
    
    def _filter_by_pattern(self, all_test_ids: List[str], pattern: str) -> List[str]:
        try:
            regex = re.compile(pattern)
            matched_ids = [test_id for test_id in all_test_ids if regex.search(test_id)]
            logger.info(f"Pattern '{pattern}' matched {len(matched_ids)} tests")
            return matched_ids
        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern}': {e}")
            raise
    
    def validate_test_ids(self, 
                         test_ids: List[str], 
                         all_test_ids: Optional[List[str]] = None,
                         **filters) -> List[str]:
        if all_test_ids is None:
            if self.benchmark is None:
                logger.warning("Cannot validate test IDs without benchmark or all_test_ids")
                return test_ids
            all_test_ids = self.benchmark.list_test_ids(**filters)
        
        valid_test_ids = set(all_test_ids)
        validated_ids = []
        invalid_ids = []
        
        for test_id in test_ids:
            if test_id in valid_test_ids:
                validated_ids.append(test_id)
            else:
                invalid_ids.append(test_id)
        
        if invalid_ids:
            logger.warning(f"Invalid test IDs found: {invalid_ids}")
        
        logger.info(f"Validated {len(validated_ids)} out of {len(test_ids)} test IDs")
        return validated_ids
    
    def filter_tests_for_resume(self, 
                               target_test_ids: List[str],
                               completed_tests: Set[str],
                               failed_tests: Set[str],
                               retry_failed: bool = True) -> List[str]:
        tests_to_run = []
        
        for test_id in target_test_ids:
            if test_id in completed_tests:
                logger.debug(f"Skipping completed test: {test_id}")
                continue
            elif test_id in failed_tests:
                if retry_failed:
                    tests_to_run.append(test_id)
                    logger.debug(f"Retrying failed test: {test_id}")
                else:
                    logger.debug(f"Skipping failed test: {test_id}")
            else:
                tests_to_run.append(test_id)
        
        logger.info(f"Resume: {len(tests_to_run)} tests to run "
                   f"(skipped {len(target_test_ids) - len(tests_to_run)} completed/failed)")
        
        return tests_to_run
    
    def sort_test_ids(self, test_ids: List[str], sort_method: str = "natural") -> List[str]:
        if sort_method == "natural":
            return self._natural_sort(test_ids)
        elif sort_method == "alphabetical":
            return sorted(test_ids)
        elif sort_method == "reverse":
            return sorted(test_ids, reverse=True)
        else:
            logger.warning(f"Unknown sort method '{sort_method}', using natural sort")
            return self._natural_sort(test_ids)
    
    def _natural_sort(self, test_ids: List[str]) -> List[str]:
        def natural_key(text):
            def try_int(s):
                try:
                    return int(s)
                except ValueError:
                    return s
            
            return [try_int(c) for c in re.split(r'(\d+)', text)]
        
        return sorted(test_ids, key=natural_key)
    
    def group_tests_by_attribute(self, 
                                test_ids: List[str],
                                attribute_extractor: Callable[[str], str]) -> Dict[str, List[str]]:
        groups = {}
        
        for test_id in test_ids:
            try:
                attribute = attribute_extractor(test_id)
                if attribute not in groups:
                    groups[attribute] = []
                groups[attribute].append(test_id)
            except Exception as e:
                logger.warning(f"Failed to extract attribute for {test_id}: {e}")
                if 'unknown' not in groups:
                    groups['unknown'] = []
                groups['unknown'].append(test_id)
        
        return groups
    
    def get_test_summary(self, test_ids: List[str]) -> Dict[str, Any]:
        summary = {
            'total_tests': len(test_ids),
            'first_test': test_ids[0] if test_ids else None,
            'last_test': test_ids[-1] if test_ids else None,
        }
        
        if test_ids:
            prefixes = set()
            for test_id in test_ids[:10]:
                parts = test_id.split('_')
                if len(parts) > 1:
                    prefixes.add(parts[0])
            
            if len(prefixes) == 1:
                summary['common_prefix'] = list(prefixes)[0]
            elif len(prefixes) > 1:
                summary['multiple_prefixes'] = list(prefixes)
        
        return summary
    
    def create_test_batches(self, 
                          test_ids: List[str], 
                          batch_size: int,
                          strategy: str = "sequential") -> List[List[str]]:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if strategy == "sequential":
            batches = []
            for i in range(0, len(test_ids), batch_size):
                batches.append(test_ids[i:i + batch_size])
            return batches
        
        elif strategy == "round_robin":
            num_batches = (len(test_ids) + batch_size - 1) // batch_size
            batches = [[] for _ in range(num_batches)]
            
            for i, test_id in enumerate(test_ids):
                batch_idx = i % num_batches
                batches[batch_idx].append(test_id)
            
            return [batch for batch in batches if batch]
        
        elif strategy == "balanced":
            num_batches = (len(test_ids) + batch_size - 1) // batch_size
            batches = [[] for _ in range(num_batches)]
            
            for i, test_id in enumerate(test_ids):
                batch_idx = i % num_batches
                batches[batch_idx].append(test_id)
            
            return [batch for batch in batches if batch]
        
        else:
            raise ValueError(f"Unknown batching strategy: {strategy}")
    
    def save_test_selection(self, 
                          test_ids: List[str], 
                          output_file: Union[str, Path],
                          include_metadata: bool = True):
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if include_metadata:
                f.write(f"# Test selection generated at {logger.getEffectiveLevel()}\\n")
                f.write(f"# Total tests: {len(test_ids)}\\n")
                f.write(f"# \\n")
            
            for test_id in test_ids:
                f.write(f"{test_id}\\n")
        
        logger.info(f"Saved {len(test_ids)} test IDs to {output_path}")

from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class TestCaseAdapter:
    
    @staticmethod
    def get_id(test_case: Any) -> str:
        if hasattr(test_case, 'test_id'):
            return test_case.test_id
        
        if hasattr(test_case, 'id'):
            return test_case.id
        
        raise ValueError(f"Cannot get ID from TestCase object: {type(test_case)}")
    
    @staticmethod
    def get_category(test_case: Any) -> Optional[str]:
        if hasattr(test_case, 'category'):
            return test_case.category
        
        if hasattr(test_case, 'metadata') and isinstance(test_case.metadata, dict):
            return test_case.metadata.get('category')
        
        if hasattr(test_case, 'get_category'):
            return test_case.get_category()
        
        return None
    
    @staticmethod
    def get_questions(test_case: Any) -> List[Any]:
        if hasattr(test_case, 'question'):
            return test_case.question
        
        if hasattr(test_case, 'content') and test_case.content:
            if isinstance(test_case.content, dict) and 'question' in test_case.content:
                return test_case.content['question']
            elif isinstance(test_case.content, list):
                return test_case.content
        
        if hasattr(test_case, 'metadata') and isinstance(test_case.metadata, dict):
            return test_case.metadata.get('question', [])
        
        return []
    
    @staticmethod
    def get_functions(test_case: Any) -> List[Dict[str, Any]]:
        if hasattr(test_case, 'function'):
            return test_case.function
        
        if hasattr(test_case, 'content') and test_case.content:
            if isinstance(test_case.content, dict) and 'function' in test_case.content:
                return test_case.content['function']
        
        if hasattr(test_case, 'metadata') and isinstance(test_case.metadata, dict):
            return test_case.metadata.get('function', [])
        
        return []
    
    @staticmethod
    def get_initial_config(test_case: Any) -> Dict[str, Any]:
        if hasattr(test_case, 'initial_config') and test_case.initial_config:
            return test_case.initial_config
        
        if hasattr(test_case, 'metadata') and isinstance(test_case.metadata, dict):
            return test_case.metadata.get('initial_config', {})
        
        return {}
    
    @staticmethod
    def get_involved_classes(test_case: Any) -> List[str]:
        if hasattr(test_case, 'involved_classes') and test_case.involved_classes:
            return test_case.involved_classes
        
        if hasattr(test_case, 'metadata') and isinstance(test_case.metadata, dict):
            involved = test_case.metadata.get('involved_classes', [])
            if involved:
                return involved
            
            initial_config = test_case.metadata.get('initial_config', {})
            if initial_config and isinstance(initial_config, dict):
                return list(initial_config.keys())
        
        return []
    
    @staticmethod
    def is_multi_turn(test_case: Any) -> bool:
        test_id = TestCaseAdapter.get_id(test_case)
        if 'multi_turn' in test_id.lower():
            return True
        
        category = TestCaseAdapter.get_category(test_case)
        if category and 'multi_turn' in category.lower():
            return True
        
        questions = TestCaseAdapter.get_questions(test_case)
        if questions and isinstance(questions, list) and len(questions) > 0:
            if isinstance(questions[0], list):
                return True
        
        return False
    
    @staticmethod
    def convert_to_bfcl_format(test_case: Any) -> 'BFCLTestCase':
        from benchmarks.bfcl.data.loader import TestCase as BFCLTestCase
        
        if isinstance(test_case, BFCLTestCase):
            return test_case
        
        return BFCLTestCase(
            test_id=TestCaseAdapter.get_id(test_case),
            category=TestCaseAdapter.get_category(test_case) or "unknown",
            question=TestCaseAdapter.get_questions(test_case),
            function=TestCaseAdapter.get_functions(test_case),
            initial_config=TestCaseAdapter.get_initial_config(test_case),
            involved_classes=TestCaseAdapter.get_involved_classes(test_case),
            metadata=getattr(test_case, 'metadata', {})
        )
    
    @staticmethod
    def log_test_case_info(test_case: Any, prefix: str = ""):
        try:
            test_id = TestCaseAdapter.get_id(test_case)
            category = TestCaseAdapter.get_category(test_case)
            is_multi = TestCaseAdapter.is_multi_turn(test_case)
            
            logger.info(f"{prefix}TestCase Info:")
            logger.info(f"  - Type: {type(test_case).__name__}")
            logger.info(f"  - ID: {test_id}")
            logger.info(f"  - Category: {category}")
            logger.info(f"  - Multi-turn: {is_multi}")
            
            if is_multi:
                involved = TestCaseAdapter.get_involved_classes(test_case)
                logger.info(f"  - Involved classes: {involved}")
        except Exception as e:
            logger.error(f"{prefix}Failed to log TestCase info: {e}")


def get_test_id(test_case: Any) -> str:
    return TestCaseAdapter.get_id(test_case)


def is_multi_turn_test(test_case: Any) -> bool:
    return TestCaseAdapter.is_multi_turn(test_case)
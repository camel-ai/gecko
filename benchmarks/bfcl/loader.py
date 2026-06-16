
import os
import logging
from typing import Any, Dict, List, Optional

try:
    from .data.loader import BFCLDataLoader, TestCase as BFCLTestCase
except ImportError as exc:
    raise ImportError(
        "BFCL data loader is required but not available. "
        "Ensure benchmarks.bfcl.data is importable."
    ) from exc

logger = logging.getLogger(__name__)


class BFCLTestLoader:
    
    def __init__(self, 
                 data_dir: Optional[str] = None, 
                 ground_truth_dir: Optional[str] = None):
        self.data_dir = data_dir
        self.ground_truth_dir = ground_truth_dir
        
        self._loader = BFCLDataLoader(data_dir, ground_truth_dir)
    
    def list_categories(self) -> List[str]:
        return self._loader.list_categories()
    
    def list_test_ids(self, category: Optional[str] = None) -> List[str]:
        return self._loader.list_test_ids(category)
    
    def load_test_case(self, test_id: str) -> Any:
        return self._loader.load_test_case(test_id)
    
    def load_ground_truth(self, test_id: str) -> Optional[Any]:
        return self._loader.load_ground_truth(test_id)
    
    def test_case_exists(self, test_id: str) -> bool:
        try:
            self.load_test_case(test_id)
            return True
        except Exception:
            return False
    
    def get_test_case_info(self, test_id: str) -> Dict[str, Any]:
        if hasattr(self._loader, 'get_test_case_info'):
            return self._loader.get_test_case_info(test_id)
        raise AttributeError("BFCLDataLoader missing get_test_case_info implementation")
    
    def get_category_stats(self, category: str) -> Dict[str, Any]:
        if hasattr(self._loader, 'get_category_stats'):
            return self._loader.get_category_stats(category)
        raise AttributeError("BFCLDataLoader missing get_category_stats implementation")

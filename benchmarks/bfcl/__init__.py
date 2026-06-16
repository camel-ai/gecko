"""
BFCL Benchmark Plugin
BFCL (Berkeley Function Call Leaderboard) benchmark implementation
"""

from .benchmark import BFCLBenchmark
from .evaluation.evaluator import BFCLEvaluator
from .loader import BFCLTestLoader
from .utils import (
    compress_single_turn_function_name,
    derive_single_turn_endpoint_name,
    derive_single_turn_runtime_function_name,
    derive_single_turn_schema_name,
    extract_numeric_id,
    sanitize_single_turn_function_name,
    sort_bfcl_test_ids,
)

from .. import register_benchmark
register_benchmark("bfcl", BFCLBenchmark)

__all__ = [
    'BFCLBenchmark',
    'BFCLEvaluator',
    'BFCLTestLoader',
    'compress_single_turn_function_name',
    'derive_single_turn_endpoint_name',
    'derive_single_turn_runtime_function_name',
    'derive_single_turn_schema_name',
    'extract_numeric_id',
    'sanitize_single_turn_function_name',
    'sort_bfcl_test_ids',
]

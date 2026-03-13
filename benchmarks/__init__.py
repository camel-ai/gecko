from typing import Dict, Type, List
from .base.benchmark import BaseBenchmark

BENCHMARK_REGISTRY: Dict[str, Type[BaseBenchmark]] = {}


def register_benchmark(name: str, benchmark_class: Type[BaseBenchmark]):
    BENCHMARK_REGISTRY[name] = benchmark_class


def get_benchmark(benchmark_name: str, **config) -> BaseBenchmark:
    if benchmark_name not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {list_available_benchmarks()}")

    benchmark_class = BENCHMARK_REGISTRY[benchmark_name]
    return benchmark_class(**config)


def list_available_benchmarks() -> List[str]:
    return list(BENCHMARK_REGISTRY.keys())


try:
    from . import bfcl
except ImportError as e:
    import logging
    logging.warning(f"Failed to import BFCL benchmark: {e}")

# tau2 benchmark not included in this repository

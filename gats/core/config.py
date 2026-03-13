from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GATSConfig:
    # === Model ===
    model: str = "gpt-4o"

    # === Retry / Test-Time Scaling ===
    max_retries: int = 3
    target_score: float = 1.0

    # === Agent behavior ===
    agent_prompt: Optional[str] = None
    agent_max_iterations: int = 10
    agent_timeout: Optional[int] = None
    agent_persistence: bool = False
    enable_tool_filtering: bool = False

    # === Evaluation ===
    judge_prompt: Optional[str] = None
    checklist_prompt: Optional[str] = None
    base_checklist_items: Optional[List[str]] = None
    enable_checklist: bool = True
    include_agent_response_in_judge: bool = True
    enable_tool_result_folding: bool = True

    # === Gecko runtime ===
    gecko_url: str = "http://localhost:8000"
    override_openapi_servers: bool = True
    collect_gecko_usage: bool = True

    # === Execution ===
    max_workers: int = 1

    # === Debug ===
    debug: bool = False
    verbose: bool = False

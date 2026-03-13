from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GATSTask:
    """A single task for GATS to execute."""

    id: str
    turns: List[str]  # User messages (1 for single-turn, N for multi-turn)
    tool_schemas: List[str]  # OpenAPI spec file paths
    initial_config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Per-task agent prompt override (e.g., base prompt + task system messages).
    # If set, takes precedence over GATSConfig.agent_prompt for this task.
    agent_prompt: Optional[str] = None


@dataclass
class GATSAttempt:
    """Result of a single retry attempt within a turn."""

    index: int
    tool_calls: List[Dict[str, Any]]
    score: float
    feedback: Dict[str, Any]
    config_after: Dict[str, Any]
    agent_response: str
    execution_time: float


@dataclass
class GATSTurn:
    """Result of a single turn."""

    index: int
    question: str
    best_attempt: int
    score: float
    attempts: List[GATSAttempt]
    checklist: List[Dict[str, Any]]
    config_after: Dict[str, Any]
    execution_time: float


@dataclass
class GATSResult:
    """Complete result for a task."""

    task_id: str
    success: bool
    turns: List[GATSTurn]
    total_time: float
    events: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def final_score(self) -> float:
        return self.turns[-1].score if self.turns else 0.0

    @property
    def total_attempts(self) -> int:
        return sum(len(t.attempts) for t in self.turns)

    @property
    def all_tool_calls(self) -> List[List[Dict]]:
        """Per-turn best-attempt tool calls."""
        result = []
        for turn in self.turns:
            if 0 <= turn.best_attempt < len(turn.attempts):
                result.append(turn.attempts[turn.best_attempt].tool_calls)
            else:
                result.append([])
        return result

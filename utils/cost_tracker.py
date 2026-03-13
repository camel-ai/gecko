from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class LLMUsageCategory(Enum):
    """Categories of LLM usage in the system"""
    MAIN_INFERENCE = "main_inference"
    TASK_EVALUATOR_CHECKLIST = "task_evaluator_checklist"
    TASK_EVALUATOR_JUDGE = "task_evaluator_judge"
    MOCK_SERVER_RESPONSE = "mock_server_response"
    PARAM_VALIDATION = "param_validation"


@dataclass
class LLMUsage:
    """Single LLM usage record"""
    category: LLMUsageCategory
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "category": self.category.value,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "metadata": self.metadata
        }


@dataclass
class CostTracker:
    """Tracks all LLM costs for a test execution"""
    usage_records: List[LLMUsage] = field(default_factory=list)
    
    def add_usage(self, usage: LLMUsage):
        """Add a usage record"""
        self.usage_records.append(usage)
    
    def add_from_response(
        self, 
        response: Any,
        model: str,
        category: LLMUsageCategory,
        metadata: Dict[str, Any] = None
    ) -> float:
        """
        Add usage from a CAMEL response object and calculate cost
        
        Args:
            response: CAMEL response object with info attribute
            model: Model name for cost calculation
            category: Usage category
            metadata: Optional metadata
            
        Returns:
            Cost of this LLM call
        """
        from utils.model_utils import calculate_cost
        
        # Calculate cost
        cost = 0.0
        input_tokens = 0
        output_tokens = 0
        
        try:
            # Extract token usage from response
            if hasattr(response, 'info') and response.info:
                usage_info = response.info.get('usage', {})
                input_tokens = usage_info.get('prompt_tokens', 0)
                output_tokens = usage_info.get('completion_tokens', 0)
                
                # Calculate cost using the utility function
                cost = calculate_cost(response, model)
        except Exception as e:
            # Log but don't fail
            print(f"Warning: Failed to extract cost from response: {e}")
        
        # Add usage record
        usage = LLMUsage(
            category=category,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            metadata=metadata or {}
        )
        self.add_usage(usage)
        
        return cost
    
    def get_total_cost(self) -> float:
        """Get total cost across all categories"""
        return sum(usage.cost for usage in self.usage_records)
    
    def get_cost_by_category(self) -> Dict[str, float]:
        """Get cost breakdown by category"""
        costs = {}
        for category in LLMUsageCategory:
            category_cost = sum(
                usage.cost 
                for usage in self.usage_records 
                if usage.category == category
            )
            if category_cost > 0:
                costs[category.value] = category_cost
        return costs
    
    def get_usage_by_category(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed usage breakdown by category"""
        usage_stats = {}
        for category in LLMUsageCategory:
            category_records = [
                usage for usage in self.usage_records 
                if usage.category == category
            ]
            if category_records:
                usage_stats[category.value] = {
                    "count": len(category_records),
                    "total_input_tokens": sum(u.input_tokens for u in category_records),
                    "total_output_tokens": sum(u.output_tokens for u in category_records),
                    "total_cost": sum(u.cost for u in category_records)
                }
        return usage_stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "total_cost": self.get_total_cost(),
            "cost_by_category": self.get_cost_by_category(),
            "usage_by_category": self.get_usage_by_category(),
            "detailed_records": [usage.to_dict() for usage in self.usage_records]
        }
    
    def merge(self, other: 'CostTracker'):
        """Merge another cost tracker into this one"""
        self.usage_records.extend(other.usage_records)
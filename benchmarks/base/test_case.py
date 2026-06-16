
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TestCase:
    
    id: str
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    content: Any = None
    
    expected_outputs: Optional[List[Any]] = None
    
    evaluation_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            raise ValueError("Test case id cannot be empty")
        
        if not isinstance(self.metadata, dict):
            raise ValueError("Metadata must be a dictionary")
    
    def get_category(self) -> Optional[str]:
        return self.metadata.get('category')
    
    def get_difficulty(self) -> Optional[str]:
        return self.metadata.get('difficulty')
    
    def get_tags(self) -> List[str]:
        return self.metadata.get('tags', [])
    
    def add_tag(self, tag: str):
        if 'tags' not in self.metadata:
            self.metadata['tags'] = []
        if tag not in self.metadata['tags']:
            self.metadata['tags'].append(tag)


import logging
from typing import Any, Dict, List, Optional, Type, Union
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AgentFactory:
    
    _registered_agents: Dict[str, Type[BaseAgent]] = {}
    
    @classmethod
    def register_agent(cls, name: str, agent_class: Type[BaseAgent]):
        cls._registered_agents[name] = agent_class
        logger.info(f"Registered agent type: {name}")
    
    @classmethod
    def create_agent(cls, agent_type: str = "chat", **kwargs) -> BaseAgent:
        if agent_type not in cls._registered_agents:
            cls._load_default_agents()
        
        if agent_type not in cls._registered_agents:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = cls._registered_agents[agent_type]
        return agent_class(**kwargs)
    
    @classmethod
    def _load_default_agents(cls):
        try:
            from .chat_agent import ChatAgent
            cls.register_agent("chat", ChatAgent)
        except ImportError as e:
            logger.warning(f"Failed to load ChatAgent: {e}")
    
    @classmethod
    def list_available_agents(cls) -> List[str]:
        return list(cls._registered_agents.keys())


def create_agent(agent_type: str = "chat", 
                model_name: str = "gpt-4.1-mini",
                timeout: int = 300,
                system_message: Optional[str] = None,
                **kwargs) -> BaseAgent:
    return AgentFactory.create_agent(
        agent_type=agent_type,
        model_name=model_name,
        timeout=timeout,
        system_message=system_message,
        **kwargs
    )

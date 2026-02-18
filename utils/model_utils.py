"""Model utilities for CAMEL backend creation."""

import json
import logging
from typing import Any, Dict, List, Optional

from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.toolkits import FunctionTool

logger = logging.getLogger(__name__)


def create_model(
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 16384,
    tools: List[FunctionTool] = None,
):
    """Create a CAMEL model backend from a supported model alias."""
    model_configs = {
        "gpt-4o": {
            "platform": ModelPlatformType.OPENAI,
            "type": ModelType.GPT_4O,
        },
        "gpt-4o-mini": {
            "platform": ModelPlatformType.OPENAI,
            "type": ModelType.GPT_4O_MINI,
        },
        "gpt-4.1-mini": {
            "platform": ModelPlatformType.OPENAI,
            "type": ModelType.GPT_4_1_MINI,
        },
        "gpt-4.1-nano": {
            "platform": ModelPlatformType.OPENAI,
            "type": ModelType.GPT_4_1_NANO,
        },
        "gpt-5": {
            "platform": ModelPlatformType.OPENAI,
            "type": ModelType.GPT_5,
            "allow_max_tokens": False,
            "allow_temperature": False,
        },
        "gpt-5-mini": {
            "platform": ModelPlatformType.OPENAI,
            "type": ModelType.GPT_5_MINI,
            "allow_max_tokens": False,
            "allow_temperature": False,
        },
        "gpt-5-nano": {
            "platform": ModelPlatformType.OPENAI,
            "type": ModelType.GPT_5_NANO,
            "allow_max_tokens": False,
            "allow_temperature": False,
        },
        "gpt-5-thinking": {
            "platform": ModelPlatformType.OPENAI,
            "type": ModelType.GPT_5,
            "allow_max_tokens": False,
            "allow_temperature": False,
            "extra_config": {"reasoning_effort": "medium"},
        },
        "gpt-5-thinking-pro": {
            "platform": ModelPlatformType.OPENAI,
            "type": ModelType.GPT_5,
            "allow_max_tokens": False,
            "allow_temperature": False,
            "extra_config": {"reasoning_effort": "high"},
        },
    }

    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}")

    config = model_configs[model_name]
    config_dict = ChatGPTConfig(
        temperature=temperature if config.get("allow_temperature", True) else None,
        max_tokens=max_tokens if config.get("allow_max_tokens", True) else None,
        tools=tools,
        **(config.get("extra_config", {})),
    ).as_dict()

    return ModelFactory.create(
        model_platform=config["platform"],
        model_type=config["type"],
        model_config_dict=config_dict,
    )


def load_json(json_str, default_value={}, verbose=True):
    """
    Enhanced JSON loading function that handles various JSON format errors.
    
    Args:
        json_str (str): JSON string to parse
        default_value (Any, optional): Default value to return if parsing fails. Defaults to {}
        verbose (bool, optional): Whether to print detailed error messages. Defaults to True
    
    Returns:
        Any: Parsed JSON object, or default_value if parsing fails
    """
    if not json_str:
        if verbose:
            logger.warning("Empty JSON string provided")
        return default_value
    
    # First try standard json parsing
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        if verbose:
            logger.debug(f"Standard JSON parsing failed: {e}")
    
    # Try various JSON repair strategies
    try:
        # Try json5 which is more lenient
        import json5
        return json5.loads(json_str)
    except Exception as e:
        if verbose:
            logger.debug(f"JSON5 parsing failed: {e}")
    
    # Try demjson3 for even more lenient parsing
    try:
        import demjson3
        return demjson3.decode(json_str, strict=False)
    except Exception as e:
        if verbose:
            logger.debug(f"demjson3 parsing failed: {e}")
    
    # Try json_repair for fixing common JSON issues
    try:
        import json_repair
        repaired = json_repair.repair_json(json_str)
        return json.loads(repaired)
    except Exception as e:
        if verbose:
            logger.debug(f"JSON repair failed: {e}")
    
    # If all else fails, return default value
    if verbose:
        logger.warning(f"All JSON parsing attempts failed, returning default value")
    return default_value

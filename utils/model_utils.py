import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.toolkits import FunctionTool
from camel.configs import ChatGPTConfig

logger = logging.getLogger(__name__)


def strip_thinking_content(text: str) -> str:
    """Strip model reasoning blocks like <think>...</think> before JSON parsing."""
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()
    if not cleaned:
        return ""

    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()
    elif cleaned.lstrip().startswith("<think>"):
        cleaned = re.sub(r"^\s*<think>\s*", "", cleaned, count=1, flags=re.IGNORECASE).strip()

    return cleaned


def strip_code_fences(text: str) -> str:
    """Remove outer markdown code fences (``` / ```json) if present."""
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()
    if not cleaned:
        return ""

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return cleaned


def sanitize_llm_json_text(text: str) -> str:
    """Normalize LLM output for JSON parsing: remove think blocks then code fences."""
    return strip_code_fences(strip_thinking_content(text))


def create_model(
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    tools: List[FunctionTool] = None,
    timeout: Optional[float] = None,
    max_retries: int = 3,
):
    """
    Create a model instance
    
    Args:
        model_name: Model name
        temperature: Temperature parameter
        max_tokens: Maximum token count
        tools: List of tools
        timeout: Per-request timeout in seconds for model backend
        max_retries: Maximum retries for backend requests
        
    Returns:
        Created model instance
    """
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
        "gpt-5": {
            "platform": ModelPlatformType.OPENAI,
            "type": ModelType.GPT_5,  # Maps to "gpt-5" (instant version)
            "allow_max_tokens": False,
            "allow_temperature": False,  # Only supports temperature=1.0
        },
        "gpt-5-mini": {
            "platform": ModelPlatformType.OPENAI,
            "type": ModelType.GPT_5_MINI,  # Maps to "gpt-5-mini"
            "allow_max_tokens": False,
            "allow_temperature": False,
        },
        "gpt-5-nano": {
            "platform": ModelPlatformType.OPENAI,
            "type": ModelType.GPT_5_NANO,  # Maps to "gpt-5-nano"
            "allow_max_tokens": False,
            "allow_temperature": False,
        },
        "gpt-5-thinking": {
            "platform": ModelPlatformType.OPENAI,
            "type": ModelType.GPT_5,  # Map to base GPT-5
            "allow_max_tokens": False,
            "allow_temperature": False,
            "extra_config": {"reasoning_effort": "medium"},
        },
        "deepinfra-qwen3-14b": {
            "platform": ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            "type": "Qwen/Qwen3-14B",
            "url": "https://api.deepinfra.com/v1/openai",
            "api_key": "DEEPINFRA_API_KEY",
        }
    }


    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}")

    config = model_configs[model_name]
    
    extra_config = dict(config.get("extra_config", {}) or {})
    effective_max_tokens = max_tokens
    model_max_tokens_cap = config.get("max_tokens_cap")
    if isinstance(model_max_tokens_cap, int) and model_max_tokens_cap > 0:
        effective_max_tokens = min(max_tokens, model_max_tokens_cap)

    base_runtime_config: Dict[str, Any] = {
        "temperature": temperature if config.get("allow_temperature", True) else None,
        "max_tokens": effective_max_tokens if config.get("allow_max_tokens", True) else None,
        "tools": tools,
    }
    base_runtime_config.update(extra_config)

    # OpenAI-compatible providers often require provider-specific request fields.
    if config["platform"] == ModelPlatformType.OPENAI_COMPATIBLE_MODEL:
        config_dict = {
            k: v
            for k, v in base_runtime_config.items()
            if v is not None
        }
    else:
        allowed_fields = set(ChatGPTConfig.model_fields.keys())
        known_config: Dict[str, Any] = {}
        dropped_keys: List[str] = []
        for k, v in base_runtime_config.items():
            if v is None:
                continue
            if k in allowed_fields:
                known_config[k] = v
            else:
                dropped_keys.append(k)

        if dropped_keys:
            logger.warning(
                "Dropping unsupported model config fields for %s: %s",
                model_name,
                sorted(dropped_keys),
            )

        config_dict = ChatGPTConfig(**known_config).as_dict()

    model_args = {
        "model_platform": config["platform"],
        "model_type": config["type"],
        "model_config_dict": config_dict,
        "timeout": timeout,
        "max_retries": max_retries,
    }

    if "url" in config:
        model_args["url"] = config["url"]
    if "api_key" in config:
        api_key_val = os.getenv(config["api_key"])
        model_args["api_key"] = api_key_val

    model = ModelFactory.create(**model_args)

    # Some CAMEL backends may override retry config during initialization.
    # Force-sync timeout/retry settings to keep per-call timeout behavior predictable.
    try:
        if hasattr(model, "_timeout") and timeout is not None:
            setattr(model, "_timeout", timeout)
        if hasattr(model, "_max_retries"):
            setattr(model, "_max_retries", max_retries)

        for client_attr in ("_client", "_async_client"):
            client = getattr(model, client_attr, None)
            if client is None:
                continue
            if timeout is not None:
                for timeout_attr in ("timeout", "_timeout"):
                    try:
                        setattr(client, timeout_attr, timeout)
                    except Exception:
                        pass
            for retry_attr in ("max_retries", "_max_retries"):
                try:
                    setattr(client, retry_attr, max_retries)
                except Exception:
                    pass
    except Exception as exc:
        logger.debug(f"Failed to enforce timeout/retry on model backend: {exc}")

    return model


def calculate_cost(response, model_name="gpt-4o"):
    """
    Calculate the cost of a model response based on token usage.
    
    Args:
        response: The model response object containing token usage info
        model_name: The name of the model used
        
    Returns:
        float: The calculated cost in dollars
    """
    if model_name == "gpt-4o":
        input_cost = 2.5
        output_cost = 10
    elif model_name == "gpt-4o-mini":
        input_cost = 0.15
        output_cost = 0.6
    elif model_name == "gpt-4.1-mini":
        input_cost = 0.4
        output_cost = 1.6
    elif model_name == "gpt-4.1-nano":
        input_cost = 0.1
        output_cost = 0.4
    elif model_name == "gpt-5":
        input_cost = 1.25
        output_cost = 10
    elif model_name == "gpt-5-mini":
        input_cost = 0.25
        output_cost = 2
    elif model_name == "gpt-5-nano":
        input_cost = 0.05
        output_cost = 0.4
    elif model_name == "deepinfra-qwen3-14b":
        input_cost = 0.06
        output_cost = 0.24
    else:
        input_cost = 0.0
        output_cost = 0.0
    
    try:
        # Get token counts from response
        if hasattr(response, 'info') and response.info and 'usage' in response.info:
            usage = response.info['usage']
            # Note: In the original code, these are swapped!
            input_tokens = usage.get('completion_tokens', 0) / 1000000
            output_tokens = usage.get('prompt_tokens', 0) / 1000000
        elif hasattr(response, 'usage'):
            # Alternative format
            usage = response.usage
            input_tokens = getattr(usage, 'completion_tokens', 0) / 1000000
            output_tokens = getattr(usage, 'prompt_tokens', 0) / 1000000
        else:
            logger.warning(f"No usage information found in response for cost calculation")
            return 0.0
        
        total_cost = input_cost * input_tokens + output_cost * output_tokens
        return total_cost
        
    except Exception as e:
        logger.error(f"Error calculating cost: {e}")
        return 0.0


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

    json_str = sanitize_llm_json_text(json_str)
    
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

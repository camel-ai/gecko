
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ConfigManager:
    
    def __init__(self, config_dir: Union[str, Path] = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._config_cache = {}
    
    def load_config(self, 
                   config_name: str, 
                   default_config: Optional[Dict[str, Any]] = None,
                   use_cache: bool = True) -> Dict[str, Any]:
        if use_cache and config_name in self._config_cache:
            return self._config_cache[config_name].copy()
        
        config_file = self.config_dir / f"{config_name}.json"
        config = default_config.copy() if default_config else {}
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    config.update(file_config)
                    logger.debug(f"Loaded config from {config_file}")
            except Exception as e:
                logger.error(f"Failed to load config from {config_file}: {e}")
        
        config = self._apply_env_overrides(config, config_name)
        
        if use_cache:
            self._config_cache[config_name] = config.copy()
        
        return config
    
    def save_config(self, config_name: str, config: Dict[str, Any]):
        config_file = self.config_dir / f"{config_name}.json"
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self._config_cache[config_name] = config.copy()
            logger.info(f"Saved config to {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {config_file}: {e}")
            raise
    
    def _apply_env_overrides(self, config: Dict[str, Any], config_name: str) -> Dict[str, Any]:
        prefix = f"{config_name.upper()}_"
        
        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                config_key = env_key[len(prefix):].lower()
                
                converted_value = self._convert_env_value(env_value)
                config[config_key] = converted_value
                
                logger.debug(f"Applied env override: {config_key} = {converted_value}")
        
        return config
    
    def _convert_env_value(self, value: str) -> Any:
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
            return False
        
        try:
            if '.' not in value:
                return int(value)
            else:
                return float(value)
        except ValueError:
            pass
        
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        return value
    
    def get_nested_value(self, config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_nested_value(self, config: Dict[str, Any], key_path: str, value: Any):
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        merged = {}
        
        for config in configs:
            merged = self._deep_merge(merged, config)
        
        return merged
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = base.copy()
        
        for key, value in override.items():
            if (key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def validate_config(self, 
                       config: Dict[str, Any], 
                       schema: Dict[str, Any]) -> List[str]:
        errors = []
        
        for key, spec in schema.items():
            if isinstance(spec, tuple):
                expected_type, required = spec
            else:
                expected_type, required = spec, False
            
            if required and key not in config:
                errors.append(f"Missing required field: {key}")
                continue
            
            if key in config:
                value = config[key]
                if not isinstance(value, expected_type):
                    errors.append(f"Field '{key}' should be {expected_type.__name__}, got {type(value).__name__}")
        
        return errors
    
    def create_default_config(self, config_name: str, default_values: Dict[str, Any]):
        config_file = self.config_dir / f"{config_name}.json"
        
        if not config_file.exists():
            self.save_config(config_name, default_values)
            logger.info(f"Created default config: {config_file}")
    
    def list_config_files(self) -> List[str]:
        config_files = []
        for file_path in self.config_dir.glob("*.json"):
            config_files.append(file_path.stem)
        return sorted(config_files)
    
    def clear_cache(self):
        self._config_cache.clear()
        logger.debug("Config cache cleared")
    
    def export_config(self, config_name: str, output_path: Union[str, Path]):
        config = self.load_config(config_name, use_cache=False)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported config '{config_name}' to {output_path}")
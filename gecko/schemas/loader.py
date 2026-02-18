import json
import os
from typing import Any, Dict, Optional, List, Union

import yaml
from openapi_spec_validator import validate_spec


class SchemaLoader:
    def __init__(self, schemas_dir: Union[str, List[str]]):
        # Support single directory or a list of directories
        if isinstance(schemas_dir, list):
            self.schema_dirs: List[str] = [d for d in schemas_dir if isinstance(d, str) and d]
        else:
            self.schema_dirs = [schemas_dir]
        self.schemas_cache: Dict[str, Dict[str, Any]] = {}

    def find_schema_file(self, api_name: str) -> Optional[str]:
        """Find the schema file for the given API name.
        Supports .json/.yaml/.yml, case-insensitive filenames, nested '<api_name>/openapi.json',
        and recursive search in subdirectories.
        """
        for base in self.schema_dirs:
            # Direct candidates in base directory
            candidates = [
                os.path.join(base, f"{api_name}.json"),
                os.path.join(base, f"{api_name}.yaml"),
                os.path.join(base, f"{api_name}.yml"),
                os.path.join(base, api_name, "openapi.json"),
                os.path.join(base, api_name, "openapi.yaml"),
                os.path.join(base, api_name, "openapi.yml"),
            ]
            for path in candidates:
                if os.path.exists(path):
                    return path
            
            # Case-insensitive scan in base (flat files)
            try:
                for entry in os.listdir(base):
                    lower = entry.lower()
                    if lower in {f"{api_name}.json", f"{api_name}.yaml", f"{api_name}.yml"}:
                        p = os.path.join(base, entry)
                        if os.path.exists(p):
                            return p
            except Exception:
                pass
            
            # Recursive search in subdirectories
            result = self._recursive_search(base, api_name)
            if result:
                return result
                
        return None

    def _recursive_search(self, directory: str, api_name: str) -> Optional[str]:
        """Recursively search for schema files in subdirectories."""
        try:
            for entry in os.listdir(directory):
                entry_path = os.path.join(directory, entry)
                if os.path.isdir(entry_path):
                    # Check if this subdirectory contains the schema
                    candidates = [
                        os.path.join(entry_path, f"{api_name}.json"),
                        os.path.join(entry_path, f"{api_name}.yaml"),
                        os.path.join(entry_path, f"{api_name}.yml"),
                    ]
                    for candidate in candidates:
                        if os.path.exists(candidate):
                            return candidate
                    
                    # Case-insensitive search in this subdirectory
                    try:
                        for sub_entry in os.listdir(entry_path):
                            lower = sub_entry.lower()
                            if lower in {f"{api_name}.json", f"{api_name}.yaml", f"{api_name}.yml"}:
                                p = os.path.join(entry_path, sub_entry)
                                if os.path.exists(p):
                                    return p
                    except Exception:
                        pass
                    
                    # Continue recursive search
                    result = self._recursive_search(entry_path, api_name)
                    if result:
                        return result
        except Exception:
            pass
        return None

    def load_schema(self, schema_path: str) -> Dict[str, Any]:
        """Load and validate OpenAPI schema."""
        if schema_path in self.schemas_cache:
            return self.schemas_cache[schema_path]

        with open(schema_path, 'r', encoding='utf-8') as f:
            if schema_path.endswith('.yaml') or schema_path.endswith('.yml'):
                schema = yaml.safe_load(f)
            else:
                schema = json.load(f)
        
        # Validate the schema
        validate_spec(schema)
        self.schemas_cache[schema_path] = schema
        return schema 
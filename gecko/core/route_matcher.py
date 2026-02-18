import logging
from typing import Any, Dict, Optional, Tuple

from fastapi import Request

logger = logging.getLogger(__name__)


class RouteMatcher:
    """Handles route matching and API info extraction."""
    
    def __init__(self, schema_loader):
        """Initialize the route matcher with a schema loader."""
        self.schema_loader = schema_loader
    
    def extract_api_info(self, request: Request) -> Tuple[Optional[str], Optional[str]]:
        """Extract API name and remaining path from the request.

        Handles two path formats:
        1. Two-segment: /api_name/endpoint (original format)
        2. Single-segment: /endpoint (searches all schemas for matching path)
        """
        path = request.url.path
        # Normalize incoming path: ensure leading slash, strip single trailing slash
        if not path.startswith('/'):
            path = '/' + path
        if len(path) > 1 and path.endswith('/'):
            path = path[:-1]

        parts = path[1:].split('/')  # skip leading '/'

        # Two-segment format: /api_name/endpoint
        if len(parts) >= 2:
            api_name = parts[0]
            remaining_path = '/' + '/'.join(parts[1:])
            # Normalize remaining_path trailing slash
            if len(remaining_path) > 1 and remaining_path.endswith('/'):
                remaining_path = remaining_path[:-1]
            return api_name, remaining_path

        # Single-segment format: /endpoint
        # Search through all schemas to find which one contains this path
        if len(parts) == 1:
            api_name = self._find_schema_containing_path(path, request.method.lower())
            if api_name:
                return api_name, path

        return None, None

    def _find_schema_containing_path(self, path: str, method: str) -> Optional[str]:
        """Search all schemas to find which one contains the given path.

        Args:
            path: The request path (e.g., '/book_reservation')
            method: HTTP method (e.g., 'post')

        Returns:
            API name (schema title or filename) if found, None otherwise
        """
        import os

        # Get all schema files from the schema directories
        schema_files = []
        for schema_dir in self.schema_loader.schema_dirs:
            if os.path.exists(schema_dir) and os.path.isdir(schema_dir):
                for filename in os.listdir(schema_dir):
                    if filename.endswith(('.json', '.yaml', '.yml')):
                        schema_files.append(os.path.join(schema_dir, filename))

        # Search each schema for the path
        for schema_file in schema_files:
            try:
                schema = self.schema_loader.load_schema(schema_file)
                paths = schema.get('paths', {})

                # Check if this schema contains the path with the right method
                if path in paths:
                    path_item = paths[path]
                    if method in path_item:
                        # Found matching path and method
                        # Extract API name from schema info.title or filename
                        api_name = schema.get('info', {}).get('title')
                        if not api_name:
                            # Fallback to filename without extension
                            api_name = os.path.splitext(os.path.basename(schema_file))[0]
                        return api_name
            except Exception:
                # Skip invalid schemas
                continue

        return None

    def _pick_any_operation(self, path_template: str, paths: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Pick the first available operation for the given path template, ignoring HTTP method."""
        path_item = paths.get(path_template)
        if not path_item:
            return None, None
        for method_key, operation in path_item.items():
            if method_key.lower() in {"get", "post", "put", "delete", "patch", "options", "head"}:
                return path_template, operation
        return None, None

    def _find_matching_path_template(self, path: str, paths: Dict[str, Any]) -> Optional[str]:
        """Find a schema path template that matches the given request path, ignoring HTTP method."""
        import re
        for path_template, path_item in paths.items():
            # Convert path template to regex pattern
            path_pattern = path_template
            if not path_pattern.startswith('/'):
                path_pattern = '/' + path_pattern
            # Strip trailing slash in template
            if len(path_pattern) > 1 and path_pattern.endswith('/'):
                path_pattern = path_pattern[:-1]
            # Replace path params
            path_pattern = path_pattern.replace('{', '(?P<').replace('}', '>[^/]+)')
            pattern = re.compile(f'^{path_pattern}$')
            if pattern.match(path):
                return path_template
        return None

    def find_matching_endpoint(self, path: str, schema: Dict[str, Any], method: str, api_name: Optional[str] = None) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Find the matching endpoint in the schema with normalization fallback."""
        # Normalize incoming path
        if len(path) > 1 and path.endswith('/'):
            path = path[:-1]
        if not path.startswith('/'):
            path = '/' + path
        
        paths = schema.get('paths', {})
        
        # 1) Direct match against original schema keys (ignore method first)
        match_template = self._find_matching_path_template(path, paths)
        if match_template:
            operation = paths.get(match_template, {}).get(method)
            if operation:
                return match_template, operation
            any_path, any_op = self._pick_any_operation(match_template, paths)
            if any_path and any_op:
                return any_path, any_op
        
        normalized_paths: Dict[str, Any] = {}
        normalized_to_original: Dict[str, str] = {}
        
        # 2) Fallback A: Strip exact api_name prefix from schema keys
        if api_name:
            prefix = f"/{api_name}"
            needs_normalize = any(k.startswith(prefix + '/') or k == prefix or k.rstrip('/').startswith(prefix + '/') for k in paths.keys())
            if needs_normalize:
                for original_key, v in paths.items():
                    key = original_key
                    if not key.startswith('/'):
                        key = '/' + key
                    if len(key) > 1 and key.endswith('/'):
                        key = key[:-1]
                    if key.startswith(prefix + '/'):
                        nk = key[len(prefix):]
                        if not nk.startswith('/'):
                            nk = '/' + nk
                    elif key == prefix:
                        nk = '/'
                    else:
                        nk = key
                    normalized_paths[nk] = v
                    normalized_to_original[nk] = original_key
                match_template = self._find_matching_path_template(path, normalized_paths)
                if match_template:
                    operation = normalized_paths.get(match_template, {}).get(method)
                    if operation:
                        return normalized_to_original.get(match_template, match_template), operation
                    any_path, any_op = self._pick_any_operation(match_template, normalized_paths)
                    if any_path and any_op:
                        return normalized_to_original.get(any_path, any_path), any_op
        
        # 2) Fallback B: Case-insensitive api_name segment stripping from schema keys
        if api_name:
            norm_ci_paths: Dict[str, Any] = {}
            ci_map: Dict[str, str] = {}
            for original_key, v in paths.items():
                key = original_key
                if not key.startswith('/'):
                    key = '/' + key
                if len(key) > 1 and key.endswith('/'):
                    key = key[:-1]
                segments = [seg for seg in key.split('/') if seg != '']
                if len(segments) >= 1 and segments[0].lower() == api_name.lower():
                    stripped = '/' + '/'.join(segments[1:]) if len(segments) > 1 else '/'
                else:
                    stripped = key
                norm_ci_paths[stripped] = v
                ci_map[stripped] = original_key
            match_template = self._find_matching_path_template(path, norm_ci_paths)
            if match_template:
                operation = norm_ci_paths.get(match_template, {}).get(method)
                if operation:
                    return ci_map.get(match_template, match_template), operation
                any_path, any_op = self._pick_any_operation(match_template, norm_ci_paths)
                if any_path and any_op:
                    return ci_map.get(any_path, any_path), any_op
        
        # 3) Heuristic fallback: match by last path segment (function name)
        requested_last = path.rsplit('/', 1)[-1]
        # try original keys
        for original_key in paths.keys():
            key_norm = original_key
            if not key_norm.startswith('/'):
                key_norm = '/' + key_norm
            if len(key_norm) > 1 and key_norm.endswith('/'):
                key_norm = key_norm[:-1]
            last_seg = key_norm.rsplit('/', 1)[-1]
            if last_seg == requested_last:
                any_path, any_op = self._pick_any_operation(original_key, paths)
                if any_path and any_op:
                    logger.debug(
                        "Heuristic match by last segment: requested=%s schema_key=%s",
                        path,
                        original_key,
                    )
                    return original_key, any_op
        # try normalized keys if we built them
        for norm_key, original_key in normalized_to_original.items() if 'normalized_to_original' in locals() else []:
            key_norm = norm_key
            if len(key_norm) > 1 and key_norm.endswith('/'):
                key_norm = key_norm[:-1]
            last_seg = key_norm.rsplit('/', 1)[-1]
            if last_seg == requested_last:
                any_path, any_op = self._pick_any_operation(norm_key, normalized_paths)
                if any_path and any_op:
                    logger.debug(
                        "Heuristic match by normalized segment: requested=%s schema_key=%s",
                        path,
                        original_key,
                    )
                    return original_key, any_op
        
        # 4) Fallback C: search by operationId equals requested function name (case-insensitive)
        requested_fn = requested_last.lower()
        for original_key, path_item in paths.items():
            for method_key, operation in path_item.items():
                if method_key.lower() in {"get", "post", "put", "delete", "patch", "options", "head"}:
                    op_id = str(operation.get("operationId", "")).lower()
                    if op_id == requested_fn:
                        logger.debug(
                            "Match by operationId: requested=%s schema_key=%s method=%s",
                            path,
                            original_key,
                            method_key,
                        )
                        return original_key, operation
        
        logger.debug(
            "No endpoint matched: api_name=%s path=%s method=%s available_paths=%s",
            api_name,
            path,
            method,
            list(paths.keys()),
        )
        return None, None
    
    def load_api_schema(self, api_name: str) -> Optional[Dict[str, Any]]:
        """Load the API schema for the given API name.

        Supports two lookup methods:
        1. By filename (e.g., 'airline_write' matches 'airline_write.json')
        2. By title in schema (e.g., 'AirlineWriteAPI' matches any schema with that title)
        """
        import os

        # Method 1: Try filename-based lookup first (original behavior)
        schema_path = self.schema_loader.find_schema_file(api_name)
        if schema_path:
            return self.schema_loader.load_schema(schema_path)

        # Method 2: Search all schemas for matching title
        for schema_dir in self.schema_loader.schema_dirs:
            if not os.path.exists(schema_dir) or not os.path.isdir(schema_dir):
                continue

            for filename in os.listdir(schema_dir):
                if not filename.endswith(('.json', '.yaml', '.yml')):
                    continue

                try:
                    schema_file = os.path.join(schema_dir, filename)
                    schema = self.schema_loader.load_schema(schema_file)
                    schema_title = schema.get('info', {}).get('title', '')

                    # Match by title (case-insensitive for flexibility)
                    if schema_title.lower() == api_name.lower():
                        return schema
                except Exception:
                    continue

        return None 

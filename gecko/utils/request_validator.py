import logging
from typing import Any, Dict, Optional, Tuple

from fastapi import Request
from camel.agents import ChatAgent
from pydantic import BaseModel, Field
from typing import List
import copy
import json
import re
from .global_config import get_agent_timeout
class RequestValidator:
    """Validates requests against OpenAPI schemas using openapi_core."""
    
    def __init__(self, schema: Dict[str, Any], validation_model: str = "gpt-4.1-mini"):
        """Initialize the validator with an OpenAPI schema.
        Args:
            schema: OpenAPI schema dict
            validation_model: LLM model for request validation (default: gpt-4.1-mini)
        """
        self.schema = schema
        self.validation_model = validation_model

    _SEMANTIC_STRIP_KEYS = {
        "enum",
        "const",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "additionalProperties",
        "required",
    }

    def _check_numeric_bounds(self, value: float, schema: Dict[str, Any], display_context: str) -> Optional[Dict[str, str]]:
        if "minimum" in schema:
            if value < schema["minimum"]:
                return {"error_message": f"{display_context} should be >= {schema['minimum']}"}
        if "exclusiveMinimum" in schema:
            if value <= schema["exclusiveMinimum"]:
                return {"error_message": f"{display_context} should be > {schema['exclusiveMinimum']}"}
        if "maximum" in schema:
            if value > schema["maximum"]:
                return {"error_message": f"{display_context} should be <= {schema['maximum']}"}
        if "exclusiveMaximum" in schema:
            if value >= schema["exclusiveMaximum"]:
                return {"error_message": f"{display_context} should be < {schema['exclusiveMaximum']}"}
        if "multipleOf" in schema:
            step = schema["multipleOf"]
            try:
                # Allow small epsilon for float
                if abs((value / step) - round(value / step)) > 1e-9:
                    return {"error_message": f"{display_context} should be a multiple of {step}"}
            except Exception:
                return {"error_message": f"{display_context} should be a multiple of {step}"}
        return None

    def _check_string_rules(self, value: str, schema: Dict[str, Any], display_context: str) -> Optional[Dict[str, str]]:
        if "minLength" in schema and len(value) < schema["minLength"]:
            return {"error_message": f"{display_context} length should be >= {schema['minLength']}"}
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return {"error_message": f"{display_context} length should be <= {schema['maxLength']}"}
        if "pattern" in schema:
            pattern = schema["pattern"]
            try:
                if not re.fullmatch(pattern, value):
                    return {"error_message": f"{display_context} does not match required pattern"}
            except re.error:
                # If pattern invalid, skip regex check
                pass
        # simple format checks (best-effort)
        fmt = schema.get("format")
        if fmt == "date":
            # YYYY-MM-DD
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                return {"error_message": f"{display_context} should be a date in YYYY-MM-DD format"}
        elif fmt == "date-time":
            # very loose ISO 8601 check
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?(Z|[+-]\d{2}:\d{2})?", value):
                return {"error_message": f"{display_context} should be a date-time in ISO 8601 format"}
        elif fmt == "email":
            if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
                return {"error_message": f"{display_context} should be a valid email"}
        return None

    def validate_schema_value(self, value: Any, schema: Dict[str, Any], context: str) -> Optional[Dict[str, str]]:
        # Simplify context for better agent understanding - remove technical prefixes
        display_context = context
        if display_context.startswith("Request body."):
            display_context = display_context[13:]  # Remove "Request body."
        elif display_context == "Request body":
            display_context = "request data"
        expected_type = schema.get("type")
        expected_format = schema.get("format")
        enum = schema.get("enum")

        # Basic type checks + extended rule checks
        if expected_type == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return {"error_message": f"{display_context} should be a number, not {type(value).__name__}"}
            if isinstance(value, int) and expected_format in ("float", "double"):
                return {"error_message": f"{display_context} should be a float (e.g., {float(value)}), not an integer ({value})"}
            bounds_err = self._check_numeric_bounds(float(value), schema, display_context)
            if bounds_err:
                return bounds_err

        elif expected_type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                return {"error_message": f"{display_context} should be an integer, not {type(value).__name__}"}
            bounds_err = self._check_numeric_bounds(int(value), schema, display_context)
            if bounds_err:
                return bounds_err

        elif expected_type == "string":
            if not isinstance(value, str):
                return {"error_message": f"{display_context} should be a string, not {type(value).__name__}"}
            if enum and value not in enum:
                return {"error_message": f"{display_context} must be one of {enum}, got '{value}'"}
            fmt_err = self._check_string_rules(value, schema, display_context)
            if fmt_err:
                return fmt_err

        elif expected_type == "boolean":
            if not isinstance(value, bool):
                return {"error_message": f"{display_context} should be a boolean, not {type(value).__name__}"}

        elif expected_type == "array":
            if not isinstance(value, list):
                return {"error_message": f"{display_context} should be an array, not {type(value).__name__}. Expected: array (e.g., [1, 2, 3]), Got: {value}"}
            if "minItems" in schema and len(value) < schema["minItems"]:
                return {"error_message": f"{display_context} should have at least {schema['minItems']} items"}
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                return {"error_message": f"{display_context} should have at most {schema['maxItems']} items"}
            if schema.get("uniqueItems"):
                seen = set()
                for item in value:
                    key = json.dumps(item, sort_keys=True, default=str) if not isinstance(item, (int, float, str, bool)) else item
                    if key in seen:
                        return {"error_message": f"{display_context} should contain unique items"}
                    seen.add(key)
            item_schema = copy.deepcopy(schema.get("items", {}))
            # Some generated schemas incorrectly attach enum to the array itself
            # even though each item is the enumerated value. Treat that as item enum.
            if enum and "enum" not in item_schema:
                item_schema["enum"] = enum
            for idx, item in enumerate(value):
                error = self.validate_schema_value(item, item_schema, f"{context}[{idx}]")
                if error:
                    return error

        elif expected_type == "object":
            if not isinstance(value, dict):
                return {"error_message": f"{display_context} should be an object, not {type(value).__name__}"}
            # Required properties
            required = schema.get("required", [])
            for key in required:
                if key not in value:
                    return {"error_message": f"{display_context} is missing required property '{key}'"}
            # Additional properties
            if schema.get("additionalProperties") is False:
                allowed_keys = set((schema.get("properties") or {}).keys())
                extra = [k for k in value.keys() if k not in allowed_keys]
                if extra:
                    return {"error_message": f"{display_context} contains unexpected properties: {extra}"}
            # Validate each property
            props = schema.get("properties", {})
            for key, prop_schema in props.items():
                if key in value:
                    error = self.validate_schema_value(value[key], prop_schema, f"{context}.{key}")
                    if error:
                        return error

        return None  # No errors

    def _collect_schema_field_map(self, operation: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        field_map: Dict[str, Dict[str, Any]] = {}
        for param in operation.get("parameters", []):
            name = param.get("name")
            if name:
                field_map[name] = copy.deepcopy(param.get("schema", {}))
        if "requestBody" in operation:
            json_schema = (
                operation["requestBody"]
                .get("content", {})
                .get("application/json", {})
                .get("schema", {})
            )
            for name, schema in (json_schema.get("properties") or {}).items():
                field_map[name] = copy.deepcopy(schema)
        return field_map

    def _sanitize_schema_for_semantic_validation(self, schema: Any) -> Any:
        if isinstance(schema, list):
            return [self._sanitize_schema_for_semantic_validation(item) for item in schema]
        if not isinstance(schema, dict):
            return schema

        sanitized: Dict[str, Any] = {}
        for key, value in schema.items():
            if key in self._SEMANTIC_STRIP_KEYS:
                continue
            if key in {"properties", "items", "anyOf", "oneOf", "allOf"}:
                sanitized[key] = self._sanitize_schema_for_semantic_validation(value)
            else:
                sanitized[key] = copy.deepcopy(value)
        return sanitized

    def _semantic_error_contradicts_schema(
        self, args: Dict[str, Any], field_map: Dict[str, Dict[str, Any]], error_message: str
    ) -> bool:
        lowered = (error_message or "").lower()
        if not lowered:
            return False

        for name, value in args.items():
            schema = field_map.get(name)
            if not schema:
                continue

            # Exact enum matches should never be rejected by semantic validation.
            if schema.get("type") == "string" and isinstance(value, str):
                enum = schema.get("enum")
                if enum and value in enum and name.lower() in lowered and "enum" in lowered:
                    return True

            # Exact deterministic validation wins over contradictory semantic rejections.
            err = self.validate_schema_value(value, schema, name)
            if err is None and name.lower() in lowered and (
                "enum" in lowered or "pattern" in lowered or "format" in lowered
            ):
                return True

        return False

    async def validate_request(self, request: Request, path: str, method: str, api_name: str) -> Tuple[bool, Optional[Dict[str, str]]]:
        try:
            path_item = self.schema.get('paths', {}).get(path)
            if not path_item:
                return False, {"error_message": f"Path {path} not found in schema"}

            operation = path_item.get(method.lower())
            if not operation:
                return False, {"error_message": f"Method {method} not allowed for path {path}"}

            # Ensure server URL is set for openapi-core, but keep validation non-fatal
            try:
                self.schema["servers"] = [{"url": f"{request.base_url}{api_name}"}]
            except Exception:
                pass

            # --- Query & Path parameters ---
            if "parameters" in operation:
                for param in operation["parameters"]:
                    name = param.get("name")
                    location = param.get("in")
                    required = param.get("required", False)
                    schema = param.get("schema", {})
                    value = None

                    if location == "query":
                        value = request.query_params.get(name)
                    elif location == "path":
                        value = request.path_params.get(name)
                    else:
                        continue  # header/cookie skipped

                    if required and value is None:
                        return False, {"error_message": f"Missing required {location} parameter '{name}'"}

                    if value is not None:
                        # Convert to correct type for validation
                        converted_value = value
                        if schema.get("type") == "integer":
                            if str(value).isdigit():
                                converted_value = int(value)
                            else:
                                return False, {"error_message": f"{location.capitalize()} parameter '{name}' should be an integer"}
                        elif schema.get("type") == "number":
                            try:
                                converted_value = float(value)
                            except ValueError:
                                return False, {"error_message": f"{location.capitalize()} parameter '{name}' should be a number"}
                        elif schema.get("type") == "boolean":
                            if str(value).lower() in ["true", "1"]:
                                converted_value = True
                            elif str(value).lower() in ["false", "0"]:
                                converted_value = False
                            else:
                                return False, {"error_message": f"{location.capitalize()} parameter '{name}' should be a boolean"}

                        error = self.validate_schema_value(converted_value, schema, f"{location.capitalize()} parameter '{name}'")
                        if error:
                            return False, error

            # --- Content-Type validation for body requests ---
            content_type = request.headers.get('content-type', '')
            logging.debug(f"Content-Type check: method={method.upper()}, has_requestBody={'requestBody' in operation}, content_type='{content_type}'")
            if method.upper() in ['POST', 'PUT', 'PATCH'] and "requestBody" in operation:
                # Enforce JSON content type when requestBody is defined.
                if not content_type.startswith('application/json'):
                    logging.info(f"Content-Type validation failed: got '{content_type}', expected 'application/json'")
                    return False, {"error_message": f"Invalid Content-Type '{content_type}'. Expected 'application/json'"}
            
            # --- Request Body ---
            body = await request.body()
            logging.debug(f"Request body for {method} {path}: {body}")
            logging.debug(f"Content-Type: {content_type}")

            # Apply body validation rules by HTTP method.
            method_upper = method.upper()
            if method_upper in ['GET', 'HEAD', 'DELETE', 'OPTIONS']:
                # Body for these methods is ignored even if requestBody exists.
                if body and "requestBody" in operation:
                    logging.warning(f"{method_upper} request with body detected, ignoring requestBody validation for {path}")
            elif method_upper in ['POST', 'PUT', 'PATCH']:
                # Methods that normally carry a body.
                if body:
                    try:
                        body_json = json.loads(body)
                        if "requestBody" in operation:
                            json_schema = operation["requestBody"].get("content", {}).get("application/json", {}).get("schema", {})
                            error = self.validate_schema_value(body_json, json_schema, "Request body")
                            if error:
                                return False, error
                    except json.JSONDecodeError:
                        return False, {"error_message": "Invalid JSON in request body"}
                elif "requestBody" in operation:
                    # Enforce required request body for write methods.
                    req_body = operation["requestBody"]
                    if req_body.get("required", False):
                        return False, {"error_message": "Missing required request body"}
            else:
                # Keep legacy behavior for unknown methods.
                if body:
                    try:
                        body_json = json.loads(body)
                        if "requestBody" in operation:
                            json_schema = operation["requestBody"].get("content", {}).get("application/json", {}).get("schema", {})
                            error = self.validate_schema_value(body_json, json_schema, "Request body")
                            if error:
                                return False, error
                    except json.JSONDecodeError:
                        return False, {"error_message": "Invalid JSON in request body"}
                elif "requestBody" in operation:
                    req_body = operation["requestBody"]
                    if req_body.get("required", False):
                        logging.warning(f"Unknown method {method_upper} with required requestBody, skipping validation for {path}")
            
            # Run semantic validation for all methods.
            try:
                valid, error_message = await self.validate_request_by_schema(request, operation)
                if not valid:
                    return False, {"error_message": error_message}
            except Exception as e:
                logging.warning(f"Semantic validation skipped due to error: {e}")
                # Do not fail the request if semantic validation raises
                pass

            return True, None

        except Exception as error:
            error_msg = str(error)
            cause_msg = str(error.__cause__) if hasattr(error, '__cause__') and error.__cause__ else None
            logging.error(f"Validation error: {error_msg}")
            if cause_msg:
                logging.error(f"Validation error cause: {cause_msg}")
                return False, {"error_message": f"Validation error: {error_msg}", "error_cause": cause_msg}
            return False, {"error_message": f"Validation error: {error_msg}"} 
    
    async def validate_request_by_schema(self, request: Request, operation: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            # Initialize args with query and path parameters only
            args = {}
            args.update(dict(request.query_params))
            args.update(request.path_params)

            # Handle request body separately, swallow JSON errors
            request_body = None
            if "requestBody" in operation:
                try:
                    request_body = await request.json()
                except Exception:
                    request_body = None
                # Combine args and request body for validation
                if request_body:
                    args.update(request_body)
            
            # Collect full schema-visible parameters (not only args-present fields)
            # so semantic validation can detect misplaced information.
            params_schema = []
            
            # Process query, path, header, cookie parameters
            for param in operation.get("parameters", []):
                if param.get("deprecated", False):
                    continue
                name = param["name"]
                desc = param.get("description", "")
                location = param["in"]
                param_schema = copy.deepcopy(param.get("schema", {}))
                param_schema["name"] = f"{name}"
                param_schema["description"] = desc or f"{name} in {location}"
                params_schema.append(param_schema)

            # Process request body parameters if they exist
            if "requestBody" in operation:
                body_content = operation["requestBody"].get("content", {})
                json_schema = body_content.get("application/json", {}).get("schema", {})
                if "properties" in json_schema:
                    for k, v in json_schema["properties"].items():
                        param_schema = copy.deepcopy(v)
                        param_schema["name"] = k
                        param_schema["description"] = v.get("description", f"Field {k} in request body")
                        params_schema.append(param_schema)

            # Validate only schema-defined parameters and ignore extra fields.
            schema_param_names = {param.get("name") for param in params_schema}
            filtered_args = {k: v for k, v in args.items() if k in schema_param_names}
            
            logging.debug(f"Schema params: {schema_param_names}")
            logging.debug(f"Original args: {args}")
            logging.debug(f"Filtered args: {filtered_args}")
            
            # Get session_id from request headers
            session_id = request.headers.get("X-Session-ID")

            # Delegate to validate_params which already handles escaping and robustness
            semantic_params_schema = [
                self._sanitize_schema_for_semantic_validation(param_schema)
                for param_schema in params_schema
            ]

            validated_result = validate_params(
                params_schema=semantic_params_schema,
                args=filtered_args,
                temperature=0.001,
                session_id=session_id,
                validation_model=self.validation_model
            )
            if not validated_result[0]:
                field_map = self._collect_schema_field_map(operation)
                if self._semantic_error_contradicts_schema(filtered_args, field_map, validated_result[1]):
                    logging.warning(
                        "Ignoring contradictory semantic validation error: %s",
                        validated_result[1],
                    )
                    return True, ""
            return validated_result
        except Exception as e:
            logging.warning(f"validate_request_by_schema failed: {e}")
            # Treat as valid on failure
            return True, ""

class ValidateParamsResponse(BaseModel):
    valid: bool = Field(description="Whether the params are valid.")
    error_message: Optional[str] = Field(description="The error message if the params are invalid.")

def validate_params(
    params_schema: List[Dict],
    args: Dict,
    temperature: float = 0.001,
    session_id: Optional[str] = None,
    validation_model: str = "gpt-4.1-mini"
) -> Tuple[bool, str]:
    validate_params_prompt = """
Please validate the given function call arguments for semantic fit only.

**Validation Rules**:
1. **Scope**
   - Only validate arguments defined in the provided schemas.
   - Ignore arguments not present in the schema.
   - Type checking and other hard schema constraints have already been handled elsewhere.
   - Do NOT do enum membership checks, character-level regex matching, min/max/range checks, or length/item-count checks.

2. **Semantic Checks**
   - Focus only on semantic fit between each argument and its parameter description.
   - Use descriptions and examples to judge meaning, not literal hard constraints.
   - Detect redundant/overlapping information across arguments (e.g. `item="large pizza"` and `size="large"`).
   - Do NOT reject arguments based on dynamic runtime-state predicates not present in args (e.g. "must already exist", "must be logged in").
   - **Mandatory misplacement rule (hard fail)**: If an optional parameter is omitted or empty, and another free-form field contains semantics that clearly belong in that dedicated parameter, mark invalid.
   - **Described-format compliance (hard fail)**: When a description explicitly specifies a format using phrases like "in the format of" followed by a template and examples (e.g., "in the format of 'City, State', such as 'Los Angeles, CA'"), the value MUST follow that format. A bare value missing a required component (e.g., "Chicago" instead of "Chicago, IL") or using full names where abbreviations are shown in the examples (e.g., "California" instead of "CA") is invalid.
   - If a value is semantically reasonable for the described field, prefer valid.

3. **Error Messages**
   - Concise, precise, and human-readable.  
   - Do not include or suggest correct values.  
   - Only state which argument is invalid and why.
   - For semantic misplacement, use this template:
     `"<free_form_arg> contains semantics that must be provided via <structured_arg>; semantic misplacement."`

**Output Format**:
```

valid=\<true|false> error\_message="\<if false, list each invalid argument and reason>"

```

**Example**:
- **params_schema**  
  `[{"location": "The city that you want to go, e.g. 'Beijing, China'"}, {"date": "The start date for the booking, format: YYYY-MM-DD"}]`
- **args**  
  `{"location": "London", "date": "01/01/2024"}`
- **Output**  
  `valid=true error_message=""`

**Few-shot (semantic misplacement)**:
- **params_schema**
  `[{"name":"text","type":"string","description":"Free-form message body."},{"name":"labels","type":"array","description":"List of labels prefixed with #."}]`
- **args**
  `{"text":"release update #urgent"}`
- **Output**
  `valid=false error_message="text contains semantics that must be provided via labels; semantic misplacement."`

**Few-shot (described-format compliance)**:
- **params_schema**
  `[{"name":"city","type":"string","description":"The city name, in the format of 'City, State' or 'City, Country'. Examples: 'San Francisco, CA' or 'Paris, FR'."}]`
- **args**
  `{"city": "Nairobi, Kenya"}`
- **Output**
  `valid=false error_message="city uses full country name 'Kenya'; the described format examples show two-letter abbreviations (e.g., 'FR'); expected abbreviated form."`
"""
 
    # Use create_model utility for all models
    from utils.model_utils import create_model as create_camel_model
    model = create_camel_model(validation_model, max_tokens=16384, temperature=temperature)
    validate_params_assistant = ChatAgent(
        validate_params_prompt,
        model=model,
        step_timeout=get_agent_timeout(),
    )
    # Compose message using JSON only (no braces-escaping needed)
    message_payload = {
        "params_schema": params_schema,
        "args": args,
    }
    message = json.dumps(message_payload, ensure_ascii=False)
    response = validate_params_assistant.step(message, response_format=ValidateParamsResponse)
    
    try:
        if session_id:
            from ..utils.llm_usage import extract_token_usage

            usage = extract_token_usage(response)
            from ..handlers.session_handler import session_handler

            session_handler.record_llm_usage(
                session_id,
                category="request_validation",
                model=str(validation_model),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
    except Exception as usage_exc:
        logging.warning(f"Failed to record request validation token usage: {usage_exc}")
    
    response_json = response.msgs[0].parsed
    return response_json.valid, response_json.error_message

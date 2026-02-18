"""
Schema Utilities - Tools for processing OpenAPI schemas.

This module provides utilities for resolving schema references, extracting
parameter descriptions, and formatting schema information for LLM prompts.
"""

from typing import Any, Dict, List


def resolve_refs(schema: Dict[str, Any], full_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve $ref references in the schema.

    Recursively resolves JSON references ($ref) in the schema by looking them up
    in the full schema document.

    Args:
        schema: The schema object that may contain $ref references
        full_schema: The complete schema document containing component definitions

    Returns:
        Schema with all references resolved
    """
    if not isinstance(schema, dict):
        return schema

    result = {}
    for key, value in schema.items():
        if key == "$ref":
            # Reference format: "#/components/schemas/SomeName"
            ref_path = value.split("/")[1:]  # Skip the "#"
            ref_value = full_schema
            for part in ref_path:
                ref_value = ref_value[part]
            return resolve_refs(ref_value, full_schema)
        elif isinstance(value, dict):
            result[key] = resolve_refs(value, full_schema)
        elif isinstance(value, list):
            result[key] = [resolve_refs(item, full_schema) if isinstance(item, dict) else item for item in value]
        else:
            result[key] = value
    return result


def extract_schema_properties(schema_obj: Dict[str, Any], indent: str = "") -> str:
    """Extract properties and their descriptions from a schema object.

    Formats schema properties in a human-readable format suitable for LLM prompts,
    including handling of oneOf/anyOf/allOf, nested objects, and required fields.

    Args:
        schema_obj: Schema object to extract properties from
        indent: Indentation string for nested properties

    Returns:
        Formatted string describing the schema properties
    """
    result = ""
    if isinstance(schema_obj, dict):
        # Handle $ref references
        if '$ref' in schema_obj:
            return f"{indent}(Reference to another schema)\n"

        # Handle oneOf/anyOf/allOf
        if 'oneOf' in schema_obj:
            result += f"{indent}One of:\n"
            for i, option in enumerate(schema_obj['oneOf'], 1):
                result += f"{indent}  Option {i}:\n"
                result += extract_schema_properties(option, indent + "    ")
        elif 'anyOf' in schema_obj:
            result += f"{indent}Any of:\n"
            for i, option in enumerate(schema_obj['anyOf'], 1):
                result += f"{indent}  Option {i}:\n"
                result += extract_schema_properties(option, indent + "    ")

        # Handle regular properties
        if 'properties' in schema_obj:
            props = schema_obj['properties']
            required = schema_obj.get('required', [])
            for prop_name, prop_schema in props.items():
                req_marker = " (REQUIRED)" if prop_name in required else " (optional)"
                prop_type = prop_schema.get('type', 'any')
                prop_desc = prop_schema.get('description', 'No description')
                result += f"{indent}- **{prop_name}**{req_marker} [{prop_type}]: {prop_desc}\n"

                # Handle nested objects
                if prop_type == 'object' and 'properties' in prop_schema:
                    result += extract_schema_properties(prop_schema, indent + "  ")
    return result


def extract_schema_properties_simple(schema_obj: Dict[str, Any], indent: str = "") -> str:
    """Simple property extraction for basic schema information.

    A lighter-weight version of extract_schema_properties that provides just
    the essential information without deep nesting.

    Args:
        schema_obj: Schema object to extract properties from
        indent: Indentation string for formatting

    Returns:
        Formatted string with basic property information
    """
    result = ""
    if isinstance(schema_obj, dict):
        if 'oneOf' in schema_obj:
            for i, option in enumerate(schema_obj['oneOf'], 1):
                result += f"{indent}Option {i}:\n"
                result += extract_schema_properties_simple(option, indent + "  ")
        elif 'properties' in schema_obj:
            props = schema_obj['properties']
            required = schema_obj.get('required', [])
            for prop_name, prop_schema in props.items():
                req_marker = " (REQUIRED)" if prop_name in required else ""
                prop_type = prop_schema.get('type', 'any')
                prop_desc = prop_schema.get('description', '')
                result += f"{indent}- {prop_name}{req_marker}: {prop_desc}\n"
    return result


def extract_parameter_descriptions(operation: Dict[str, Any]) -> str:
    """Extract parameter descriptions from an operation.

    Formats path/query parameters and request body parameters from an OpenAPI
    operation into a readable description.

    Args:
        operation: OpenAPI operation object

    Returns:
        Formatted string describing all parameters
    """
    descriptions = ""

    # Extract path/query parameters
    if 'parameters' in operation:
        descriptions += "\n**Path/Query Parameters:**\n"
        for param in operation['parameters']:
            param_name = param.get('name', '')
            param_in = param.get('in', '')
            param_desc = param.get('description', 'No description')
            param_required = param.get('required', False)
            param_type = param.get('schema', {}).get('type', 'string') if 'schema' in param else 'string'
            req_marker = " (REQUIRED)" if param_required else " (optional)"
            descriptions += f"- **{param_name}**{req_marker} [{param_type}] in {param_in}: {param_desc}\n"

    # Extract request body parameters
    if 'requestBody' in operation:
        rb = operation['requestBody']
        descriptions += "\n**Request Body Parameters:**\n"

        if 'content' in rb:
            for content_type, content in rb['content'].items():
                if 'schema' in content:
                    schema = content['schema']
                    # Add overall description if present
                    if 'description' in schema:
                        descriptions += f"Description: {schema['description']}\n"
                    # Extract properties
                    props_desc = extract_schema_properties(schema)
                    if props_desc:
                        descriptions += props_desc
                    break  # Use first content type

    return descriptions


def extract_response_descriptions(operation: Dict[str, Any]) -> str:
    """Extract response field descriptions from an operation.

    Formats the response schema fields from an OpenAPI operation into
    a readable description, focusing on the 200 success response.

    Args:
        operation: OpenAPI operation object

    Returns:
        Formatted string describing response fields
    """
    descriptions = ""

    if 'responses' in operation:
        descriptions += "\n**Response Fields:**\n"
        # Focus on 200 response
        if '200' in operation['responses']:
            resp = operation['responses']['200']
            if 'content' in resp:
                for content_type, content in resp['content'].items():
                    if 'schema' in content:
                        schema = content['schema']
                        props_desc = extract_schema_properties(schema)
                        if props_desc:
                            descriptions += props_desc
                        break  # Use first content type

    return descriptions


def extract_toolkit_info(schema: Dict[str, Any]) -> str:
    """Extract toolkit information from OpenAPI schema.

    Extracts and formats general API information, tags, and description
    from the OpenAPI schema's info section.

    Args:
        schema: Full OpenAPI schema

    Returns:
        Formatted string with toolkit information
    """
    if not schema or not isinstance(schema, dict):
        return ""

    toolkit_info = ""
    info = schema.get('info', {})
    if info:
        toolkit_info = f"**Title**: {info.get('title', 'Unknown API')}\n"
        toolkit_info += f"**Version**: {info.get('version', 'Unknown')}\n"
        toolkit_info += f"**Description**: {info.get('description', 'No description available')}\n"

        # Add available tags/categories if present
        tags = schema.get('tags', [])
        if tags:
            toolkit_info += "\n**Available Categories/Tags**:\n"
            for tag in tags[:10]:  # Limit to first 10 tags
                name = tag.get('name', '')
                desc = tag.get('description', '')
                toolkit_info += f"- **{name}**: {desc}\n" if desc else f"- {name}\n"

    return toolkit_info

# BFCL Error Types
"""
Error type definitions for BFCL single-turn evaluation.
Based on official BFCL evaluation error classification.
"""

# AST Decoder Errors
AST_DECODER_FAILED = "ast_decoder:decoder_failed"
AST_DECODER_WRONG_FORMAT = "ast_decoder:decoder_wrong_output_format"

# Simple Function Checker Errors
SIMPLE_WRONG_FUNC_NAME = "simple_function_checker:wrong_func_name"
SIMPLE_WRONG_COUNT = "simple_function_checker:wrong_count"
SIMPLE_MISSING_REQUIRED = "simple_function_checker:missing_required"
SIMPLE_UNEXPECTED_PARAM = "simple_function_checker:unexpected_param"
SIMPLE_MISSING_OPTIONAL = "simple_function_checker:missing_optional"
SIMPLE_UNCLEAR = "simple_function_checker:unclear"

# Type Errors
TYPE_ERROR_SIMPLE = "type_error:simple"
TYPE_ERROR_NESTED = "type_error:nested"
TYPE_ERROR_JAVA = "type_error:java"
TYPE_ERROR_JS = "type_error:js"

# Value Errors
VALUE_ERROR_STRING = "value_error:string"
VALUE_ERROR_LIST = "value_error:list/tuple"
VALUE_ERROR_DICT_KEY = "value_error:dict_key"
VALUE_ERROR_DICT_VALUE = "value_error:dict_value"
VALUE_ERROR_LIST_DICT_COUNT = "value_error:list_dict_count"
VALUE_ERROR_OTHERS = "value_error:others"

# Multiple Function Checker Errors
MULTIPLE_WRONG_COUNT = "multiple_function_checker:wrong_count"

# Parallel Function Checker Errors
PARALLEL_WRONG_COUNT = "parallel_function_checker_no_order:wrong_count"
PARALLEL_CANNOT_FIND_MATCH = "parallel_function_checker_no_order:cannot_find_match"
PARALLEL_ENFORCE_ORDER_WRONG_COUNT = "parallel_function_checker_enforce_order:wrong_count"

# Dict Checker Errors
DICT_CHECKER_UNCLEAR = "dict_checker:unclear"
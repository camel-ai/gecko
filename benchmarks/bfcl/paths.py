"""Canonical paths for the project-local BFCL v4 data bundle."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BFCL_DATA_DIR = PROJECT_ROOT / "data" / "bfcl"
BFCL_TASK_DIR = BFCL_DATA_DIR / "task"
BFCL_POSSIBLE_ANSWER_DIR = BFCL_DATA_DIR / "possible_answer"
BFCL_MULTI_TURN_FUNC_DOC_DIR = BFCL_DATA_DIR / "multi_turn_func_doc"
BFCL_OPENAPI_DIR = BFCL_DATA_DIR / "openapi"
BFCL_OPENAPI_SINGLE_TURN_DIR = BFCL_OPENAPI_DIR / "single_turn"
BFCL_OPENAPI_MULTI_TURN_DIR = BFCL_OPENAPI_DIR / "multi_turn"

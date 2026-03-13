import os
from pathlib import Path

from .category_mapping import VERSION_PREFIX

LOCAL_SERVER_PORT = 1053
LOCAL_SERVER_MAX_CONCURRENT_REQUEST = 100
H100_X8_PRICE_PER_HOUR = 23.92

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.getenv("BFCL_PROJECT_ROOT", Path(__file__).resolve().parents[4]))

RESULT_PATH = PROJECT_ROOT / "result"
SCORE_PATH = PROJECT_ROOT / "score"
DOTENV_PATH = PROJECT_ROOT / ".env"
TEST_IDS_TO_GENERATE_PATH = PROJECT_ROOT / "test_case_ids_to_generate.json"
LOCK_DIR = PROJECT_ROOT / ".file_locks"

DATA_ROOT = PROJECT_ROOT / "data" / "bfcl_v4"
PROMPT_PATH = DATA_ROOT / "task"
MULTI_TURN_FUNC_DOC_PATH = DATA_ROOT / "multi_turn_func_doc"
POSSIBLE_ANSWER_PATH = DATA_ROOT / "possible_answer"
MEMORY_PREREQ_CONVERSATION_PATH = DATA_ROOT / "memory_prereq_conversation"
UTILS_PATH = PACKAGE_ROOT / "scripts"
FORMAT_SENSITIVITY_IDS_PATH = PROMPT_PATH / f"{VERSION_PREFIX}_format_sensitivity.json"

RESULT_FILE_PATTERN = f"{VERSION_PREFIX}_*_result.json"

RED_FONT = "\033[91m"
RESET = "\033[0m"

RESULT_PATH.mkdir(parents=True, exist_ok=True)
SCORE_PATH.mkdir(parents=True, exist_ok=True)
LOCK_DIR.mkdir(parents=True, exist_ok=True)

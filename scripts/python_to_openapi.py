#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on sys.path for local imports when run as a script.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from openapi_converter.llm_pipeline import convert_python_to_openapi

load_dotenv()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enhanced Python to OpenAPI 3.1 converter with rule-based state extraction"
    )
    parser.add_argument("input_file", help="Main Python file to convert (e.g., tools.py)")
    parser.add_argument("-o", "--output", default="openapi_spec.json", help="Output JSON file")
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="LLM model to use (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--additional-files",
        nargs="+",
        default=[],
        help="Additional Python files for context (e.g., data_model.py)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Concurrent endpoint workers (default: 10)",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=120.0,
        help="Per-LLM-call timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--class-name",
        default=None,
        help="Optional target class name to convert (default: auto first public class)",
    )
    parser.add_argument(
        "--description-only",
        action="store_true",
        help="Only generate toolkit metadata/description and skip endpoint generation",
    )
    parser.add_argument(
        "--state-output",
        default=None,
        help="Optional path to save extracted state data (info.x-default-state) as JSON",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for blocking diagnosis",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.debug:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        for handler in root_logger.handlers:
            handler.setLevel(logging.INFO)
        logging.getLogger("scripts.openapi_converter").setLevel(logging.DEBUG)
        logging.getLogger("scripts.openapi_converter.llm_pipeline").setLevel(logging.DEBUG)
        logging.getLogger("scripts.openapi_converter.openapi_utils").setLevel(logging.DEBUG)
        logging.getLogger("openapi_converter").setLevel(logging.DEBUG)
        logging.getLogger("openapi_converter.llm_pipeline").setLevel(logging.DEBUG)
        logging.getLogger("openapi_converter.openapi_utils").setLevel(logging.DEBUG)
        logging.getLogger("openai").setLevel(logging.DEBUG)
        logging.getLogger("httpcore").setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        # Keep CAMEL internals quiet to avoid giant prompt dumps.
        logging.getLogger("camel.base_model").setLevel(logging.WARNING)
        logging.getLogger("camel.camel.agents.chat_agent").setLevel(logging.WARNING)
        logging.getLogger("camel.camel.utils.token_counting").setLevel(logging.WARNING)
    return convert_python_to_openapi(
        input_file=args.input_file,
        output=args.output,
        model=args.model,
        additional_files=args.additional_files,
        workers=args.workers,
        llm_timeout=args.llm_timeout,
        class_name=args.class_name,
        description_only=args.description_only,
        state_output=args.state_output,
    )


if __name__ == "__main__":
    sys.exit(main())

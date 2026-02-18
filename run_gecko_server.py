import argparse
import os
import sys

import uvicorn

# Add project root to Python path.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load .env variables if available.
try:
    from utils.env_loader import load_environment_variables

    load_environment_variables()
except ImportError:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gecko server")
    parser.add_argument(
        "--schemas_dir",
        type=str,
        default="data/openapi",
        help="Directory containing OpenAPI schemas",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--response-model",
        type=str,
        default="gpt-5-mini",
        help="LLM model for response generation (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--state-model",
        type=str,
        default="gpt-5-mini",
        help="LLM model for state update (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--validation-model",
        type=str,
        default="gpt-5-mini",
        help="LLM model for request validation (default: gpt-5-mini)",
    )

    args = parser.parse_args()

    state_model = args.state_model.strip()
    validation_model = args.validation_model.strip()
    if state_model.lower() in {"none", "null", ""}:
        parser.error("--state-model cannot be none/null/empty")
    if validation_model.lower() in {"none", "null", ""}:
        parser.error("--validation-model cannot be none/null/empty")

    print(f"Starting Gecko with schemas from: {args.schemas_dir}")

    if args.workers > 1:
        with open("app_module.py", "w", encoding="utf-8") as f:
            f.write(
                "from gecko import GeckoServer\n"
                "app = GeckoServer(\n"
                f"    schemas_dir={args.schemas_dir!r},\n"
                f"    response_model={args.response_model!r},\n"
                f"    state_model={state_model!r},\n"
                f"    validation_model={validation_model!r},\n"
                ").app\n"
            )
        uvicorn.run("app_module:app", host=args.host, port=args.port, workers=args.workers)
        return

    from gecko import GeckoServer

    server = GeckoServer(
        schemas_dir=args.schemas_dir,
        response_model=args.response_model,
        state_model=state_model,
        validation_model=validation_model,
    )
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

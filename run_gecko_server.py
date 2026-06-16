import argparse
import asyncio
import logging
import multiprocessing as mp
import os
import socket
import sys
import time

import uvicorn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from utils.env_loader import load_environment_variables

    load_environment_variables()
except ImportError:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _serve_with_reuseport(host: str, port: int, app_path: str) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((host, port))
    config = uvicorn.Config(app_path)
    server = uvicorn.Server(config)
    asyncio.run(server.serve(sockets=[sock]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gecko server")
    parser.add_argument(
        "--schemas-dir",
        "--schemas_dir",
        dest="schemas_dir",
        type=str,
        action="append",
        default=None,
        help=(
            "Directory containing OpenAPI schemas. Repeat to layer override "
            "schemas before fallback schemas. Defaults to data/bfcl/openapi."
        ),
    )
    parser.add_argument("--workers", type=int, default=15, help="Number of worker processes")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--response-model",
        type=str,
        default="gpt-5.5",
        help="LLM model for response generation (default: gpt-5.5)",
    )
    parser.add_argument(
        "--state-model",
        type=str,
        default="gpt-5.5",
        help="LLM model for state update (default: gpt-5.5)",
    )
    parser.add_argument(
        "--validation-model",
        type=str,
        default="gpt-5.5",
        help="LLM model for request validation (default: gpt-5.5)",
    )
    args = parser.parse_args()
    schemas_dir = args.schemas_dir or ["data/bfcl/openapi"]

    state_model = args.state_model.strip()
    validation_model = args.validation_model.strip()

    os.environ["GECKO_WORKERS"] = str(args.workers)
    print(f"Starting Gecko with schemas from: {schemas_dir}")

    if args.workers > 1:
        with open("app_module.py", "w", encoding="utf-8") as f:
            f.write(
                "import os\n"
                "from gecko import GeckoServer\n"
                f"os.environ['GECKO_WORKERS'] = {str(args.workers)!r}\n"
                "app = GeckoServer(\n"
                f"    schemas_dir={schemas_dir!r},\n"
                f"    response_model={args.response_model!r},\n"
                f"    state_model={state_model!r},\n"
                f"    validation_model={validation_model!r},\n"
                f"    workers={args.workers!r},\n"
                ").app\n"
            )

        ctx = mp.get_context("spawn")

        def _spawn() -> mp.Process:
            p = ctx.Process(
                target=_serve_with_reuseport,
                args=(args.host, args.port, "app_module:app"),
            )
            p.start()
            return p

        procs = [_spawn() for _ in range(args.workers)]
        shutting_down = False
        try:
            while not shutting_down:
                time.sleep(2)
                for i, p in enumerate(procs):
                    if not p.is_alive():
                        logging.warning(
                            "[supervisor] worker %d (pid=%s) exited with code=%s; respawning",
                            i,
                            p.pid,
                            p.exitcode,
                        )
                        p.join()
                        procs[i] = _spawn()
        except KeyboardInterrupt:
            shutting_down = True
            for p in procs:
                if p.is_alive():
                    p.terminate()
            for p in procs:
                p.join()
        return

    from gecko import GeckoServer

    server = GeckoServer(
        schemas_dir=schemas_dir,
        response_model=args.response_model,
        state_model=state_model,
        validation_model=validation_model,
        workers=args.workers,
    )
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

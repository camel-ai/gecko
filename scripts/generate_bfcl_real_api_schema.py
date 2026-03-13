#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

PREFIX_ALIASES: dict[str, list[str]] = {
    "GorillaFileSystem": ["gorillafilesystem"],
    "MathAPI": ["mathapi"],
    "MessageAPI": ["messageapi"],
    "TicketAPI": ["ticketapi"],
    "TradingBot": ["tradingbot"],
    "TravelAPI": ["travelapi", "travelbooking"],
    "TwitterAPI": ["twitterapi", "postingapi"],
    "VehicleControlAPI": ["vehiclecontrolapi", "vehiclecontrol"],
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_alias_operation(operation: dict[str, Any], prefix: str, fn_name: str) -> dict[str, Any]:
    aliased = copy.deepcopy(operation)
    op_id = aliased.get("operationId", fn_name)
    aliased["operationId"] = f"{prefix}_{op_id}"
    return aliased


def generate_bfcl_real_api_schema(repo_root: Path) -> Path:
    source_dir = repo_root / "data" / "openapi" / "multi_turn" / "mock"
    output_file = (
        repo_root
        / "data"
        / "openapi"
        / "multi_turn"
        / "real"
        / "BFCLMultiTurnRealAPI.json"
    )

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    merged_paths: dict[str, Any] = {}

    for spec_path in sorted(source_dir.glob("*.json")):
        spec = _read_json(spec_path)
        domain_name = spec_path.stem
        prefixes = PREFIX_ALIASES.get(domain_name, [])

        for path, path_item in spec.get("paths", {}).items():
            if path in merged_paths:
                raise ValueError(f"Duplicate path detected while merging: {path}")

            merged_paths[path] = copy.deepcopy(path_item)

            fn_name = path.lstrip("/")
            post_op = path_item.get("post") if isinstance(path_item, dict) else None

            if not isinstance(post_op, dict):
                continue

            for prefix in prefixes:
                alias_path = f"/{prefix}_{fn_name}"
                if alias_path in merged_paths:
                    raise ValueError(f"Duplicate alias path detected: {alias_path}")

                merged_paths[alias_path] = {
                    "post": _build_alias_operation(post_op, prefix, fn_name)
                }

    output_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "BFCL Multi-turn Real API",
            "version": "1.0.0",
            "description": (
                "Canonical schema for BFCL multi-turn real operations. "
                "Includes original endpoint names and prefixed aliases for "
                "tool-name compatibility in real execution mode."
            ),
        },
        "servers": [
            {
                "url": "http://localhost:8000",
                "description": "Local mock server",
            }
        ],
        "paths": {k: merged_paths[k] for k in sorted(merged_paths.keys())},
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(output_spec, f, indent=2)
        f.write("\n")

    return output_file


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    output_path = generate_bfcl_real_api_schema(repo_root)
    print(f"Generated BFCL RealAPI schema: {output_path}")


if __name__ == "__main__":
    main()

"""Paths for in-repo tau2 retail data."""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[5] / "data" / "taubench"
RETAIL_DATA_DIR = DATA_DIR / "domains" / "retail"
RETAIL_DB_PATH = str(RETAIL_DATA_DIR / "db.json")
RETAIL_POLICY_PATH = str(RETAIL_DATA_DIR / "policy.md")
RETAIL_TASK_SET_PATH = str(RETAIL_DATA_DIR / "tasks.json")

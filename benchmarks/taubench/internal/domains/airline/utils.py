"""Paths for in-repo tau2 airline data."""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[5] / "data" / "taubench"
AIRLINE_DATA_DIR = DATA_DIR / "domains" / "airline"
AIRLINE_DB_PATH = str(AIRLINE_DATA_DIR / "db.json")
AIRLINE_POLICY_PATH = str(AIRLINE_DATA_DIR / "policy.md")
AIRLINE_TASK_SET_PATH = str(AIRLINE_DATA_DIR / "tasks.json")

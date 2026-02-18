"""Environment variable loading helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_environment_variables(env_file: Optional[str] = None) -> bool:
    """Load environment variables from a .env file.

    Args:
        env_file: Optional path to a .env file. If omitted, uses project root `.env`.

    Returns:
        True if values were loaded successfully, otherwise False.
    """
    try:
        from dotenv import load_dotenv

        if env_file is None:
            env_file = str(Path(__file__).resolve().parent.parent / ".env")

        if not Path(env_file).exists():
            logger.warning("Environment file not found: %s", env_file)
            return False

        success = load_dotenv(env_file, override=False)
        if not success:
            logger.warning("Failed to load environment file: %s", env_file)
            return False

        logger.info("Environment variables loaded from: %s", env_file)
        if os.getenv("OPENAI_API_KEY"):
            logger.info("OPENAI_API_KEY detected")
        else:
            logger.warning("OPENAI_API_KEY not found in environment variables")
        return True
    except ImportError:
        logger.error("python-dotenv is not installed. Run: pip install python-dotenv")
        return False
    except Exception as exc:
        logger.error("Error loading environment variables: %s", exc)
        return False

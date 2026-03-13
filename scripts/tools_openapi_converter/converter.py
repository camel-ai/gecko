"""Thin orchestrator that wires the pipeline stages together."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from camel.agents import ChatAgent

from .enrichment import EndpointEnricher
from .models import strip_markdown_fence
from .parser import ToolDescriptionParser
from .schema import SchemaProcessor
from .validation import SpecValidator

logger = logging.getLogger(__name__)


class ToolToOpenAPIConverter:
    """Convert free-form tool descriptions to OpenAPI 3.1.0."""

    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        *,
        include_default_state: bool = False,
    ) -> None:
        self.model_name = model_name
        self.include_default_state = include_default_state
        self._model = None  # lazily created, reused across all LLM calls
        self._schema = SchemaProcessor()
        self._parser = ToolDescriptionParser(self._schema, self._call_llm_json)
        self._enricher = EndpointEnricher(self._schema, self._call_llm_json)

    def convert(self, payload: Any, api_name: Optional[str] = None) -> Dict[str, Any]:
        tool_irs = self._parser.extract_tool_irs(payload)
        if not tool_irs:
            raise ValueError("No tool descriptions found in input payload")

        resolved_api_name, api_description = self._enricher.generate_api_metadata(
            tool_irs,
            requested_api_name=api_name,
        )
        spec: Dict[str, Any] = {
            "openapi": "3.1.0",
            "info": {
                "title": resolved_api_name,
                "version": "1.0.0",
                "description": api_description,
            },
            "servers": [{"url": "http://localhost:8000"}],
            "paths": {},
        }
        if self.include_default_state:
            spec["info"]["x-default-state"] = {
                "global": {"runtime_defaults": {}},
                "tools": {},
            }

        logger.info("Converting %d tools into endpoints...", len(tool_irs))
        for idx, tool in enumerate(tool_irs, 1):
            logger.info("  [%d/%d] %s", idx, len(tool_irs), tool.name)
            endpoint, tool_state = self._enricher.build_endpoint(
                tool,
                include_tool_state=self.include_default_state,
            )
            spec["paths"][f"/{tool.name}"] = {"post": endpoint}
            if self.include_default_state and tool_state:
                spec["info"]["x-default-state"]["tools"][tool.name] = tool_state

        SpecValidator.post_process(spec)
        SpecValidator.mandatory_validate(spec)
        return spec

    # ------------------------------------------------------------------
    # Shared LLM caller (injected into parser & enricher)
    # ------------------------------------------------------------------

    def _call_llm_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                if self._model is None:
                    from utils.model_utils import create_model

                    self._model = create_model(
                        self.model_name,
                        max_tokens=8192,
                        temperature=0.001,
                    )
                agent = ChatAgent(system_prompt, model=self._model)
                response = agent.step(user_prompt)
                content = strip_markdown_fence(response.msg.content)
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError("LLM output must be a JSON object")
                return parsed
            except Exception as exc:
                last_exc = exc
                self._model = None
                if attempt >= 3:
                    break
                logger.warning(
                    "LLM JSON call failed on attempt %d/3 for model=%s: %s",
                    attempt,
                    self.model_name,
                    exc,
                )
                time.sleep(0.5 * attempt)
        raise RuntimeError(f"LLM call failed: {last_exc}") from last_exc

"""Small runtime utilities for the in-repo tau2 airline/retail subset."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict


class BaseModelNoExtra(BaseModel):
    """Pydantic base matching tau2 DB models: reject unknown top-level fields."""

    model_config = ConfigDict(extra="forbid")


def get_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_file(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_file(path: str | Path, data: Any, **kwargs: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, **kwargs)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def get_dict_hash(data: dict[str, Any]) -> str:
    payload = json.dumps(_jsonable(data), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_pydantic_hash(model: BaseModel) -> str:
    return get_dict_hash(model.model_dump(mode="json", exclude_none=False))


def _deep_update(base: Any, update: Any) -> Any:
    if isinstance(base, dict) and isinstance(update, dict):
        merged = deepcopy(base)
        for key, value in update.items():
            merged[key] = _deep_update(merged.get(key), value)
        return merged
    return deepcopy(update)


ModelT = TypeVar("ModelT", bound=BaseModel)


def update_pydantic_model_with_dict(model: ModelT, update_data: dict[str, Any]) -> ModelT:
    merged = _deep_update(model.model_dump(mode="json", exclude_none=False), update_data)
    return type(model).model_validate(merged)

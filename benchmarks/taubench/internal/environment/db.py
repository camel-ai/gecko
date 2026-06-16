"""DB base class for the in-repo tau2 airline/retail runtime."""

from __future__ import annotations

from typing import Any

from benchmarks.taubench.internal.utils import (
    BaseModelNoExtra,
    dump_file,
    get_pydantic_hash,
    load_file,
)


class DB(BaseModelNoExtra):
    """Domain database base class."""

    @classmethod
    def load(cls, path: str) -> "DB":
        return cls.model_validate(load_file(path))

    def dump(self, path: str, exclude_defaults: bool = False, **kwargs: Any) -> None:
        dump_file(path, self.model_dump(exclude_defaults=exclude_defaults), **kwargs)

    def get_json_schema(self) -> dict[str, Any]:
        return self.model_json_schema()

    def get_hash(self) -> str:
        return get_pydantic_hash(self)

    def get_statistics(self) -> dict[str, Any]:
        return {}


def get_db_json_schema(db: DB | None = None) -> dict[str, Any]:
    return {} if db is None else db.get_json_schema()

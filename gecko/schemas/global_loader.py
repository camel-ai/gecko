from typing import Optional

# Simple global registry for the active SchemaLoader instance
_global_schema_loader = None


def set_global_schema_loader(loader) -> None:
    global _global_schema_loader
    _global_schema_loader = loader


def get_global_schema_loader():
    return _global_schema_loader



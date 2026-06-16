"""Configuration helpers for the tau2-bench integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


SUPPORTED_DOMAINS = {"airline", "retail"}


@dataclass(frozen=True)
class DomainToolConfig:
    read_tools: List[str]
    write_tools: List[str]
    generic_tools: List[str]
    mock_openapi_file: str


DOMAIN_TOOLS: Dict[str, DomainToolConfig] = {
    "airline": DomainToolConfig(
        read_tools=[
            "search_direct_flight",
            "search_onestop_flight",
            "get_user_details",
            "get_reservation_details",
            "list_all_airports",
            "get_flight_status",
        ],
        write_tools=[
            "book_reservation",
            "cancel_reservation",
            "send_certificate",
            "update_reservation_baggages",
            "update_reservation_flights",
            "update_reservation_passengers",
        ],
        generic_tools=[
            "calculate",
            "transfer_to_human_agents",
        ],
        mock_openapi_file="AirlineMockAPI.json",
    ),
    "retail": DomainToolConfig(
        read_tools=[
            "find_user_id_by_name_zip",
            "find_user_id_by_email",
            "get_user_details",
            "get_order_details",
            "get_product_details",
            "get_item_details",
            "list_all_product_types",
        ],
        write_tools=[
            "cancel_pending_order",
            "exchange_delivered_order_items",
            "modify_pending_order_address",
            "modify_pending_order_items",
            "modify_pending_order_payment",
            "modify_user_address",
            "return_delivered_order_items",
        ],
        generic_tools=[
            "calculate",
            "transfer_to_human_agents",
        ],
        mock_openapi_file="RetailMockAPI.json",
    ),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def taubench_openapi_dir() -> Path:
    return repo_root() / "data" / "taubench" / "openapi"


def taubench_mock_openapi_dir() -> Path:
    return taubench_openapi_dir() / "mock"


def validate_domain(domain: str) -> str:
    normalized = (domain or "").strip().lower()
    if normalized not in SUPPORTED_DOMAINS:
        raise ValueError(
            f"Unsupported tau2 domain {domain!r}. Supported domains: {sorted(SUPPORTED_DOMAINS)}"
        )
    return normalized

from benchmarks.bfcl.multi_turn.executor import CLASS_FILE_PATH_MAPPING

MULTI_TURN_FUNC_DOC_FILE_MAPPING = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
    "WebSearchAPI": "web_search.json",
    "MemoryAPI_kv": "memory_kv.json",
    "MemoryAPI_vector": "memory_vector.json",
    "MemoryAPI_rec_sum": "memory_rec_sum.json",
}

BACKEND_PATH_PREFIX = "benchmarks.bfcl.multi_turn.func_source_code"

STATELESS_CLASSES = ["MathAPI"]

OMIT_STATE_INFO_CLASSES = [
    "MemoryAPI_kv",
    "MemoryAPI_vector",
    "MemoryAPI_rec_sum",
    "WebSearchAPI",
]

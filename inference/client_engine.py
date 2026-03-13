
import json
import logging
import os
import requests
import time
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

# Import BFCL components (these don't depend on FastAPI)
from benchmarks.bfcl.benchmark import BFCLBenchmark
from benchmarks.bfcl.data.loader import TestCase
from utils.test_case_adapter import TestCaseAdapter
from benchmarks.bfcl.utils import derive_single_turn_schema_name

def get_bfcl_benchmark():
    return BFCLBenchmark()

logger = logging.getLogger(__name__)


class TaskInfraError(RuntimeError):
    """Infrastructure-level error: task should be treated as not executed."""


@dataclass
class InferenceConfig:
    model_name: str = "gpt-4.1-mini"
    max_retries: int = 3
    target_score: float = 1.0
    max_turns: Optional[int] = None
    trace_compact: bool = False
    
    mock_server_url: str = "http://localhost:8000"
    enable_real_execution: bool = True
    
    agent_timeout: int = 60
    system_message: Optional[str] = None
    agent_max_iteration: int = 30
    agent_summarize_threshold: Optional[int] = None
    
    enable_judge: bool = True
    enable_checklist: bool = True
    final_state_only: bool = False
    
    enable_tool_filtering: bool = False


@dataclass
class TurnResult:
    turn_idx: int
    success: bool
    mock_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    real_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    judge_score: float = 0.0
    error: Optional[str] = None
    mock_config: Optional[Dict[str, Any]] = None
    calibrated_config: Optional[Dict[str, Any]] = None
    llm_costs: Dict[str, float] = field(default_factory=dict)
    checklist: List[Dict[str, Any]] = field(default_factory=list)
    all_attempts: List[Dict[str, Any]] = field(default_factory=list)
    task_agent_response: str = ""


@dataclass
class InferenceResult:
    test_id: str
    success: bool
    turns: List[TurnResult] = field(default_factory=list)
    overall_success: bool = False
    final_score: float = 0.0
    total_time: float = 0.0
    
    mock_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    real_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    judge_score: float = 0.0
    
    config_history: List[Dict[str, Any]] = field(default_factory=list)

    total_turns: int = 0
    executed_turns: int = 0
    max_turns_applied: Optional[int] = None

    process_trace: Dict[str, Any] = field(default_factory=dict)
    
    total_llm_costs: Dict[str, float] = field(default_factory=dict)


class MockServerClient:

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.timeout = 60
        self._lock_guard = threading.Lock()
        self._single_flight_locks: Dict[str, threading.Lock] = {}
    
    def _task_lock_key(self, test_case: Optional[TestCase] = None, session_id: Optional[str] = None) -> str:
        if session_id:
            return f"session:{session_id}"
        if test_case is not None:
            try:
                test_id = TestCaseAdapter.get_id(test_case)
                if test_id:
                    return f"task:{test_id}"
            except Exception:
                pass
        return "global"

    @contextmanager
    def _single_flight(self, key: str):
        with self._lock_guard:
            lock = self._single_flight_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._single_flight_locks[key] = lock
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    def create_session(
        self,
        test_case: Optional[TestCase] = None,
        *,
        timeout: Optional[float] = None,
        retries: int = 0,
        backoff_sec: float = 0.5,
    ) -> str:
        request_timeout = float(timeout if timeout is not None else self.timeout)
        max_attempts = 1
        last_error: Optional[Exception] = None

        key = self._task_lock_key(test_case=test_case)
        with self._single_flight(key):
            for attempt in range(1, max_attempts + 1):
                try:
                    response = self.session.get(
                        f"{self.base_url}/session-id",
                        timeout=request_timeout,
                    )
                    response.raise_for_status()
                    session_id = response.json()["session_id"]
                    self._last_session_id = session_id
                    logger.info(f"Created session: {session_id}")
                    return session_id
                except Exception as e:
                    last_error = e
                    logger.error(
                        "Failed to create session (timeout=%.1fs): %s",
                        request_timeout,
                        last_error,
                    )
                    raise TaskInfraError(
                        f"UNTESTED_TASK: create-session failed (timeout={request_timeout:.1f}s): {last_error}"
                    ) from e

    @staticmethod
    def _normalize_real_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize tool call payloads for /update-state-from-real."""
        normalized = []
        for tc in tool_calls or []:
            if not isinstance(tc, dict):
                continue
            name = tc.get("name") or tc.get("function") or tc.get("function_name") or ""
            if not name:
                continue
            arguments = tc.get("arguments") or tc.get("args") or {}
            normalized.append(
                {
                    "name": name,
                    "arguments": arguments if isinstance(arguments, dict) else arguments,
                    "result": tc.get("result"),
                }
            )
        return normalized

    def sync_state_from_real_results(
        self,
        *,
        base_state: Dict[str, Any],
        tool_calls: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        test_case: Optional[TestCase] = None,
        io_timeout: Optional[float] = None,
        retries: int = 0,
    ) -> Dict[str, Any]:
        """Sync real tool results to mock server and return updated state."""
        sid = session_id or self.create_session(test_case)
        request_timeout = float(io_timeout if io_timeout is not None else self.timeout)

        self.set_session_state(
            sid,
            base_state or {},
            bootstrap_mode=(
                "skip"
                if isinstance((base_state or {}).get("runtime_state"), dict)
                and bool((base_state or {}).get("runtime_state"))
                else "auto"
            ),
            timeout=request_timeout,
            retries=retries,
            backoff_sec=0.5,
        )

        normalized = self._normalize_real_tool_calls(tool_calls)
        if not normalized:
            return base_state or {}

        key = self._task_lock_key(session_id=sid)
        with self._single_flight(key):
            try:
                headers = {"X-Session-ID": sid}
                response = self.session.post(
                    f"{self.base_url}/update-state-from-real",
                    headers=headers,
                    json={"tool_calls": normalized},
                    timeout=request_timeout,
                )
                response.raise_for_status()
                payload = response.json() if response.content else {}
                updated = payload.get("updated_state") if isinstance(payload, dict) else None
                if isinstance(updated, dict):
                    return updated
            except Exception as e:
                logger.error(f"Failed to sync state from real results: {e}")
                raise TaskInfraError(
                    f"UNTESTED_TASK: update-state-from-real failed (session={sid}): {e}"
                ) from e

        return self.get_session_state(
            sid,
            timeout=request_timeout,
            retries=retries,
            backoff_sec=0.5,
        )
    
    def execute_tool_calls(self, session_id: str, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        key = self._task_lock_key(session_id=session_id)
        with self._single_flight(key):
            try:
                headers = {"X-Session-ID": session_id}
                response = self.session.post(
                    f"{self.base_url}/api/sessions/{session_id}/execute",
                    headers=headers,
                    json={"tool_calls": tool_calls},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()["results"]
            except Exception as e:
                logger.error(f"Failed to execute tool calls: {e}")
                raise TaskInfraError(
                    f"UNTESTED_TASK: execute failed (session={session_id}): {e}"
                ) from e
    
    def get_session_cost(self, session_id: str) -> Dict[str, float]:
        try:
            headers = {"X-Session-ID": session_id}
            response = requests.get(
                f"{self.base_url}/get-session-cost",
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            cost_data = response.json()
            return {
                "costs": cost_data.get("costs", {}),
                "total": cost_data.get("total", 0.0)
            }
        except Exception as e:
            logger.warning(f"Failed to get session costs: {e}")
            return {"costs": {}, "total": 0.0}

    def get_session_llm_usage(
        self,
        session_id: str,
        *,
        include_events: bool = False,
        limit: int = 10000,
        since_id: int = 0,
    ) -> Dict[str, Any]:
        """Get per-session LLM token usage recorded by the mock server."""
        try:
            headers = {"X-Session-ID": session_id}
            params = {
                "include_events": bool(include_events),
                "limit": int(limit),
                "since_id": int(since_id),
            }
            response = requests.get(f"{self.base_url}/get-session-llm-usage", headers=headers, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json() if response.content else {}
            if isinstance(payload, dict):
                return payload
            return {}
        except Exception as e:
            logger.warning(f"Failed to get session LLM usage: {e}")
            return {}

    @property
    def last_session_id(self) -> Optional[str]:
        """Expose last session_id for callers that need to reuse it."""
        return getattr(self, "_last_session_id", None)
    
    def set_session_state(
        self,
        session_id: str,
        state: Dict[str, Any],
        *,
        bootstrap_mode: str = "auto",
        timeout: Optional[float] = None,
        retries: int = 0,
        backoff_sec: float = 0.5,
    ) -> bool:
        headers = {"X-Session-ID": session_id}
        payload = {"state": state, "bootstrap_mode": bootstrap_mode}
        request_timeout = float(timeout if timeout is not None else self.timeout)
        max_attempts = 1
        last_error: Optional[Exception] = None

        key = self._task_lock_key(session_id=session_id)
        with self._single_flight(key):
            for attempt in range(1, max_attempts + 1):
                try:
                    response = self.session.post(
                        f"{self.base_url}/set-session-state",
                        headers=headers,
                        json=payload,
                        timeout=request_timeout,
                    )
                    response.raise_for_status()
                    logger.info(f"Set state for session {session_id}")
                    return True
                except Exception as e:
                    last_error = e
                    logger.error(
                        "Failed to set session state (timeout=%.1fs, session=%s): %s",
                        request_timeout,
                        session_id,
                        last_error,
                    )
                    raise TaskInfraError(
                        f"UNTESTED_TASK: set-session-state failed (session={session_id}, timeout={request_timeout:.1f}s): {last_error}"
                    ) from e
    
    def get_session_state(
        self,
        session_id: str,
        *,
        timeout: Optional[float] = None,
        retries: int = 0,
        backoff_sec: float = 0.5,
    ) -> Dict[str, Any]:
        headers = {"X-Session-ID": session_id}
        request_timeout = float(timeout if timeout is not None else self.timeout)
        max_attempts = 1
        last_error: Optional[Exception] = None

        key = self._task_lock_key(session_id=session_id)
        with self._single_flight(key):
            for attempt in range(1, max_attempts + 1):
                try:
                    response = self.session.get(
                        f"{self.base_url}/get-session-state",
                        headers=headers,
                        timeout=request_timeout,
                    )
                    response.raise_for_status()
                    return response.json().get("state", {})
                except Exception as e:
                    last_error = e
                    logger.error(
                        "Failed to get session state (timeout=%.1fs, session=%s): %s",
                        request_timeout,
                        session_id,
                        last_error,
                    )
                    raise TaskInfraError(
                        f"UNTESTED_TASK: get-session-state failed (session={session_id}, timeout={request_timeout:.1f}s): {last_error}"
                    ) from e
    
    def get_available_tools(self, test_case: TestCase, session_id: str = None) -> List[Dict[str, Any]]:
        try:
            from camel.toolkits import OpenAPIToolkit
            
            openapi_toolkit = OpenAPIToolkit()
            if session_id:
                openapi_toolkit.set_session_id(session_id)
            openapi_toolkit.set_override_server_url(self.base_url)
            
            tool_list = []
            
            test_id = TestCaseAdapter.get_id(test_case)
            if self._is_multi_turn_test(test_id):
                tool_list = self._load_multi_turn_tools(openapi_toolkit, test_case, session_id)
            else:
                tool_list = self._load_single_turn_tools(openapi_toolkit, test_case, session_id)
            
            return tool_list
            
        except Exception as e:
            logger.error(f"Failed to get tools: {e}")
            return []

    def _is_multi_turn_test(self, test_id: str) -> bool:
        return 'multi_turn' in test_id.lower()

    def _load_single_turn_tools(self, openapi_toolkit, test_case: TestCase, session_id: str) -> List[Dict[str, Any]]:
        tool_list = []
        test_id = TestCaseAdapter.get_id(test_case)
        compact_spec_path = self._get_single_turn_compact_spec_path(test_id)
        if os.path.exists(compact_spec_path):
            with open(compact_spec_path, 'r', encoding='utf-8') as f:
                openapi_json = json.load(f)

            api_name = openapi_json["info"]["title"]
            toolkit = openapi_toolkit.generate_openapi_funcs(api_name, openapi_json)
            schemas = openapi_toolkit.openapi_spec_to_openai_schemas(api_name, openapi_json)

            from camel.toolkits import FunctionTool
            tools = [FunctionTool(func=func, openai_tool_schema=schema)
                     for func, schema in zip(toolkit, schemas)]
            tool_list.extend(tools)
            return tool_list

        for func_idx, func in enumerate(TestCaseAdapter.get_functions(test_case)):
            api_spec_path = self._get_openapi_spec_path(test_id, func_idx)

            if os.path.exists(api_spec_path):
                with open(api_spec_path, 'r') as f:
                    openapi_json = json.load(f)

                api_name = openapi_json["info"]["title"]
                toolkit = openapi_toolkit.generate_openapi_funcs(api_name, openapi_json)
                schemas = openapi_toolkit.openapi_spec_to_openai_schemas(api_name, openapi_json)

                from camel.toolkits import FunctionTool
                tools = [FunctionTool(func=func, openai_tool_schema=schema)
                        for func, schema in zip(toolkit, schemas)]
                tool_list.extend(tools)
            else:
                logger.warning(f"OpenAPI spec not found: {api_spec_path}")
        
        return tool_list

    def _load_multi_turn_tools(self, openapi_toolkit, test_case: TestCase, session_id: str) -> List[Dict[str, Any]]:
        MULTI_TURN_TOOLKIT_MAPPING = {
            "GorillaFileSystem": "GorillaFileSystem",
            "PostingAPI": "TwitterAPI",
            "TwitterAPI": "TwitterAPI", 
            "MathAPI": "MathAPI",
            "MessageAPI": "MessageAPI",
            "TicketAPI": "TicketAPI",
            "TradingBot": "TradingBot",
            "TravelAPI": "TravelAPI",
            "TravelBooking": "TravelAPI",
            "VehicleControl": "VehicleControlAPI",
            "VehicleControlAPI": "VehicleControlAPI"
        }
        
        tool_list = []
        
        involved_classes = test_case.metadata.get('involved_classes', [])
        
        if not involved_classes:
            initial_config = test_case.metadata.get('initial_config', {})
            if initial_config:
                involved_classes = list(initial_config.keys())
        
        if not involved_classes:
            involved_classes = self._infer_involved_classes_from_test_case(test_case)
        
        test_id = TestCaseAdapter.get_id(test_case)
        logger.info(f"Multi-turn test {test_id} involved classes: {involved_classes}")
        
        for class_name in involved_classes:
            if class_name in MULTI_TURN_TOOLKIT_MAPPING:
                schema_name = MULTI_TURN_TOOLKIT_MAPPING[class_name]
                api_spec_path = self._get_multi_turn_spec_path(schema_name)
                
                if os.path.exists(api_spec_path):
                    logger.info(f"Loading multi-turn schema: {api_spec_path}")
                    with open(api_spec_path, 'r') as f:
                        openapi_json = json.load(f)
                    
                    api_name = openapi_json["info"]["title"]
                    toolkit = openapi_toolkit.generate_openapi_funcs(api_name, openapi_json)
                    schemas = openapi_toolkit.openapi_spec_to_openai_schemas(api_name, openapi_json)
                    
                    from camel.toolkits import FunctionTool
                    tools = [FunctionTool(func=func, openai_tool_schema=schema) 
                            for func, schema in zip(toolkit, schemas)]
                    tool_list.extend(tools)
                else:
                    logger.warning(f"Multi-turn OpenAPI spec not found: {api_spec_path}")
        
        return tool_list

    def _infer_involved_classes_from_test_case(self, test_case: TestCase) -> List[str]:
        if hasattr(test_case, 'involved_classes'):
            return TestCaseAdapter.get_involved_classes(test_case)
        
        return ["GorillaFileSystem", "TwitterAPI"]

    def _get_multi_turn_spec_path(self, schema_name: str) -> str:
        # Use local schema directory
        openapi_dir = "data/bfcl_v4/openapi"
        multi_turn_mock_path = os.path.join(openapi_dir, "multi_turn", "mock", f"{schema_name}.json")
        if os.path.exists(multi_turn_mock_path):
            return multi_turn_mock_path
        multi_turn_path = os.path.join(openapi_dir, "multi_turn", f"{schema_name}.json")
        if os.path.exists(multi_turn_path):
            return multi_turn_path
        return os.path.join(openapi_dir, f"{schema_name}.json")

    def execute_real_tools(self, test_case: TestCase, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        test_id = TestCaseAdapter.get_id(test_case)
        if not self._is_multi_turn_test(test_id):
            return []
        
        real_results = []
        
        try:
            tool_instances = self._initialize_real_tool_instances(test_case)
            
            for tool_call in tool_calls:
                function_name = tool_call.get('function', '')
                arguments = tool_call.get('arguments', {})
                
                result = self._execute_single_real_tool(tool_instances, function_name, arguments)
                
                real_results.append({
                    'function': function_name,
                    'arguments': arguments,
                    'result': result
                })
                
        except Exception as e:
            logger.error(f"Real tool execution failed: {e}")
            
        return real_results

    def _initialize_real_tool_instances(self, test_case: TestCase) -> Dict[str, Any]:
        # Multi-turn toolkit mapping
        MULTI_TURN_TOOLKIT_MAPPING = {
            "GorillaFileSystem": {
                "module": "benchmarks.bfcl.multi_turn.func_source_code.gorilla_file_system", 
                "class": "GorillaFileSystem"
            },
            "TwitterAPI": {
                "module": "benchmarks.bfcl.multi_turn.func_source_code.posting_api",
                "class": "TwitterAPI"  
            },
            "MathAPI": {
                "module": "benchmarks.bfcl.multi_turn.func_source_code.math_api",
                "class": "MathAPI"
            },
            "MessageAPI": {
                "module": "benchmarks.bfcl.multi_turn.func_source_code.message_api", 
                "class": "MessageAPI"
            },
            "TicketAPI": {
                "module": "benchmarks.bfcl.multi_turn.func_source_code.ticket_api",
                "class": "TicketAPI"
            },
            "TradingBot": {
                "module": "benchmarks.bfcl.multi_turn.func_source_code.trading_bot",
                "class": "TradingBot"
            },
            "TravelAPI": {
                "module": "benchmarks.bfcl.multi_turn.func_source_code.travel_booking",
                "class": "TravelAPI"
            },
            "VehicleControlAPI": {
                "module": "benchmarks.bfcl.multi_turn.func_source_code.vehicle_control",
                "class": "VehicleControlAPI"
            }
        }
        
        self._patch_dependencies()
        
        tool_instances = {}
        
        initial_config = test_case.metadata.get('initial_config', {})
        involved_classes = test_case.metadata.get('involved_classes', [])
        
        if not involved_classes and initial_config:
            involved_classes = list(initial_config.keys())
        
        logger.info(f"Initializing real tools - involved_classes: {involved_classes}, initial_config keys: {list(initial_config.keys())}")
        
        for class_name in involved_classes:
            if class_name in MULTI_TURN_TOOLKIT_MAPPING:
                mapping = MULTI_TURN_TOOLKIT_MAPPING[class_name]
                try:
                    module = __import__(mapping['module'])
                    tool_class = getattr(module, mapping['class'])
                    
                    instance = tool_class()
                    
                    if hasattr(instance, '_load_scenario') and class_name in initial_config:
                        instance._load_scenario(initial_config[class_name])
                    
                    tool_instances[class_name] = instance
                    logger.info(f"Initialized real tool instance: {class_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize {class_name}: {e}")
                    import traceback
                    logger.error(f"Traceback for {class_name}: {traceback.format_exc()}")
        
        return tool_instances

    def _patch_dependencies(self):
        import sys
        import types
        
        if 'benchmarks.bfcl.multi_turn.func_source_code.long_context' not in sys.modules:
            long_context_module = types.ModuleType('long_context')
            long_context_module.FILE_CONTENT_EXTENSION = "Sample content..."
            long_context_module.FILES_TAIL_USED = ['log.txt', 'report.txt']
            long_context_module.POPULATE_FILE_EXTENSION = ['image1.jpg', 'image2.jpg']
            sys.modules['benchmarks.bfcl.multi_turn.func_source_code.long_context'] = long_context_module

    def _execute_single_real_tool(self, tool_instances: Dict[str, Any], function_name: str, arguments: Dict[str, Any]) -> Any:
        clean_function_name = function_name
        prefix_candidates = {
            "gorillafilesystem",
            "postingapi",
            "twitterapi",
            "messageapi",
            "ticketapi",
            "mathapi",
            "tradingbot",
            "travelapi",
            "travelbooking",
            "vehiclecontrolapi",
            "vehiclecontrol",
        }
        if isinstance(function_name, str) and "_" in function_name:
            prefix = function_name.split("_", 1)[0].lower()
            if prefix in prefix_candidates:
                clean_function_name = function_name.split("_", 1)[1]
        
        function_mapping = {
            'authenticatetwitter': 'authenticate_twitter',
            'postinggetloginstatus': 'get_login_status'
        }
        
        if clean_function_name in function_mapping:
            clean_function_name = function_mapping[clean_function_name]
        
        logger.debug(f"Looking for method '{clean_function_name}' (original: {function_name})")
        
        for class_name, instance in tool_instances.items():
            if hasattr(instance, clean_function_name):
                method = getattr(instance, clean_function_name)
                try:
                    if isinstance(arguments, dict) and 'requestBody' in arguments:
                        kwargs = arguments['requestBody']
                    elif isinstance(arguments, dict):
                        kwargs = arguments
                    else:
                        kwargs = {}

                    logger.debug(f"Executing {clean_function_name} with args: {kwargs}")
                    
                    result = method(**kwargs) if kwargs else method()
                    return result
                except Exception as e:
                    logger.error(f"Error executing {clean_function_name}: {e}")
                    return {"error": str(e)}
        
        available_methods = []
        for class_name, instance in tool_instances.items():
            methods = [attr for attr in dir(instance) if not attr.startswith('_') and callable(getattr(instance, attr))]
            available_methods.extend(f"{class_name}.{method}" for method in methods)
        
        error_msg = f"Method '{clean_function_name}' not found in any tool instance. Available methods: {available_methods}"
        logger.error(error_msg)
        return {"error": error_msg}

    @staticmethod
    def normalize_arguments_to_schema(
        arguments: Dict[str, Any],
        schema: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Normalize tool arguments to match an object schema without inventing values."""
        if not isinstance(arguments, dict):
            return {}
        if not isinstance(schema, dict) or not schema:
            return dict(arguments)

        if "requestBody" in arguments and isinstance(arguments["requestBody"], dict):
            normalized_body = MockServerClient.normalize_arguments_to_schema(
                arguments["requestBody"],
                schema,
            )
            normalized = dict(arguments)
            normalized["requestBody"] = normalized_body
            return normalized

        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return dict(arguments)

        normalized: Dict[str, Any] = {}
        extras: Dict[str, Any] = {}

        for key, value in arguments.items():
            if key not in properties:
                extras[key] = value
                continue

            child_schema = properties.get(key)
            if isinstance(value, dict) and isinstance(child_schema, dict):
                normalized[key] = MockServerClient.normalize_arguments_to_schema(
                    value,
                    child_schema,
                )
            else:
                normalized[key] = value

        candidate_assignments: List[tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        for key, child_schema in properties.items():
            if key in normalized or not isinstance(child_schema, dict):
                continue

            child_props = child_schema.get("properties")
            accepted: Dict[str, Any] = {}

            if isinstance(child_props, dict) and child_props:
                for extra_key, extra_value in extras.items():
                    if extra_key in child_props:
                        accepted[extra_key] = extra_value
            elif child_schema.get("type") == "object" and child_schema.get("additionalProperties") is not False:
                accepted = dict(extras)

            if accepted:
                candidate_assignments.append((key, accepted, child_schema))

        if len(candidate_assignments) == 1:
            nested_key, accepted, child_schema = candidate_assignments[0]
            for extra_key in accepted:
                extras.pop(extra_key, None)
            normalized[nested_key] = MockServerClient.normalize_arguments_to_schema(
                accepted,
                child_schema,
            )

        normalized.update(extras)
        return normalized
    
    def get_tool_calls_from_session(self, session_id: str, previous_count: int = 0) -> List[Dict[str, Any]]:
        """
        Get executed tool calls from mock server session history.
        
        Args:
            session_id: Session ID
            previous_count: Number of tool calls already processed in previous turns
            
        Returns:
            List of NEW tool calls with their results (only from current turn)
        """
        try:
            response = self.session.get(
                f"{self.base_url}/get-session-history", 
                headers={"X-Session-ID": session_id}
            )
            response.raise_for_status()
            session_history = response.json()
            
            tool_calls = []
            # Built-in API endpoints that should not be treated as tool calls
            builtin_apis = [
                "/get-session-state", 
                "/get-session-history", 
                "/set-session-state", 
                "/get-session-cost",
                "/update_state",
                "/session-id"  # GET endpoint, but include for completeness
            ]
            
            for request in session_history["history"]:
                # Exclude built-in tool calls
                if request["request"]["path"] not in builtin_apis:
                    # Add tool call with 200 status code
                    if request["response"]["status_code"] == 200:
                        body = request["response"]["body"]
                        
                        # Parse JSON response body if it's a string
                        if isinstance(body, str):
                            try:
                                import json
                                body = json.loads(body)
                            except json.JSONDecodeError:
                                # Keep as string if not valid JSON
                                pass
                        
                        # Only use template if body is None or empty
                        if body is None or body == "":
                            body = f"Simulated successful response for {request['request']['path']}"
                        
                        tool_calls.append({
                            "name": request["request"]["path"], 
                            "arguments": request["request"]["body"], 
                            "result": body
                        })
            
            # Only return new tool calls from current turn
            new_tool_calls = tool_calls[previous_count:] if previous_count < len(tool_calls) else []
            return new_tool_calls
        except Exception as e:
            logger.error(f"Failed to get tool calls from session: {e}")
            return []
    
    def _map_tool_names(self, tool_calls: List[Dict[str, Any]], function_schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map OpenAPI tool names back to original function names.
        
        Args:
            tool_calls: Tool calls from session history
            function_schemas: Original function schemas from test case
            
        Returns:
            Tool calls with mapped function names
        """
        # Create mapping from OpenAPI names to original names
        name_mapping = {}
        for func_schema in function_schemas:
            original_name = func_schema['name']
            # The OpenAPI name pattern is typically: {test_id}_{func_idx}post_{function_name}
            # We'll try to match based on function name suffix
            for tool_call in tool_calls:
                openapi_name = tool_call.get('name', '').lstrip('/')
                if openapi_name.endswith(original_name):
                    name_mapping[openapi_name] = original_name
                    break
        
        # Map the tool call names
        mapped_calls = []
        for tool_call in tool_calls:
            mapped_call = tool_call.copy()
            openapi_name = tool_call.get('name', '').lstrip('/')
            if openapi_name in name_mapping:
                mapped_call['function'] = name_mapping[openapi_name]
            else:
                # Fallback: try to extract function name from path
                mapped_call['function'] = openapi_name.split('_')[-1] if '_' in openapi_name else openapi_name
            
            # Ensure consistent format
            if 'name' in mapped_call:
                del mapped_call['name']
            
            mapped_calls.append(mapped_call)
        
        return mapped_calls
    
    def _get_openapi_spec_path(self, test_id: str, func_idx: int) -> str:
        # Use local schema directory
        openapi_dir = "data/bfcl_v4/openapi"
        return os.path.join(openapi_dir, f"{test_id}_{func_idx}.json")

    def _get_single_turn_compact_spec_path(self, test_id: str) -> str:
        """Get per-task single-turn OpenAPI spec path using compact BFCL schema names."""
        openapi_dir = os.path.join("data/bfcl_v4/openapi", "single_turn")
        return os.path.join(openapi_dir, f"{derive_single_turn_schema_name(test_id)}.json")


class ClientInferenceEngine:
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.benchmark = get_bfcl_benchmark()
        self.mock_client = MockServerClient(config.mock_server_url)
        
        self._init_agent()
        
        logger.info(f"Client Inference Engine initialized with model: {config.model_name}")
    
    def _init_agent(self):
        try:
            from .agents.chat_agent import ChatAgent
            self.agent = ChatAgent(
                model_name=self.config.model_name,
                timeout=self.config.agent_timeout,
                system_message=self.config.system_message
            )
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def run_single_turn(self, test_id: str, config: Optional[InferenceConfig] = None) -> InferenceResult:
        if config:
            self.config = config
        
        start_time = time.time()
        
        try:
            test_case = self.benchmark.load_test_case(test_id)
            
            session_id = self.mock_client.create_session(test_case)
            self._current_session_id = session_id
            
            available_tools = self.mock_client.get_available_tools(test_case, session_id)
            
            if hasattr(self.agent, 'set_tools') and available_tools:
                self.agent.set_tools(available_tools)
            
            turn_result = self._run_single_turn_with_retry(
                test_case, session_id, available_tools, 0
            )
            
            result = InferenceResult(
                test_id=test_id,
                success=turn_result.success,
                turns=[turn_result],
                overall_success=turn_result.success,
                final_score=turn_result.judge_score,
                total_time=time.time() - start_time,
                mock_tool_calls=turn_result.mock_tool_calls,
                real_tool_calls=turn_result.real_tool_calls,
                judge_score=turn_result.judge_score
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Single turn execution failed for {test_id}: {e}")
            return InferenceResult(
                test_id=test_id,
                success=False,
                total_time=time.time() - start_time
            )
    
    def run_multi_turn(self, test_id: str, config: Optional[InferenceConfig] = None) -> InferenceResult:
        if config:
            self.config = config
        
        try:
            test_case = self.benchmark.load_test_case(test_id)
            
            from .multi_turn_executor import MultiTurnExecutor
            executor = MultiTurnExecutor(self.config)
            
            multi_turn_result = executor.execute_multi_turn(test_case)
            
            turns = []
            for turn in multi_turn_result.turns:
                inference_turn = TurnResult(
                    turn_idx=turn.turn_idx,
                    success=turn.success,
                    mock_tool_calls=[],
                    real_tool_calls=turn.real_tool_calls,
                    execution_time=turn.execution_time,
                    judge_score=turn.judge_score,
                    error=turn.error
                )
                turns.append(inference_turn)
            
            result = InferenceResult(
                test_id=test_id,
                success=multi_turn_result.success,
                turns=turns,
                overall_success=multi_turn_result.success,
                final_score=multi_turn_result.final_score,
                total_time=multi_turn_result.total_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Multi-turn execution failed for {test_id}: {e}")
            import traceback
            traceback.print_exc()
            return InferenceResult(
                test_id=test_id,
                success=False,
                turns=[],
                overall_success=False,
                final_score=0.0,
                total_time=0.0
            )
    
    def _run_single_turn_with_retry(self, test_case: TestCase, session_id: str, 
                                  available_tools: List[Dict], turn_idx: int,
                                  conversation_history: Optional[List[Dict]] = None) -> TurnResult:
        
        for attempt in range(self.config.max_retries + 1):
            try:
                turn_start = time.time()
                
                questions = TestCaseAdapter.get_questions(test_case)
                if turn_idx < len(questions):
                    current_turn_messages = questions[turn_idx]
                    if current_turn_messages:
                        current_question = current_turn_messages[0]
                    else:
                        test_id = TestCaseAdapter.get_id(test_case)
                        logger.warning(f"Turn {turn_idx} has no messages for test {test_id}")
                        return TurnResult(turn_idx, False, error="No messages in turn")
                else:
                    test_id = TestCaseAdapter.get_id(test_case)
                    logger.warning(f"Turn {turn_idx} out of range for test {test_id}")
                    return TurnResult(turn_idx, False, error="Turn index out of range")
                
                messages = []
                if conversation_history:
                    messages.extend(conversation_history)
                messages.append(current_question)
                
                prompt = self._build_prompt(messages, TestCaseAdapter.get_functions(test_case))
                
                response = self.agent.generate_response(
                    prompt=prompt,
                    context={
                        "available_tools": available_tools,
                        "temperature": 0.1
                    }
                )
                
                expected_tool_calls = self.agent.parse_tool_calls(response.raw_response or "")
                
                previous_count = sum(len(turn.mock_tool_calls) for turn in [])
                actual_tool_calls = self.mock_client.get_tool_calls_from_session(session_id, previous_count)
                
                mapped_tool_calls = self.mock_client._map_tool_names(actual_tool_calls, TestCaseAdapter.get_functions(test_case))
                
                real_results = []
                if expected_tool_calls:
                    real_results = self.mock_client.execute_tool_calls(session_id, expected_tool_calls)
                
                judge_score = self._calculate_judge_score(
                    test_case, turn_idx, mapped_tool_calls, real_results
                )
                
                success = bool(mapped_tool_calls) and judge_score >= self.config.target_score
                
                if success or attempt == self.config.max_retries:
                    return TurnResult(
                        turn_idx=turn_idx,
                        success=success,
                        mock_tool_calls=mapped_tool_calls,
                        real_tool_calls=real_results,
                        execution_time=time.time() - turn_start,
                        judge_score=judge_score
                    )
                
                logger.info(f"Turn {turn_idx} attempt {attempt + 1} failed, retrying...")
                
            except Exception as e:
                logger.error(f"Turn {turn_idx} attempt {attempt + 1} error: {e}")
                if attempt == self.config.max_retries:
                    return TurnResult(
                        turn_idx=turn_idx,
                        success=False,
                        execution_time=time.time() - turn_start if 'turn_start' in locals() else 0,
                        error=str(e)
                    )
        
        return TurnResult(turn_idx, False, error="Max retries exceeded")
    
    def _build_prompt(self, messages: List[Dict], functions: List[Dict]) -> str:
        prompt_parts = []
        
        if functions:
            prompt_parts.append("Available functions:")
            for func in functions:
                func_desc = f"- {func['name']}: {func.get('description', '')}"
                if 'parameters' in func:
                    func_desc += f"\n  Parameters: {func['parameters']}"
                prompt_parts.append(func_desc)
            prompt_parts.append("")
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            prompt_parts.append(f"{role.title()}: {content}")
        
        return "\n".join(prompt_parts)
    
    def _calculate_judge_score(self, test_case: TestCase, turn_idx: int, 
                             tool_calls: List[Dict], results: List[Dict]) -> float:
        if not self.config.enable_judge:
            return 1.0 if tool_calls else 0.0
        
        try:
            test_id = TestCaseAdapter.get_id(test_case)
            evaluation = self.benchmark.evaluate_turn(
                test_id, turn_idx, tool_calls, results
            )
            return evaluation.score if hasattr(evaluation, 'score') else 0.0
        except Exception as e:
            logger.warning(f"Judge evaluation failed: {e}")
            return 1.0 if tool_calls else 0.0


_client_engine_instance = None

def get_client_inference_engine(config: InferenceConfig) -> ClientInferenceEngine:
    global _client_engine_instance
    if _client_engine_instance is None or _client_engine_instance.config != config:
        _client_engine_instance = ClientInferenceEngine(config)
    return _client_engine_instance

get_inference_engine = get_client_inference_engine

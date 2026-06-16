import asyncio
import json
import logging
import random
from datetime import datetime
from copy import deepcopy
from typing import Any, Dict, List, Optional

from camel.agents import ChatAgent
from fastapi import Request
from inference.utils.log_context import bind_log_context
from pydantic import BaseModel, ConfigDict, Field

from .request_details import RequestDetails
from .config_updater import update_state
from .config_updater import classify_tool_call_status
from .schema_utils import (
    resolve_refs,
    extract_parameter_descriptions,
    extract_response_descriptions,
    extract_toolkit_info,
)
from utils.model_utils import create_model, sanitize_llm_json_text
import json_repair

logger = logging.getLogger(__name__)

LLM_AGENT_TIMEOUT_SECONDS = 600.0


RESPONSE_SYSTEM_PROMPT = """
You are an API simulation engine that generates JSON responses strictly following OpenAPI 3.1 schemas.

CORE PRINCIPLES (in priority order):

**1. MANDATORY Brevity — Hard Output Limit** — This is a simulation, not a real service.
  - Every string value: at most 200 characters. No exceptions.
  - Biological sequences (DNA, RNA, protein, amino acids): always return exactly a short sample like "ATCGATCGATCG" or "MVLSPADKTN" regardless of requested length.
  - Any other generated blob (code, logs, text, documents, binary): return a very short placeholder.
  - **Arrays/lists: return exactly 30 items maximum, even if the schema says "length = N years", "one per year", "one per item", or similar. You are simulating, not producing real data. Ignore any array length hints in descriptions.**
  - For numeric metadata fields that describe size (length, count, sequence_length, etc.): set them to the value the REQUEST asked for, NOT the length of your truncated output.
  - NEVER expand content to match a requested length/count. The request may say length=500 or years=20 — you still output at most 30 array items and at most 200 characters per string.

**2. Schema Adherence** — Match the schema exactly (structure, names, types, formats, required fields).

**3. No Extra Rules** — Do not invent constraints beyond the tool definition.

**4. Conversation-Gated Preconditions** — Requirements such as user confirmation,
consent, approval, or permission are agent/judge responsibilities unless the
OpenAPI request schema exposes an explicit argument or state field for that
gate. Do not return a "confirmation required" or "permission required" error
solely because the current API request payload does not show conversation
history. Validate concrete request arguments and system state instead.
"""

STATE_FOLLOWING_SYSTEM_PROMPT = """
**System State Validation** — Validate all operations against the current System State (including toolkit-specific runtime_state). If requirements are not met, return an error.

**Semantic Validation** — Referenced entities must exist, be accessible, and be valid for the requested operation.

**Reasonable Defaults** — If schema-required values are missing from System State, synthesize realistic values (UUIDs, correctly formatted timestamps, tokens, positive balances, etc.) that do not contradict System State. Do not use 0/null/empty placeholders when the contract says a generated value has a nonzero range or specific format.

**State Consistency** — Reflect mutations consistently; subsequent operations must observe prior successful changes.

STATE PRIORITY RULES:

- Apply this priority when deciding values and constraints:
  1) Current System State (latest runtime/config state)
  2) Request + Operation context
  3) Schema Default State (`info.x-default-state`) where operation-specific defaults override global defaults
  4) Reasonable synthesized values
- If Current System State conflicts with `x-default-state`, trust Current System State.
- If `x-default-state` provides fixed defaults or validation-like constraints, follow them unless overridden by higher-priority state.
- Canonical business truth is top-level toolkit state (`<ToolkitName>.*`); `runtime_state.toolkits.<ToolkitName>` is runtime context only.
- `real_tool_calls` entries marked `state_role=observed_state` are part of Current System State. Use exact lookup observations for the matching entity/key; treat search/list observations as observed candidates, not complete databases.
- If the same business key appears in both top-level toolkit state and runtime_state with conflicting values, trust top-level toolkit state.


VALIDATION GUIDELINES:

8. **Exact Matching** — Entity names, identifiers, and paths must match EXACTLY. Similar names are NOT the same (e.g., 'user_123' ≠ 'user_124').

9. **Navigating Nested Structures** — When checking if an entity exists in a nested structure:
   a) Identify the relevant path/location from runtime_state (may be nested under runtime_state.toolkits.<ToolkitName> or flat)
   b) Parse the path into components if needed
   c) Navigate step-by-step through the structure, following the nesting pattern (e.g., parent → .contents → child → .contents)
   d) Check existence at the final level only - do not assume entities from parent/sibling/child levels

10. **Scope Boundaries** — Operations with scope constraints (e.g., "current directory", "active workspace", "selected items") can ONLY access direct members of that scope, not nested or related scopes.

11. **Case Sensitivity** — All identifiers, names, and keys are case-sensitive unless explicitly stated otherwise in the tool definition.

12. **Schema Branch Selection** — If the response schema uses oneOf/anyOf, choose EXACTLY ONE branch and output a concrete instance of that branch. Never output the schema itself (no oneOf/anyOf/type/properties/description in the response).
13. **Condition Evaluation Discipline** — Treat validation rules as executable conditions:
   - Trigger an error branch only when the condition is positively true from request + state + schema defaults.
   - If a condition is not provably true, do not assume failure.
   - If all known failure conditions are false, prefer the success branch.
14. **Use Called-Method Static Data** — When x-default-state provides called_method_static_data/static_data (e.g., canonical lists, lookup tables), use it as authoritative for validations instead of guessing.
**Static Data Discipline** — If Schema Default State includes operation_defaults.static_data, treat it as authoritative contract data.
- For lookup operations, use the exact request fields to form the documented lookup key.
- If the key exists in static_data, return a success response based on that value.
- Do not return an empty list, null, or "not found" for a key that exists in static_data.
- Only return empty/error when the key is absent or a validation rule is triggered.
15. **Auth/Login Continuity** — After a successful auth/login response, treat the toolkit as authenticated for subsequent operations unless a later explicit logout or contract-declared error-state effect changes that state. A failed login/auth response is response-only unless x-default-state explicitly says it mutates durable auth state.
16. **Write Success Consistency** — For successful write operations, ensure response semantics are consistent with persisted canonical state mutations (e.g., created records are retrievable by returned IDs).
17. **Contract Shape Separation** — When x-default-state provides persistent_state_shapes or response_only_fields, use them to distinguish durable task state from fields that appear only in the response payload.

Example: For a hierarchical toolkit with current directory "/root/folder/subfolder", checking if "notes.md" exists:
- Navigate through the directory tree encoded in state, inserting `.contents` between directory nodes as needed.
- Check: "notes.md" exists as a direct key in the current directory node.
- Scope: Only direct children of the current directory are accessible, not files in parent or subdirectories.
"""


CONTEXT_EXTRACTION_SYSTEM_PROMPT = """
You are a state extraction assistant. Your task is to extract and summarize the relevant system state for an API operation.

Your role:
1. Read the current system state from the configuration
2. Identify what parts of the state are relevant to this specific operation
3. Extract and clearly present this information
4. DO NOT generate responses or results - only extract state

Key principles:
- Prefer canonical top-level toolkit state for business fields; use runtime_state only for transient context.
- If top-level and runtime_state conflict on business fields (e.g., authenticated flags, counters, records), treat top-level as authoritative.
- For file-system existence checks, enforce cwd strictly: if an argument is a local name (no path separators), it must exist as a direct child of the current working directory; existence only in descendant subdirectories does NOT count.
    Example: if cwd is `/current/folder`, `report.csv` exists only at `/current/folder/archive/report.csv`, and the call uses local name `report.csv`, then treat it as NOT existing in cwd (invalid for current-directory-only operations).
- For operations that access/modify resources, identify those resources' current state
- For operations with source/destination, check BOTH locations
- Always specify if collections/directories are EMPTY or list their contents
- If state contains `real_tool_calls` entries with `state_role=observed_state`, treat them as observed task state, not auxiliary logs. Exact lookup observations are authoritative for the matching arguments; search/list observations are returned candidates, not exhaustive databases.
- Include any constraints or validation rules from the operation description
- If x-default-state static_data contains a lookup table relevant to the request, explicitly state the constructed lookup key, whether it exists, and the matched value.
- Focus on what EXISTS vs what DOESN'T EXIST in the relevant scope

DO NOT evaluate validation rules and DO NOT decide whether the operation should succeed or fail. Validation rules from x-default-state will be passed to the responding model directly; your job is only to surface the state facts the responding model needs to evaluate them. Do not output triggered/passed/failed verdicts.

Use the requested structured format. Emit every field; use [] when empty.
- relevant_system_state: concise facts, including verbatim values read by the op.
- operation_constraints: relevant constraints from the operation description (state them; do not judge whether they are satisfied).

Be thorough but concise. Extract ONLY what's needed for this specific operation.
"""


class ContextExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevant_system_state: List[str] = Field(
        description="Relevant state facts for this operation.",
    )
    operation_constraints: List[str] = Field(
        description="Relevant operation constraints stated verbatim (no satisfaction judgment).",
    )


class ResponseGenerator:
    """Generates mock responses based on OpenAPI schema using LLM."""

    def __init__(
        self,
        response_model: str = "gpt-4.1-mini",
        state_model: Optional[str] = "gpt-4.1-mini",
        validation_model: str = "gpt-4.1-mini",
    ):
        """Initialize the response generator with configurable models.

        Args:
            response_model: LLM model for response generation (default: gpt-4.1-mini)
            state_model: LLM model for state update (default: gpt-4.1-mini)
            validation_model: LLM model for request validation (also used by context extractor)
        """
        self.response_model = response_model
        if isinstance(state_model, str) and state_model.strip().lower() in {"none", "null", ""}:
            self.state_model = None
        else:
            self.state_model = state_model
        self.validation_model = validation_model
        if self.state_model is None:
            self.system_prompt = RESPONSE_SYSTEM_PROMPT
        else:
            self.system_prompt = (
                RESPONSE_SYSTEM_PROMPT.rstrip()
                + "\n\n"
                + STATE_FOLLOWING_SYSTEM_PROMPT.strip()
            )

    @staticmethod
    def _response_content(response: Any) -> str:
        msg = getattr(response, "msg", None)
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        return str(content)

    @staticmethod
    def extract_toolkit_runtime_state(config: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extract runtime state for the specific toolkit from config.

        Supports both new nested structure (runtime_state.toolkits.<name>) and
        old flat structure (runtime_state) for backward compatibility.

        Args:
            config: Full system configuration
            schema: OpenAPI schema containing toolkit info

        Returns:
            Dictionary of runtime state variables for this toolkit
        """
        runtime_state = config.get('runtime_state', {})

        toolkit_name = None
        if schema and isinstance(schema, dict):
            info = schema.get('info', {})
            toolkit_name = info.get('title', '')

        if 'toolkits' in runtime_state and toolkit_name:
            toolkit_state = runtime_state.get('toolkits', {}).get(toolkit_name, {})
            if toolkit_state:
                return toolkit_state

        if 'toolkits' not in runtime_state:
            return runtime_state

        return {}

    @staticmethod
    def _operation_description_map(schema: Dict[str, Any]) -> Dict[str, str]:
        """Build operationId -> description map from schema paths."""
        descriptions: Dict[str, str] = {}
        if not isinstance(schema, dict):
            return descriptions
        paths = schema.get("paths", {})
        if not isinstance(paths, dict):
            return descriptions

        for methods in paths.values():
            if not isinstance(methods, dict):
                continue
            for operation in methods.values():
                if not isinstance(operation, dict):
                    continue
                operation_id = operation.get("operationId")
                if not isinstance(operation_id, str) or not operation_id:
                    continue
                description = operation.get("description")
                descriptions[operation_id] = description if isinstance(description, str) else ""
        return descriptions

    @staticmethod
    def _toolkit_name_from_schema(schema: Dict[str, Any]) -> str:
        if not isinstance(schema, dict):
            return ""
        info = schema.get("info", {})
        if not isinstance(info, dict):
            return ""
        title = info.get("title")
        return title.strip() if isinstance(title, str) else ""

    def _extract_schema_default_state(
        self,
        schema: Dict[str, Any],
        operation: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract operation-filtered x-default-state in NEW format only.

        Expected schema shape:
        info.x-default-state = {
          "global": {"runtime_defaults": {...}},
          "tools": {"<operationId>": {...}}
        }
        """
        if not isinstance(schema, dict):
            return {}
        info = schema.get("info", {})
        if not isinstance(info, dict):
            return {}
        x_default_state = info.get("x-default-state")
        if not isinstance(x_default_state, dict):
            return {}

        global_block = x_default_state.get("global")
        tools_block = x_default_state.get("tools")
        if not isinstance(global_block, dict) or not isinstance(tools_block, dict):
            return {}

        runtime_defaults = global_block.get("runtime_defaults")
        global_runtime_defaults = runtime_defaults if isinstance(runtime_defaults, dict) else {}

        operation_defaults: Dict[str, Any] = {}
        operation_id = operation.get("operationId") if isinstance(operation, dict) else ""
        if isinstance(operation_id, str) and operation_id:
            tool_entry = tools_block.get(operation_id)
            if isinstance(tool_entry, dict):
                static_data = tool_entry.get("static_data")
                if isinstance(static_data, dict) and static_data:
                    operation_defaults["static_data"] = static_data

                validation_rules = tool_entry.get("validation_rules")
                if isinstance(validation_rules, list) and validation_rules:
                    operation_defaults["validation_rules"] = validation_rules

                state_access = tool_entry.get("state_access")
                if isinstance(state_access, str) and state_access.strip():
                    operation_defaults["state_access"] = state_access.strip().lower()

                state_effects = tool_entry.get("state_effects")
                if isinstance(state_effects, list) and state_effects:
                    operation_defaults["state_effects"] = state_effects

                state_effects_on_success = tool_entry.get("state_effects_on_success")
                if isinstance(state_effects_on_success, list) and state_effects_on_success:
                    operation_defaults["state_effects_on_success"] = state_effects_on_success

                state_effects_on_error = tool_entry.get("state_effects_on_error")
                if isinstance(state_effects_on_error, list) and state_effects_on_error:
                    operation_defaults["state_effects_on_error"] = state_effects_on_error

                state_effects_always = tool_entry.get("state_effects_always")
                if isinstance(state_effects_always, list) and state_effects_always:
                    operation_defaults["state_effects_always"] = state_effects_always

                behavior_hints = tool_entry.get("behavior_hints")
                if isinstance(behavior_hints, list) and behavior_hints:
                    operation_defaults["behavior_hints"] = behavior_hints

                persistent_state_shapes = tool_entry.get("persistent_state_shapes")
                if isinstance(persistent_state_shapes, list) and persistent_state_shapes:
                    operation_defaults["persistent_state_shapes"] = persistent_state_shapes

                response_only_fields = tool_entry.get("response_only_fields")
                if isinstance(response_only_fields, list) and response_only_fields:
                    operation_defaults["response_only_fields"] = response_only_fields

                response_variants = tool_entry.get("response_variants")
                if isinstance(response_variants, list) and response_variants:
                    operation_defaults["response_variants"] = response_variants

                success_string_templates = tool_entry.get("success_string_templates")
                if isinstance(success_string_templates, dict) and success_string_templates:
                    operation_defaults["success_string_templates"] = success_string_templates

                called_method_static_data = tool_entry.get("called_method_static_data")
                if isinstance(called_method_static_data, dict) and called_method_static_data:
                    operation_defaults["called_method_static_data"] = called_method_static_data

                method_calls = tool_entry.get("method_calls")
                if isinstance(method_calls, list) and method_calls:
                    operation_defaults["method_calls"] = [
                        method_name
                        for method_name in method_calls
                        if isinstance(method_name, str) and method_name.strip()
                    ]
                    op_desc_map = self._operation_description_map(schema)
                    method_call_descriptions = []
                    for method_name in method_calls:
                        if isinstance(method_name, str):
                            desc = op_desc_map.get(method_name, "").strip()
                            if desc:
                                method_call_descriptions.append(desc)
                    if method_call_descriptions:
                        operation_defaults["method_call_descriptions"] = method_call_descriptions

        result: Dict[str, Any] = {}
        if global_runtime_defaults:
            result["global_runtime_defaults"] = global_runtime_defaults
        if operation_defaults:
            result["operation_defaults"] = operation_defaults
        return result

    def _materialize_missing_toolkit_state(
        self,
        current_state: Dict[str, Any],
        schema: Dict[str, Any],
        schema_default_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """If toolkit top-level state is missing, seed it from schema defaults."""
        if not isinstance(current_state, dict):
            return {}
        toolkit_name = self._toolkit_name_from_schema(schema)
        if not toolkit_name:
            return current_state

        defaults = schema_default_state.get("global_runtime_defaults")
        if not isinstance(defaults, dict) or not defaults:
            return current_state

        runtime_toolkit = (
            current_state.get("runtime_state", {})
            .get("toolkits", {})
            .get(toolkit_name, {})
        )

        existing = current_state.get(toolkit_name)
        if isinstance(existing, dict):
            merged = deepcopy(current_state)
            toolkit_state = merged.get(toolkit_name)
            if not isinstance(toolkit_state, dict):
                merged[toolkit_name] = deepcopy(defaults)
                return merged
            for key, value in defaults.items():
                if key not in toolkit_state:
                    if isinstance(runtime_toolkit, dict) and key in runtime_toolkit:
                        toolkit_state[key] = deepcopy(runtime_toolkit[key])
                    else:
                        toolkit_state[key] = deepcopy(value)
            return merged

        merged = deepcopy(current_state)
        merged[toolkit_name] = deepcopy(defaults)
        logger.info(
            "[RESPONSE] Seeded missing toolkit state from x-default-state defaults: %s",
            toolkit_name,
        )
        return merged

    @staticmethod
    def _prompt_safe_schema_default_state(schema_default_state: Dict[str, Any]) -> Dict[str, Any]:
        """Strip gecko-internal-only fields before exposing schema_default_state to LLMs.

        `global_runtime_defaults` carries init_rules / source_method / init_from —
        metadata about how gecko seeds initial state at session start. It is
        consumed internally by `_materialize_missing_toolkit_state`. The
        response_llm and ctx_extractor read state from runtime_state directly
        and gain nothing from these init hints; including them can mislead the
        model.
        """
        if not isinstance(schema_default_state, dict):
            return schema_default_state
        return {k: v for k, v in schema_default_state.items() if k != "global_runtime_defaults"}

    @staticmethod
    def _format_schema_default_state(schema_default_state: Dict[str, Any]) -> str:
        prompt_view = ResponseGenerator._prompt_safe_schema_default_state(schema_default_state)
        if not prompt_view:
            return "(No operation-relevant x-default-state found)"
        try:
            return json.dumps(prompt_view, indent=2, ensure_ascii=False)
        except Exception:
            return str(prompt_view)

    @staticmethod
    def _clone_rng(rng: random.Random) -> random.Random:
        clone = random.Random()
        clone.setstate(rng.getstate())
        return clone

    @classmethod
    def _travel_rng_from_visible_state(cls, travel_state: Dict[str, Any]) -> random.Random:
        seed = travel_state.get("random_seed", 141053)
        try:
            seed = int(seed)
        except Exception:
            seed = 141053
        rng = random.Random(seed)

        # The BFCL source keeps a single Random instance. Gecko state only stores
        # visible business fields, so reconstruct consumed draws from generated
        # identifiers that are observable in state.
        probe = cls._clone_rng(rng)
        generated_access_token = str(probe.randint(100000, 999999))
        if str(travel_state.get("access_token", "")) == generated_access_token:
            rng = probe

        credit_cards = travel_state.get("credit_card_list")
        if not isinstance(credit_cards, dict):
            credit_cards = {}
        booking_record = travel_state.get("booking_record")
        if not isinstance(booking_record, dict):
            booking_record = {}

        while True:
            probe = cls._clone_rng(rng)
            generated_card_id = str(probe.randint(100000000000, 999999999999))
            probe.randint(10000, 99999)
            if generated_card_id in credit_cards:
                rng = probe
                continue

            probe = cls._clone_rng(rng)
            generated_booking_id = str(probe.randint(1000000, 9999999))
            generated_transaction_id = str(probe.randint(10000000, 99999999))
            booking = booking_record.get(generated_booking_id)
            if isinstance(booking, dict):
                existing_transaction_id = booking.get("transaction_id")
                if existing_transaction_id in (None, generated_transaction_id, str(generated_transaction_id)):
                    rng = probe
                    continue

            break

        return rng

    @staticmethod
    def _travel_flight_cost(
        args: Dict[str, Any],
        schema_default_state: Dict[str, Any],
    ) -> Optional[float]:
        op_defaults = schema_default_state.get("operation_defaults")
        if not isinstance(op_defaults, dict):
            return None
        method_data = op_defaults.get("called_method_static_data")
        if not isinstance(method_data, dict):
            return None
        flight_cost_data = method_data.get("get_flight_cost")
        if not isinstance(flight_cost_data, dict):
            return None
        base_costs = flight_cost_data.get("base_costs")
        factors = flight_cost_data.get("factor")
        if not isinstance(base_costs, dict) or not isinstance(factors, dict):
            return None

        route_key = f"{args.get('travel_from')}|{args.get('travel_to')}"
        base_cost = base_costs.get(route_key)
        factor = factors.get(args.get("travel_class"))
        if not isinstance(base_cost, (int, float)) or not isinstance(factor, (int, float)):
            return None
        travel_date = str(args.get("travel_date", ""))
        digit_sum = sum(int(ch) for ch in travel_date if ch.isdigit())
        date_factor = 2 if digit_sum % 2 == 0 else 1
        return float(base_cost * factor * date_factor)

    @classmethod
    def _source_aligned_travel_response(
        cls,
        operation_id: str,
        args: Dict[str, Any],
        state: Dict[str, Any],
        schema_default_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        travel_state = state.get("TravelAPI") if isinstance(state, dict) else {}
        if not isinstance(travel_state, dict):
            travel_state = {}

        if operation_id == "authenticate_travel":
            rng = cls._travel_rng_from_visible_state(travel_state)
            return {
                "expires_in": 2,
                "access_token": str(rng.randint(100000, 999999)),
                "token_type": "Bearer",
                "scope": args.get("grant_type"),
            }

        if operation_id == "register_credit_card":
            if travel_state.get("token_expires_in") is None:
                return {"error": "Token not initialized"}
            if travel_state.get("token_expires_in") == 0:
                return {"error": "Token expired"}
            if args.get("access_token") != travel_state.get("access_token"):
                return {"error": "Invalid access token"}
            credit_cards = travel_state.get("credit_card_list")
            if not isinstance(credit_cards, dict):
                credit_cards = {}
            if args.get("card_number") in credit_cards:
                return {"error": "Card already registered"}
            rng = cls._travel_rng_from_visible_state(travel_state)
            return {"card_id": str(rng.randint(100000000000, 999999999999))}

        if operation_id == "book_flight":
            if travel_state.get("token_expires_in") == 0:
                return {"booking_status": False, "error": "Token expired"}
            if args.get("access_token") != travel_state.get("access_token"):
                return {"booking_status": False, "error": "Invalid access token"}
            credit_cards = travel_state.get("credit_card_list")
            if not isinstance(credit_cards, dict):
                credit_cards = {}
            card = credit_cards.get(args.get("card_id"))
            if not isinstance(card, dict):
                return {"booking_status": False, "error": "Card not registered"}
            if "balance" not in card:
                return {"booking_status": False, "error": "Balance not found"}
            travel_cost = cls._travel_flight_cost(args, schema_default_state)
            if travel_cost is None:
                return {
                    "booking_status": False,
                    "error": "No available route for the given parameters",
                }
            try:
                datetime.strptime(str(args.get("travel_date", "")), "%Y-%m-%d")
            except ValueError:
                return {
                    "booking_status": False,
                    "error": "Invalid date format. Use YYYY-MM-DD.",
                }
            if args.get("travel_class") not in {"economy", "business", "first"}:
                return {
                    "booking_status": False,
                    "error": "Invalid travel class. Must be one of {'economy', 'business', 'first'}",
                }
            try:
                balance = float(card.get("balance", 0))
            except Exception:
                balance = 0.0
            if balance < travel_cost:
                return {"booking_status": False, "error": "Insufficient funds"}
            budget_limit = travel_state.get("budget_limit")
            if budget_limit is not None:
                try:
                    if balance < float(budget_limit):
                        return {
                            "booking_status": False,
                            "error": "Balance is less than budget limit",
                        }
                except Exception:
                    pass

            rng = cls._travel_rng_from_visible_state(travel_state)
            booking_id = str(rng.randint(1000000, 9999999))
            transaction_id = str(rng.randint(10000000, 99999999))
            return {
                "booking_id": booking_id,
                "transaction_id": transaction_id,
                "booking_status": True,
                "booking_history": {},
            }

        if operation_id == "purchase_insurance":
            if travel_state.get("token_expires_in") == 0:
                return {"insurance_status": False, "error": "Token expired"}
            if args.get("access_token") != travel_state.get("access_token"):
                return {"insurance_status": False, "error": "Invalid access token"}
            budget_limit = travel_state.get("budget_limit")
            insurance_cost = args.get("insurance_cost")
            if budget_limit is not None:
                try:
                    if float(budget_limit) < float(insurance_cost):
                        return {"insurance_status": False, "error": "Exceeded budget limit"}
                except Exception:
                    pass
            booking_record = travel_state.get("booking_record")
            if not isinstance(booking_record, dict) or args.get("booking_id") not in booking_record:
                return {"insurance_status": False, "error": "Booking not found"}
            credit_cards = travel_state.get("credit_card_list")
            if not isinstance(credit_cards, dict) or args.get("card_id") not in credit_cards:
                return {"insurance_status": False, "error": "Credit card not registered"}
            rng = cls._travel_rng_from_visible_state(travel_state)
            return {
                "insurance_id": str(rng.randint(100000000, 999999999)),
                "insurance_status": True,
            }

        return None

    def _source_aligned_response(
        self,
        schema: Dict[str, Any],
        operation: Optional[Dict[str, Any]],
        request_info: Dict[str, Any],
        effective_state: Dict[str, Any],
        schema_default_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(operation, dict):
            return None
        toolkit_name = self._toolkit_name_from_schema(schema)
        operation_id = operation.get("operationId")
        if toolkit_name != "TravelAPI" or not isinstance(operation_id, str):
            return None
        args = self._extract_arguments(request_info)
        return self._source_aligned_travel_response(
            operation_id=operation_id,
            args=args,
            state=effective_state,
            schema_default_state=schema_default_state,
        )

    async def generate_response(
        self,
        response_schema: Dict[str, Any],
        schema: Dict[str, Any],
        request: Request,
        operation: Dict[str, Any] = None,
    ) -> Any:
        """Generate a mock response based on the response schema.

        Args:
            response_schema: Expected response schema from OpenAPI spec
            schema: Full OpenAPI schema document
            request: FastAPI request object
            operation: OpenAPI operation details

        Returns:
            Generated response matching the schema
        """
        try:
            current_state, has_state, session_id = self._load_session_state(request)
            request_info = await self._extract_request_info(request)
            toolkit_info = extract_toolkit_info(schema)
            param_info_for_user = self._build_param_info(operation)
            tool_definition = self._build_tool_definition(operation, request)
            schema_default_state = self._extract_schema_default_state(schema, operation)
            schema_default_state_text = self._format_schema_default_state(schema_default_state)
            effective_state = self._materialize_missing_toolkit_state(
                current_state=current_state,
                schema=schema,
                schema_default_state=schema_default_state,
            )

            result = self._source_aligned_response(
                schema=schema,
                operation=operation,
                request_info=request_info,
                effective_state=effective_state,
                schema_default_state=schema_default_state,
            )

            if result is None:
                context_info = await asyncio.to_thread(
                    self._build_context_info,
                    current_state=effective_state,
                    operation=operation,
                    request_info=request_info,
                    toolkit_info=toolkit_info,
                    schema=schema,
                    schema_default_state=schema_default_state,
                    session_id=session_id,
                )

                enhanced_system_prompt = f"""{self.system_prompt}

## Toolkit Information
{toolkit_info}

IMPORTANT: You will receive the relevant system state in the user message. Base your response on that state.
"""

                resolved_schema = resolve_refs(response_schema, schema)
                user_message = self._build_user_message(
                    context_info=context_info,
                    tool_definition=tool_definition,
                    param_info=param_info_for_user,
                    request_info=request_info,
                    resolved_schema=resolved_schema,
                    schema_default_state_text=schema_default_state_text,
                )

                response_str = await asyncio.to_thread(
                    self._call_response_llm,
                    enhanced_system_prompt=enhanced_system_prompt,
                    user_message=user_message,
                    session_id=session_id,
                )

                result = self._parse_response(response_str)

            if has_state and self.state_model is not None:
                tool_descriptions = self._build_tool_descriptions(
                    operation,
                    tool_definition,
                    request,
                    schema=schema,
                )
                tool_calls = self._build_tool_calls(operation, request_info, request, result)
                # Pass the unmodified session state to the state updater, NOT
                # `effective_state`. `_materialize_missing_toolkit_state` seeds
                # synthetic fields (e.g., `current_working_directory` copied
                # from runtime_state) at toolkit top level for the response
                # model's view. Feeding that view to the state updater makes
                # the LLM see the same field at two locations and emit a
                # dual-write patch.
                previous_state = current_state if isinstance(current_state, dict) else {}
                await asyncio.to_thread(
                    update_state,
                    previous_state=previous_state,
                    tool_calls=tool_calls,
                    tool_descriptions=tool_descriptions,
                    session_id=session_id,
                    state_model=self.state_model,
                )
            elif has_state:
                logger.debug("[STATE UPDATE] Skipped in response generator because state_model is disabled")

            return result

        except Exception as e:
            logger.exception(f"Error generating response: {str(e)}")
            raise

    def _load_session_state(self, request: Request) -> tuple[Dict[str, Any], bool, Optional[str]]:
        """Load latest session state and ensure there is at least one persisted snapshot."""
        current_state = getattr(request.state, "session_state", {})
        has_state = bool(getattr(request.state, "session_has_state", False))
        session_id = request.headers.get("X-Session-ID")
        if not has_state and session_id:
            from ..handlers.session_handler import session_handler
            session_handler.add_to_state(session_id, {})
            current_state = {}
            has_state = True
        return current_state if isinstance(current_state, dict) else {}, has_state, session_id

    async def _extract_request_info(self, request: Request) -> Dict[str, Any]:
        """Extract and sanitize request info for prompt usage."""
        request_info = await RequestDetails.extract(request)
        request_info.pop("headers", None)
        request_info.pop("client", None)
        request_info.pop("path", None)
        return request_info

    def _build_param_info(self, operation: Optional[Dict[str, Any]]) -> str:
        """Build parameter + response field descriptions string for the prompt."""
        if not operation:
            return ""
        param_desc = extract_parameter_descriptions(operation)
        resp_desc = extract_response_descriptions(operation)
        return f"{param_desc}{resp_desc}" if (param_desc or resp_desc) else ""

    def _build_context_info(
        self,
        current_state: Dict[str, Any],
        operation: Optional[Dict[str, Any]],
        request_info: Dict[str, Any],
        toolkit_info: str,
        schema: Dict[str, Any],
        schema_default_state: Dict[str, Any],
        session_id: Optional[str],
    ) -> str:
        if self.state_model is None:
            if not current_state:
                if schema_default_state:
                    return (
                        "## Relevant System State\n"
                        "(State model disabled; runtime state unavailable. Use Schema Default State below as fallback constraints)\n"
                    )
                return "## Relevant System State\n(State model disabled; no runtime state constraints)\n"
            try:
                state_text = json.dumps(current_state, indent=2, ensure_ascii=False)
            except Exception:
                state_text = str(current_state)
            logger.debug(
                "[CTX_EXTRACTOR] Skipped because state_model is disabled; using raw state snapshot"
            )
            return f"## Relevant System State\n{state_text}\n"

        if not current_state:
            if schema_default_state:
                return (
                    "## Relevant System State\n"
                    "(Empty runtime state; use Schema Default State below as fallback constraints)\n"
                )
            return "## Relevant System State\n(Empty state - no constraints)\n"
        return self.extract_operation_context_with_llm(
            config=current_state,
            operation=operation,
            request_info=request_info,
            toolkit_info=toolkit_info,
            schema=schema,
            schema_default_state=schema_default_state,
            model=self.validation_model,
            session_id=session_id,
        )

    def _build_tool_definition(self, operation: Optional[Dict[str, Any]], request: Request) -> Dict[str, Any]:
        if not operation:
            return {}
        return {
            "operation_id": operation.get("operationId", ""),
            "summary": operation.get("summary", ""),
            "description": operation.get("description", ""),
            "method": getattr(request, "method", "").upper() if hasattr(request, "method") else "",
            "path": request.url.path if hasattr(request, "url") else "",
        }

    def _build_user_message(
        self,
        context_info: str,
        tool_definition: Dict[str, Any],
        param_info: str,
        request_info: Dict[str, Any],
        resolved_schema: Dict[str, Any],
        schema_default_state_text: str,
    ) -> str:
        return f"""
{context_info}

## Operation Being Performed
**Operation ID**: {tool_definition.get('operation_id', 'Unknown')}
**Summary**: {tool_definition.get('summary', 'No summary available')}
**Description**: {tool_definition.get('description', 'No description available')}
**Method**: {tool_definition.get('method', '')}
**Path**: {tool_definition.get('path', '')}
{param_info}

## Schema Default State (x-default-state, operation-filtered)
{schema_default_state_text}

## Actual Request
            {request_info.get('body', '{}')}


## Expected Response Schema
{json.dumps(resolved_schema, indent=2)}

Return a pure JSON object matching the response schema above.
"""

    def _call_response_llm(
        self,
        enhanced_system_prompt: str,
        user_message: str,
        session_id: Optional[str],
    ) -> str:
        agent = ChatAgent(
            enhanced_system_prompt,
            step_timeout=LLM_AGENT_TIMEOUT_SECONDS,
            tool_execution_timeout=LLM_AGENT_TIMEOUT_SECONDS,
            model=create_model(
                self.response_model,
                max_tokens=8192,
                temperature=0.001,
                timeout=LLM_AGENT_TIMEOUT_SECONDS,
            ),
        )

        _rt0 = datetime.now()
        logger.debug(
            "[RESPONSE] LLM START (model=%s)",
            self.response_model,
        )
        with bind_log_context(agent_role="gecko_response_generator"):
            response = agent.step(user_message)
        _rt1 = datetime.now()
        logger.debug(
            "[RESPONSE] LLM END (elapsed=%.3fs, model=%s)",
            (_rt1 - _rt0).total_seconds(),
            self.response_model,
        )

        response_str = self._response_content(response)
        return self._strip_wrappers(response_str)

    def _strip_wrappers(self, response_str: str) -> str:
        return sanitize_llm_json_text(response_str)

    def _parse_response(self, response_str: str) -> Any:
        """Parse model output as generic JSON value (object/null/string/etc.)."""
        cleaned = response_str.strip()
        try:
            return json_repair.loads(cleaned)
        except Exception:
            return {"message": cleaned}

    def _build_tool_descriptions(
        self,
        operation: Optional[Dict[str, Any]],
        tool_definition: Dict[str, Any],
        request: Request,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not operation:
            return None
        canonical_name = self._canonical_operation_name(operation, request)
        enriched_definition = dict(tool_definition)

        toolkit_info: Dict[str, Any] = {}
        if isinstance(schema, dict):
            info = schema.get("info")
            if isinstance(info, dict):
                title = info.get("title")
                description = info.get("description")
                if isinstance(title, str) and title:
                    toolkit_info["name"] = title
                if isinstance(description, str) and description:
                    toolkit_info["description"] = description

        if toolkit_info:
            enriched_definition["toolkit"] = toolkit_info

        if isinstance(schema, dict):
            default_state_hint = self._extract_schema_default_state(schema, operation)
            if isinstance(default_state_hint, dict):
                operation_defaults = default_state_hint.get("operation_defaults")
                if isinstance(operation_defaults, dict) and operation_defaults:
                    enriched_definition["state_hints"] = operation_defaults

        return {canonical_name: enriched_definition}

    def _build_tool_calls(
        self,
        operation: Optional[Dict[str, Any]],
        request_info: Dict[str, Any],
        request: Request,
        result: Any,
    ) -> list[Dict[str, Any]]:
        if not operation:
            return []
        canonical_name = self._canonical_operation_name(operation, request)
        arguments = self._extract_arguments(request_info)
        status, reason = classify_tool_call_status(result)
        tool_call: Dict[str, Any] = {
            "name": canonical_name,
            "arguments": arguments,
            "result": result,
            "execution_status": status,
        }
        if reason:
            tool_call["error_reason"] = reason
        return [tool_call]

    def _canonical_operation_name(self, operation: Dict[str, Any], request: Request) -> str:
        operation_id = operation.get("operationId")
        path_parts = request.url.path.split('/') if hasattr(request, "url") else []
        fallback_function_name = path_parts[-1] if len(path_parts) >= 1 else ""
        return operation_id or fallback_function_name

    def _extract_arguments(self, request_info: Dict[str, Any]) -> Dict[str, Any]:
        arguments: Dict[str, Any] = {}
        body_raw = request_info.get('body')
        if body_raw:
            try:
                body_obj = json.loads(body_raw)
                if (
                    isinstance(body_obj, dict)
                    and 'requestBody' in body_obj
                    and isinstance(body_obj['requestBody'], dict)
                ):
                    body_obj = body_obj['requestBody']
                if isinstance(body_obj, dict):
                    arguments.update(body_obj)
            except json.JSONDecodeError:
                pass
        if request_info.get('query_params') and isinstance(request_info['query_params'], dict):
            arguments.update(request_info['query_params'])
        if request_info.get('path_params') and isinstance(request_info['path_params'], dict):
            arguments.update(request_info['path_params'])
        return arguments

    def _parse_context_extraction_response(
        self,
        response: Any,
    ) -> Optional[ContextExtractionResult]:
        msg = getattr(response, "msg", None)
        parsed = getattr(msg, "parsed", None)
        if isinstance(parsed, ContextExtractionResult):
            return parsed
        if isinstance(parsed, dict):
            try:
                return ContextExtractionResult.model_validate(parsed)
            except Exception:
                pass

        raw_content = self._response_content(response)
        if not raw_content:
            return None
        cleaned = sanitize_llm_json_text(raw_content)
        try:
            payload = json_repair.loads(cleaned)
            return ContextExtractionResult.model_validate(payload)
        except Exception:
            return None

    def _build_context_extraction_repair_query(
        self,
        extraction_query: str,
        raw_response: str,
        error: str,
    ) -> str:
        return "\n\n".join(
            [
                extraction_query,
                "The previous context extraction response could not be parsed as the required schema.",
                f"Validation error: {error[:1000]}",
                "Rejected response:\n" + (raw_response or "<empty>")[:4000],
                (
                    "Return a corrected JSON object only. It must have exactly these fields: "
                    '"relevant_system_state" as an array of strings and '
                    '"operation_constraints" as an array of strings. '
                    "Use [] for any empty field. Do not include markdown or extra keys."
                ),
            ]
        )

    def _render_context_extraction_result(
        self,
        result: ContextExtractionResult,
    ) -> str:
        state_lines = result.relevant_system_state or ["(No operation-relevant state found)"]
        constraint_lines = result.operation_constraints or ["(No operation-specific constraints identified)"]

        rendered = ["## Relevant System State"]
        rendered.extend(f"- {line}" for line in state_lines)
        rendered.append("")
        rendered.append("## Operation Constraints")
        rendered.extend(f"- {line}" for line in constraint_lines)

        return "\n".join(rendered).strip() + "\n"

    def extract_operation_context_with_llm(
        self,
        config: Dict[str, Any],
        operation: Dict[str, Any],
        request_info: Dict[str, Any],
        toolkit_info: str = "",
        schema: Dict[str, Any] = None,
        schema_default_state: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        session_id: str | None = None,
    ) -> str:
        """Extract relevant context and state for the operation using LLM.

        Uses a specialized LLM agent to analyze the full system configuration and
        extract only the parts relevant to the current operation.

        Args:
            config: Full system configuration
            operation: Operation details from OpenAPI spec
            request_info: Request details including body and parameters
            toolkit_info: Toolkit description string
            schema: Full OpenAPI schema
            schema_default_state: Operation-filtered x-default-state defaults and static data.
            model: LLM model to use for extraction. If None, use self.response_model.

        Returns:
            Rendered context text containing relevant state facts and operation constraints.
        """
        try:
            param_descriptions = extract_parameter_descriptions(operation) if operation else ""
            response_desc = extract_response_descriptions(operation) if operation else ""
            try:
                full_config_text = (
                    json.dumps(config, indent=2, ensure_ascii=False)
                    if isinstance(config, dict)
                    else str(config)
                )
            except Exception:
                full_config_text = str(config)

            schema_default_state_prompt_view = self._prompt_safe_schema_default_state(schema_default_state)
            try:
                schema_default_state_for_context = (
                    json.dumps(schema_default_state_prompt_view, indent=2, ensure_ascii=False)
                    if isinstance(schema_default_state_prompt_view, dict) and schema_default_state_prompt_view
                    else "(No operation-relevant x-default-state found)"
                )
            except Exception:
                schema_default_state_for_context = str(schema_default_state_prompt_view)

            extraction_query = f"""
## Toolkit Information
{toolkit_info}

## Operation Details
- Operation ID: {operation.get('operationId', 'unknown') if operation else 'unknown'}
- Summary: {operation.get('summary', '') if operation else ''}
- Description: {operation.get('description', '') if operation else ''}
{param_descriptions}
{response_desc}

## Request Information
- Method: {operation.get('method', request_info.get('method', '')) if operation else ''}
- Path: {operation.get('path', request_info.get('path', '')) if operation else ''}
- Actual Request Body: {request_info.get('body', '{}') if request_info else '{}'}

## Full System Configuration
{full_config_text}

## Schema Default State (x-default-state, operation-filtered)
{schema_default_state_for_context}

## Task
Extract and summarize the parts of the system configuration that are relevant to this operation.

Consider:
1. What is the current state/context (from runtime_state)?
2. What resources does this operation need to access or modify?
3. For operations with source/destination: what exists at each location?
4. What constraints apply based on the operation description?
5. What state facts are needed to evaluate those constraints?

Provide a clear, structured summary of the relevant state. DO NOT generate a response. DO NOT decide whether the operation passes or fails.
"""

            model_name = model or self.response_model
            extract_model = create_model(
                model_name,
                max_tokens=8192,
                temperature=0.001,
                timeout=LLM_AGENT_TIMEOUT_SECONDS,
            )
            extract_agent = ChatAgent(
                CONTEXT_EXTRACTION_SYSTEM_PROMPT,
                model=extract_model,
                step_timeout=LLM_AGENT_TIMEOUT_SECONDS,
            )
            try:
                with bind_log_context(agent_role="gecko_context_extractor"):
                    response = extract_agent.step(
                        extraction_query,
                        response_format=ContextExtractionResult,
                    )
            except Exception as exc:
                logger.warning(
                    "[CTX_EXTRACTOR] Structured extraction failed: %s",
                    exc,
                )
                raise
            parsed_context = self._parse_context_extraction_response(response)
            raw_context = self._response_content(response)
            if parsed_context is None:
                parse_error = "Structured response was not parseable as ContextExtractionResult"
                repair_query = self._build_context_extraction_repair_query(
                    extraction_query,
                    raw_context,
                    parse_error,
                )
                try:
                    with bind_log_context(agent_role="gecko_context_extractor"):
                        repair_response = extract_agent.step(
                            repair_query,
                            response_format=ContextExtractionResult,
                    )
                    repair_raw_context = self._response_content(repair_response)
                    repair_context = self._parse_context_extraction_response(repair_response)
                except Exception as repair_exc:
                    logger.warning(
                        "[CTX_EXTRACTOR] Structured extraction repair failed: %s",
                        repair_exc,
                    )
                    raise RuntimeError(
                        "Context extractor did not return a parseable structured response"
                    ) from repair_exc

                if repair_context is None:
                    raise RuntimeError(
                        "Context extractor did not return a parseable structured response"
                    )

                parsed_context = repair_context

            extracted_context = self._render_context_extraction_result(parsed_context)

            if extracted_context.startswith("```"):
                lines = extracted_context.split('\n')
                extracted_context = '\n'.join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            return extracted_context

        except Exception as e:
            logger.exception(f"LLM context extraction failed: {e}")
            raise

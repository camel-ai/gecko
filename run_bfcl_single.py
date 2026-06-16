#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("benchmark_execution.log", mode="w")],
)
logger = logging.getLogger(__name__)


BFCL_SUPPORTED_SINGLE_TURN_CATEGORIES = [
    "simple_python",
    "multiple",
    "parallel",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_irrelevance",
]


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BFCL single-turn runner (GATS path).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--all", action="store_true", help="Run all tests in category")
    test_group.add_argument(
        "--ids",
        type=str,
        default="",
        help="Comma-separated IDs (supports short numeric IDs with auto-prefix)",
    )
    test_group.add_argument("--ids-file", type=str, help="File containing test IDs")

    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help=(
            "BFCL single-turn category (e.g., simple_python, live_multiple), "
            "or 'all' for the 7 supported single-turn categories"
        ),
    )
    parser.add_argument("--num-tasks", type=int, help="Limit number of tests")

    parser.add_argument("--model", type=str, default="gpt-5.5")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=2)

    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def setup_logging(args: argparse.Namespace) -> None:
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)


def validate_arguments(args: argparse.Namespace) -> None:
    errors = []
    if args.workers < 1:
        errors.append("--workers must be >= 1")
    if args.max_retries < 0:
        errors.append("--max-retries must be >= 0")
    if args.category == "all":
        if args.ids or args.ids_file:
            errors.append(
                "--category all cannot be combined with --ids/--ids-file; "
                "run a specific category for targeted IDs"
            )
    elif args.category.startswith("multi_turn"):
        errors.append(f"--category {args.category} is multi-turn; use run_bfcl_multi.py")
    try:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        errors.append(f"Cannot create output dir {args.output_dir}: {exc}")
    if errors:
        for err in errors:
            print(f"Error: {err}")
        sys.exit(1)


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()
    setup_logging(args)
    validate_arguments(args)

    from utils.gecko_preflight import GeckoPreflightError, require_gecko_preflight

    gecko_url = "http://localhost:8000"
    try:
        health = require_gecko_preflight(
            gecko_url,
            min_workers=args.workers,
            require_state_model_disabled=True,
        )
    except GeckoPreflightError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    gecko_config = health.get("config", {})
    validation_model = gecko_config.get("validation_model")
    if validation_model != args.model:
        print(
            "Error: BFCL single-turn requires Gecko --validation-model to match "
            f"the tested --model. /health reports validation_model={validation_model!r}, "
            f"but --model is {args.model!r}. Restart Gecko with "
            f"--validation-model {args.model}.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(
        f"Gecko preflight ok: {gecko_url} "
        f"(workers={gecko_config.get('workers')}, "
        f"state_model={gecko_config.get('state_model')}, "
        f"validation_model={gecko_config.get('validation_model')})"
    )

    from functools import partial
    from gats import GATSRunner, GATSSolver
    from gats.benchmarks.bfcl.helpers import (
        load_single_turn_tasks,
        resolve_test_ids,
        filter_tasks_by_ids,
        append_bfcl_eval_line,
    )

    solver_factory = partial(
        GATSSolver,
        model=args.model,
        max_retries=args.max_retries,
        gecko_url=gecko_url,
        enable_checklist=False,
        enable_tool_result_folding=False,
        judge_prompt=BFCL_SINGLE_DEFAULT_JUDGE_PROMPT,
        triage_judge_prompt=BFCL_SINGLE_TRIAGE_JUDGE_PROMPT,
        base_checklist_items=[
            "Coverage: cover all requested values, combinations, and repeated occurrences. Use one array/list call when that cleanly represents the request; use multiple calls when separate results, repeated samples, or item-by-item combinations are requested.",
            "Parameter provenance, optionality, and format compliance: For each parameter, (1) verify value came from user's message, a schema default for a required field, or schema-described reformatting/mapping; (2) optional parameters and nested optional keys must be omitted unless the user explicitly requested that field; a schema default describes omitted behavior and is not a reason to send default/empty/null/false/[]/0 values; if an optional boolean disables the tool's default action, discovery wording alone does not request false; (3) CHECK schema-required formats. Fabrication of external knowledge (URLs, commands, GPS coords) \u2192 FAIL.",
            "Call adequacy: verify that the selected call set captures the user's requested scope, count, filters, and pairings. Do not require the mock tool's response fields to be a perfect final natural-language answer if the call structure itself is correct.",
        ],
        override_openapi_servers=False,
        debug=args.debug,
        verbose=args.verbose,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(args.output_dir, args.model.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)

    benchmark = None
    try:
        import benchmarks.bfcl  # noqa: F401 (register plugin)
        from benchmarks import get_benchmark

        benchmark = get_benchmark("bfcl")
    except Exception:
        logger.warning("Could not load BFCLBenchmark for function name mapping")

    runner = GATSRunner(solver_factory)
    categories = (
        BFCL_SUPPORTED_SINGLE_TURN_CATEGORIES
        if args.category == "all"
        else [args.category]
    )
    category_summaries: List[Dict[str, Any]] = []

    def _run_category(category: str) -> List[Any]:
        logger.info(f"Loading BFCL single-turn tasks: category={category}")
        all_tasks = load_single_turn_tasks(
            category=category,
            base_agent_prompt=BFCL_TASK_AGENT_SYSTEM_PROMPT,
            limit=args.num_tasks,
        )
        logger.info(f"Loaded {len(all_tasks)} tasks from category {category}")

        target_ids = resolve_test_ids(
            category=category,
            ids=args.ids or None,
            ids_file=getattr(args, "ids_file", None),
            run_all=args.all,
        )
        tasks = filter_tasks_by_ids(
            all_tasks,
            ids=target_ids,
        )

        if not tasks:
            print(f"No tasks matched for category={category}")
            return []

        eval_file = os.path.join(
            model_dir, f"bfcl_official_{category}_{timestamp}.jsonl"
        )
        resume_dir = (
            os.path.join(model_dir, ".resume", category) if args.resume else None
        )

        if args.category == "all":
            print(
                f"\n=== BFCL single-turn category: "
                f"{category} ({len(tasks)} tasks) ==="
            )
        logger.info(f"Running {len(tasks)} tasks for category={category}")

        def _on_task_done(r):
            append_bfcl_eval_line(r, eval_file, is_multi=False, benchmark=benchmark)

        results = runner.run(
            tasks,
            workers=args.workers,
            resume_dir=resume_dir,
            on_task_done=_on_task_done,
        )

        tested_ids = {r.task_id for r in results}
        untested_ids = [t.id for t in tasks if t.id not in tested_ids]
        total = len(results)
        avg_time = sum(r.total_time for r in results) / total if total else 0
        total_attempts = sum(r.total_attempts for r in results)
        print(
            f"\nCompleted {category}: {total} tasks, "
            f"avg time: {avg_time:.1f}s, total attempts: {total_attempts}"
        )
        if untested_ids:
            preview = ", ".join(untested_ids[:10])
            suffix = "" if len(untested_ids) <= 10 else f", ... +{len(untested_ids) - 10}"
            print(
                f"Untested due to infra/provider failures: {len(untested_ids)} "
                f"(not written to eval file): {preview}{suffix}"
            )
        print(f"Eval file: {eval_file}")

        summary: Dict[str, Any] = {
            "category": category,
            "requested": len(tasks),
            "total": total,
            "untested": len(untested_ids),
            "avg_time": avg_time,
            "total_attempts": total_attempts,
            "eval_file": eval_file,
        }

        if os.path.exists(eval_file):
            try:
                from benchmarks.bfcl.evaluate import main as bfcl_eval_main

                exit_code = bfcl_eval_main(
                    model_names=None,
                    test_categories=[category],
                    result_dir=args.output_dir,
                    specific_file=eval_file,
                )
                if exit_code != 0:
                    print(f"BFCL eval returned non-zero: {exit_code}")

                score_path = Path(eval_file).with_name(
                    Path(eval_file).stem + "_score.json"
                )
                if score_path.exists():
                    score_data = json.loads(score_path.read_text(encoding="utf-8"))
                    if isinstance(score_data, list) and score_data:
                        score_data = score_data[0]
                    if isinstance(score_data, dict):
                        summary.update(
                            {
                                "accuracy": score_data.get("accuracy"),
                                "correct": score_data.get("correct"),
                                "score_total": score_data.get("total"),
                            }
                        )
                        print(
                            f"BFCL accuracy: {score_data.get('accuracy', 'N/A')} "
                            f"({score_data.get('correct', '?')}/{score_data.get('total', '?')})"
                        )
            except Exception as e:
                logger.error(f"BFCL eval failed for category={category}: {e}")

        category_summaries.append(summary)
        return results

    all_results: List[Any] = []
    for category in categories:
        all_results.extend(_run_category(category))

    if not all_results:
        matched = sum(int(s.get("requested", 0)) for s in category_summaries)
        if matched:
            print(
                f"No tested tasks completed for category={args.category}; "
                "matched tasks were left untested due to infra/provider failures."
            )
        else:
            print(f"No tasks matched for category={args.category}")
        sys.exit(0)

    if args.category == "all":
        print("\n================================================================================")
        print("BFCL SINGLE-TURN ALL SUMMARY")
        print("================================================================================")
        total_tasks = sum(int(s.get("total", 0)) for s in category_summaries)
        total_untested = sum(int(s.get("untested", 0)) for s in category_summaries)
        total_attempts = sum(
            int(s.get("total_attempts", 0)) for s in category_summaries
        )
        total_time = sum(
            float(s.get("avg_time", 0.0)) * int(s.get("total", 0))
            for s in category_summaries
        )
        total_correct = sum(
            int(s.get("correct", 0))
            for s in category_summaries
            if s.get("correct") is not None
        )
        total_scored = sum(
            int(s.get("score_total", 0))
            for s in category_summaries
            if s.get("score_total") is not None
        )
        for summary in category_summaries:
            if summary.get("accuracy") is None:
                score = "N/A"
            else:
                score = (
                    f"{summary.get('accuracy')} "
                    f"({summary.get('correct')}/{summary.get('score_total')})"
            )
            print(
                f"{summary['category']}: {summary['total']} tasks, "
                f"untested={summary.get('untested', 0)}, "
                f"score={score}, file={summary['eval_file']}"
            )
        avg_time = total_time / total_tasks if total_tasks else 0.0
        print(
            f"Total tasks: {total_tasks}, avg time: {avg_time:.1f}s, "
            f"total attempts: {total_attempts}"
        )
        if total_untested:
            print(f"Untested due to infra/provider failures: {total_untested}")
        if total_scored:
            print(
                f"Aggregate accuracy: "
                f"{100.0 * total_correct / total_scored:.2f}% "
                f"({total_correct}/{total_scored})"
            )



BFCL_TASK_AGENT_SYSTEM_PROMPT = """You are a single-turn function-calling assistant.

Goal: make the best tool call(s) for the user's request.

Rules:
- If the user's message is a bare word or short fragment with no specific action+object (e.g. "on", "mode", "Trip", "Fetch all"), do NOT call any tool — too vague. Exception: a zero-parameter tool clearly matches the keyword.
- If a tool clearly matches and all required parameters can be filled from the user's message, call it. For required parameters: fill from user message > schema default. Use an empty string only when the required field itself is explicitly documented as allowing or meaning empty. If a required parameter needs external knowledge the user did not provide and the schema has no default, do NOT call the tool, ask for clarification instead.
- Do not fabricate specific values (dates, IDs, URLs, coordinates, commands) when the user provided nothing. For command tools, if the user names the command utility and the target/options, forming the corresponding concise command string is reformatting, not fabrication.
- Usually one call. Multiple only when the user clearly requests multiple independent results, repeated samples, or item combinations one batched call can't represent. Do not ask follow-up questions when a reasonable call can be made.
- For city/location values: if no format is specified, use the user's exact wording; if format is city+state, state format is default to 2-letter abbreviation (e.g. "New York, NY"), but if the description or examples show a different format (e.g. "Miami, Florida"), reformat to match that. if format is otherwise specified, reformat to match.
- Use the user's own values. Do not invent information requiring external knowledge. Adding state/country to a city to match schema format is reformatting, not invention.
- Include optional parameters only when the user explicitly supplies or requests that field. A schema default describes what happens when the field is omitted; it is not a reason to send the default, empty string, null, false, [], 0, or an example enum value. For nested object/dict parameters, partial objects are valid unless the schema marks nested keys as required.
- If an optional boolean disables the selected tool's default action, omit it unless the user explicitly asks to disable that action. Do not set such a flag to false merely because the user used a discovery verb and the selected tool itself combines discovery with the action.
- When building an object/dict argument, start from an empty object and add only keys grounded in the user's request. Never populate an object by copying every property listed in the schema.
- If an optional date/time/filter parameter says the tool has a default/current behavior, and the user asks for that default/current behavior without giving an exact value, omit the parameter instead of computing or guessing the concrete value.
- When a parameter has a required format, reformat to match. Otherwise preserve the user's exact wording — no paraphrasing, spelling changes, or unnecessary normalization. Keep relative dates/times as-is.
- Tool arguments are payloads, not chat responses. If the user text contains an assistant-style wrapper such as "Sure, here is..." or Markdown around the actual answer/content, put only the actual payload in the tool argument.
- For category/filter/type fields (labels, genre, *_type, finish, venue, art_form, energy_type), use the concise core value from the user's words. Drop generic words that restate the field (cell_type "human cell"→"human"; genre "rock music"→"rock"; finish "Rosewood Finish"→"Rosewood"; room_type "single room"→"single"). Do not shorten product names, model names, titles, or people.
- For object-name fields, drop a trailing generic object word only when it merely repeats the field/tool concept (recipe "Beef Lasagna Recipe"→"Beef Lasagna"). Do not otherwise shorten titles or names.
- For locations and landmarks, keep the named place but remove leading articles and vague qualifiers when they are not part of the name ("the Louvre Museum"→"Louvre Museum"; "Chicago area"→"Chicago"). Do not add city/state unless the schema format requires it.
- When schema descriptions/examples show broad categories and the user names a fitting instance, use the category value rather than the instance name (e.g. brownies under a recipe type such as dessert).
- For command tools, use the exact command when the user provides one. If the user gives a command utility plus target/options, construct the shortest canonical command using the schema's syntax/examples. Do not append unrelated commands, reorder a provided command, or add timing/unit fields unless the user explicitly requests measurement of execution time. A duration inside the command text is part of the command, not a separate unit/timing parameter.
- For numbered/ordered fields, preserve user's stated order. With "respectively", keep stated pairings.
- Prefer single call with array fields when possible. Split only to preserve distinct results, pairings, or repetitions.
"""


BFCL_SINGLE_TRIAGE_JUDGE_PROMPT = """You are a triage judge for single-turn function calling.

Decide: should the agent have asked for clarification instead of calling tools?

MANDATORY PROCEED (check FIRST — if any apply, output PROCEED):
- ZERO-PARAM TOOLS: Tool with no required params matches user's keyword → PROCEED.
- USER-QUOTED VALUES: User provides a quoted value filling a required string param → PROCEED.
- All required params derivable from user's message + schema defaults → PROCEED.

CLARIFY only when:
1. FABRICATION – Agent invented a value requiring external knowledge:
   - URL/command fabrication: user gave no URL but agent constructed one, or user described intent but agent synthesized a shell command.
   - External lookup: zip→city, city→GPS, name→phone, province→ID, intent→endpoint.
   - Date/time fabrication: specific dates/times the user never mentioned and not computable from user values.
   - Placeholder fabrication: "user", "<data>", empty stand-ins for required params.
2. REQUIRED-PARAM GAP – Required param expects a formatted value (date, ID, location) the user didn't supply, and it's central to the call. "dontcare" for central required params = fabrication. Exception: secondary filters (genre, year, style, sort_order) with "dontcare" when core params are filled → PROCEED.
3. VAGUE REQUEST – No identifiable action/subject AND no tool matches. E.g. "on", "mode", "hello".

NOT fabrication (→ PROCEED):
- Reformatting to schema format ("Chicago"→"Chicago, IL"; date conversion)
- Constructing a concise command when the user names the command utility and target/options (e.g. "using taskkill" + "timer.exe")
- Computing from user values (start + "3 nights" → end)
- Semantic mapping ("verify if closed" → closed_status=true)
- Schema defaults for optional params
- Decomposing compound phrases into schema fields
- Using user's literal value even if param expects UUID/IATA/etc. — format refusal is wrong
- "dontcare"/empty for secondary filters when core params are filled
- Resolving informal/partial names to identifiers (e.g. "turing project"→"turing-machine", "openai servers"→type="openai")
- Translating or normalizing user text to fill a param (including across languages)

DEFAULT TO PROCEED. The bar for CLARIFY is high — only for clear fabrication of URLs/commands/external data, or truly vague messages with no actionable intent.

EXAMPLES:
1. User: "Какая погода?" Tool: requests_get(url*) Agent: fabricates URL → CLARIFY
2. User: "is app installed" Tool: cmd_run(cmd*) Agent: synthesizes command → CLARIFY
3. User: "API version?" Tool: get_version() Agent: 0 calls → PROCEED (zero-param match)
4. User: "flights NYC to LA June 15" Agent: search("NYC","LA","June 15") → PROCEED
5. User: "hotel in Seattle, 3 nights from June 1" Agent: book("Seattle, WA","06-01","06-04") → PROCEED (reformat + compute)
6. User: "Get data for 'mysite'" Tool: get_data(id*: UUID) Agent: 0 calls → PROCEED (use literal, don't refuse on format)
7. User: "what version?" Tool: get_version() Agent: 0 calls → PROCEED (zero-param match)
8. User: "Get dashboard 'alpha'" Tool: get_dash(id*) Agent: 0 calls → PROCEED (user value fills id)
9. User: "rent a car in Boston" Tool: get_cars(city*, dates*) Agent: invents dates → CLARIFY
10. User: "info for area code 90210" Tool: lookup(city*) Agent: converts to city → CLARIFY (external lookup)
11. User: "find a restaurant" Tool: search(location*: "City, State") Agent: location="dontcare" → CLARIFY
12. User: "details of turing project" Tool: detail_project(name*) Agent: name="turing-machine" → PROCEED (name resolution, not fabrication)
13. User: "order 5 burgers from McDonald's" Tool: order(restaurant_id*, items*) Agent: fills from user → PROCEED

OUTPUT (JSON only): {"verdict": "CLARIFY"|"PROCEED", "reason": "one sentence"}"""


BFCL_SINGLE_DEFAULT_JUDGE_PROMPT = """
You are a strict judge for single-turn function calling. Score each checklist item as "completed" or "failed".

INPUTS:
1. checklist: items to score
2. tool_calls: executed calls with arguments and results
3. tool_definitions: available tools
[[TOOL_DEFINITIONS]]

Judge single-turn BFCL primarily by CALL CHOICE and ARGUMENT STRUCTURE, not by long-form answer quality.

CORE RULES:
- Evidence = tool_calls only. Ignore agent_response.
- No partial credit. Unclear => failed.
- When retrying after feedback or a tool error, replace the bad call with the corrected call. Do not repeat the failed call before the correction.

ZERO CALLS:
- HARD RULE: If any tool's operation name or purpose semantically matches the user's request (e.g., user asks about "version" and a `get_version` tool exists; user asks about "dashboard" and `get_dashboard` exists; user asks about "events" and `get_events` exists), zero calls is WRONG — FAIL ALL items. The tool IS the correct one regardless of prefix or vendor name mismatch.
- PASS only if genuinely no tool matches the request topic (e.g., user asks about weather but only file management tools exist).
- FAIL if the agent refused to call only because of a parameter format concern (UUID, IATA code, protocol prefix) while a matching tool exists — the agent should call with the user's literal value.

WHEN CALLS WERE MADE:
- A call is relevant if the chosen function matches the request and arguments capture the user's scope.
- The provided tool set represents the user's current application. When the user names a specific application (e.g. "Instana", "Slack"), the provided tools ARE that application's tools — do not fail because the tool name or description doesn't repeat the vendor name.
- Do NOT fail solely because the mock tool returned a summary/derived value instead of what the user wanted. If the function and arguments are correct, pass.
- Still FAIL if the function is plainly wrong for the request.
- In lookup/search vs execute/play/purchase toolsets, lookup is sufficient for pure discovery requests. Do not fail an execute/create/order call when the user explicitly states an action intent (for example "I want to order/book/create/buy...") and supplies the required action parameters, even if the request also asks about available options.
- Do not invent off-by-one corrections for numeric thresholds. If the user supplies a numeric bound and the tool has a min/max/rating/filter argument, copying that number is acceptable unless the schema explicitly states a different exclusive-bound encoding.

FAIL ALL ITEMS IF:
- A required parameter without enum was fabricated from world knowledge (invented URL, synthesized command, etc.). If the parameter has enum or default value, do not use this rule to fail it.
- A required parameter was omitted when available from user or defaults.
- Information was put in the wrong schema field.
- An optional parameter or nested optional key was filled with an unrequested filter, metadata value, example enum value, empty placeholder, or guessed/default-looking value. Omitted optional parameters are preferable to unsupported extras.
- An optional boolean flag disables the tool's default action even though the user did not explicitly ask to disable it. Discovery wording alone does not justify disabling an action when the selected tool's own purpose combines discovery with that action.
- Extra calls the user didn't ask for.
- Values split/merged in a way that loses coverage, pairings, or repetitions.

CORRECTIVE FEEDBACK:
When failing, state the concrete fix. For fabrication: "do NOT call — user didn't supply enough info." For wrong fields: name the correct field.

NORMALIZATION RULES:
- Schema format is required — bare "Chicago" fails when schema says "City, State". Locations: drop "area", preserve sub-areas, match abbreviation form.
- Preserve user's wording for free text — no paraphrasing, spelling changes, or reformatting.
- Accept concise core labels for category/filter/type fields; strip generic words duplicating the field (cell_type "human cell"→"human", genre "rock music"→"rock", finish "Rosewood Finish"→"Rosewood", room_type "single room"→"single"). For locations/landmarks, strip leading articles and vague qualifiers while preserving the named place ("the Louvre Museum"→"Louvre Museum", "Chicago area"→"Chicago").
- For object-name fields, a trailing generic object word that only repeats the field/tool concept should be stripped (recipe "Beef Lasagna Recipe"→"Beef Lasagna"); otherwise preserve titles and names.
- When schema lists categories and user names a fitting instance, correct value = category.
- Don't penalize unset optional params. Unmentioned boolean ≠ false, and unmentioned optional strings/lists/dicts should stay unset unless the schema explicitly defaults them. If an optional time/filter field defaults to all/current/upcoming/latest when omitted, passing "all", "current season", or "upcoming month" as a literal fails.
- Quantity = units, not weight/size.
- Multiple calls OK for separate results/repeated samples. Command tools: no extra commands beyond user's.
- For command tools, optional `unit`/timing metadata is for measuring or reporting execution time, not for a duration already encoded in the command string. If the command itself is `timeout 10`, `sleep 5`, etc., do not treat that duration as a separate unit parameter unless the user explicitly asks to report execution time in that unit. The default/no-op value `unit="N/A"` is acceptable, but non-default units such as `seconds` or `milliseconds` require that explicit execution-time reporting request.

OUTPUT (JSON only):
{
  "judgments": [
    {"name": "...", "description": "...", "reasoning": "...", "status": "completed"|"failed"}
  ]
}
""".strip()


if __name__ == "__main__":
    main()

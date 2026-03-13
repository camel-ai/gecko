#!/usr/bin/env python3
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import evaluation components from the project-local official BFCL layer.
from benchmarks.bfcl.official_eval.constants.enums import Language
from benchmarks.bfcl.official_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
    multi_turn_checker,
    multi_turn_irrelevance_checker,
)
from benchmarks.bfcl.official_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    is_empty_execute_response,
)

# Keep lightweight local parsers for compatibility with our result format
from benchmarks.bfcl.single_turn.utils import (
    convert_to_function_call,
    ast_parse
)


def _load_ast_checker():
    """Load local lightweight AST checker for single-turn evaluation."""
    from benchmarks.bfcl.single_turn.ast_checker import ast_checker
    return ast_checker


SINGLE_TURN_CATEGORIES = ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple", "irrelevance", "memory", "web_search", "format_sensitivity"]
MULTI_TURN_CATEGORIES = ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param",
                        "multi_turn_long_context", "multi_turn_composite"]
LIVE_CATEGORIES = [
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_relevance",
    "live_irrelevance",
]

ALL_CATEGORIES = SINGLE_TURN_CATEGORIES + MULTI_TURN_CATEGORIES + LIVE_CATEGORIES


def parse_test_category_argument(test_categories: List[str]) -> List[str]:
    """
    Parse test category arguments.
    
    Args:
        test_categories: List of category names or ["all"]
        
    Returns:
        List of resolved category names
    """
    if "all" in test_categories:
        return ALL_CATEGORIES
    
    resolved = []
    for cat in test_categories:
        if cat == "single_turn":
            resolved.extend(SINGLE_TURN_CATEGORIES)
        elif cat == "multi_turn":
            resolved.extend(MULTI_TURN_CATEGORIES)
        elif cat == "live":
            resolved.extend(LIVE_CATEGORIES)
        elif cat in ALL_CATEGORIES:
            resolved.append(cat)
        else:
            print(f"Warning: Unknown category '{cat}', skipping")
    
    return list(set(resolved))  # Remove duplicates


def load_ground_truth(category: str) -> Dict[str, Dict]:
    """Load ground truth data for a category.

    Args:
        category: Test category name
    Returns:
        Dictionary mapping test IDs to ground truth data
    """
    local_base = "data/bfcl_v4/possible_answer"
    gt_file = f"{local_base}/BFCL_v4_{category}.json"
    
    if not os.path.exists(gt_file):
        return {}
    
    ground_truths = {}
    with open(gt_file, 'r') as f:
        for line in f:
            gt = json.loads(line.strip())
            ground_truths[gt["id"]] = gt
    
    return ground_truths


def load_test_questions(category: str) -> Dict[str, Dict]:
    """Load original test questions/functions for a category.

    Args:
        category: Test category name
    Returns:
        Dictionary mapping test IDs to test questions
    """
    local_base = "data/bfcl_v4/task"
    test_file = f"{local_base}/BFCL_v4_{category}.json"
    
    if not os.path.exists(test_file):
        return {}
    
    test_cases = {}
    with open(test_file, 'r') as f:
        for line in f:
            tc = json.loads(line.strip())
            test_cases[tc["id"]] = tc
    
    return test_cases


def load_model_results(result_file: str) -> Tuple[List[Dict], Optional[str]]:
    """
    Load model results from file.
    
    Args:
        result_file: Path to results file (JSON or JSONL)
        
    Returns:
        Tuple of (results list, detected model name)
    """
    results = []
    model_name = None
    
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Result file not found: {result_file}")
    
    # Try to extract model name from filename
    filename = Path(result_file).stem
    # Common pattern: category_timestamp.json or model_category_timestamp.json
    parts = filename.split('_')
    # Look for common model indicators
    for part in parts:
        if any(model in part.lower() for model in ['gpt', '4o', 'claude', 'gemini']):
            model_name = part
            break
    
    # Load results - try JSON first, fallback to JSONL if needed
    with open(result_file, 'r') as f:
        if result_file.endswith('.jsonl'):
            # JSONL format (explicit)
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        else:
            # Try JSON format first
            try:
                data = json.load(f)
                if isinstance(data, list):
                    results = data
                else:
                    # Single object, wrap in list
                    results = [data]
            except json.JSONDecodeError:
                # If JSON fails, try JSONL format
                f.seek(0)  # Reset file pointer
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))

    return results, model_name


def evaluate_single_turn_result(result: Dict, test_case: Dict, ground_truth: Dict, 
                               category: str, model_name: str = "unknown") -> Dict:
    """
    Evaluate a single turn result.
    
    Returns:
        Dictionary with evaluation results
    """
    test_id = result.get("id", "unknown")
    
    # Handle different result formats
    model_output = result.get("result", result.get("response", []))
    
    # Parse model output if it's a string
    if isinstance(model_output, str):
        try:
            model_output = json.loads(model_output) if model_output else []
        except:
            model_output = []
    
    # Store original for debugging
    original_model_output = model_output.copy() if isinstance(model_output, list) else model_output
    
    # Preprocess model output to handle different formats
    processed_output = []
    
    if isinstance(model_output, list):
        for item in model_output:
            if isinstance(item, dict):
                # Dict format - process JSON string parameters
                processed_item = {}
                for func_name, params in item.items():
                    # If params is a JSON string, parse it
                    if isinstance(params, str):
                        try:
                            params = json.loads(params)
                        except:
                            # If it fails to parse, keep as string
                            pass
                    processed_item[func_name] = params
                processed_output.append(processed_item)
            elif isinstance(item, list):
                # Nested list format - parse string function calls
                for subitem in item:
                    if isinstance(subitem, str):
                        # String format function call - use ast_parse
                        try:
                            parsed = ast_parse(subitem, language="Python")
                            processed_output.extend(parsed)
                        except:
                            # If parsing fails, try to keep original
                            processed_output.append(subitem)
                    else:
                        processed_output.append(subitem)
            elif isinstance(item, str):
                # Direct string format - parse it
                try:
                    parsed = ast_parse(item, language="Python")
                    processed_output.extend(parsed)
                except:
                    processed_output.append(item)
            else:
                processed_output.append(item)
        model_output = processed_output
    elif isinstance(model_output, str):
        # Single string function call
        try:
            model_output = ast_parse(model_output, language="Python")
        except:
            model_output = [model_output]
    
    # Get expected output
    expected_output = ground_truth.get("ground_truth", ground_truth.get("result", []))
    
    if not isinstance(expected_output, list):
        expected_output = [expected_output]
    
    # Get function descriptions from test case
    func_descriptions = test_case.get("function", [])
    
    # Determine language for official AST checker
    if "javascript" in category:
        language = Language.JAVASCRIPT
    elif "java" in category:
        language = Language.JAVA
    else:
        language = Language.PYTHON

    # Use official AST checker for evaluation
    try:
        language_for_checker = language.value if hasattr(language, "value") else language
        if isinstance(language_for_checker, str):
            lowered = language_for_checker.lower()
            if lowered == "python":
                language_for_checker = "Python"
            elif lowered == "java":
                language_for_checker = "Java"
            elif lowered in ("javascript", "js"):
                language_for_checker = "JavaScript"
        checker_result = _load_ast_checker()(
            func_descriptions,
            model_output,
            expected_output,
            language_for_checker,
            test_category=category,
            model_name=model_name
        )
        
        eval_result = {
            "id": test_id,
            "correct": checker_result.get("valid", False),
            "error": checker_result.get("error", None),
            "error_type": checker_result.get("error_type", None),
            "model_output": original_model_output,
            "ground_truth": expected_output
        }
        # Add processed output if different from original
        if original_model_output != model_output:
            eval_result["processed_output"] = model_output
        return eval_result
        
    except Exception as e:
        return {
            "id": test_id,
            "correct": False,
            "error": f"Evaluation error: {str(e)}",
            "error_type": "evaluation_error"
        }


def decode_execute_simplified(model_result_item) -> List[str]:
    """
    Simplified version of handler.decode_execute for multi-turn evaluation.
    Converts model output to executable function call strings.
    
    Based on BFCL official decode_execute logic.
    """
    if not model_result_item:
        return []
        
    decoded_result = []
    
    # Handle list of function calls (FC mode output)
    if isinstance(model_result_item, list):
        for func_call in model_result_item:
            if isinstance(func_call, dict):
                # Convert dict format to executable string format
                # Format: {"function_name": "{parameters}"}
                for func_name, params in func_call.items():
                    if isinstance(params, str):
                        try:
                            params_dict = json.loads(params)
                        except:
                            params_dict = {}
                    else:
                        params_dict = params or {}

                    # Build executable string like: function_name(param1='value1', param2='value2')
                    param_str = ", ".join([f"{k}={repr(v)}" for k, v in params_dict.items()])
                    decoded_result.append(f"{func_name}({param_str})")
            elif isinstance(func_call, str):
                # Handle string elements (already formatted function calls)
                # E.g., "cd(folder='document')" or "mkdir(dir_name='temp')"
                decoded_result.append(func_call)
    
    # Handle string output (prompting mode or text response)
    elif isinstance(model_result_item, str):
        # Try to use the official convert_to_function_call if it's valid function call format
        try:
            # Check if it looks like function calls
            if "(" in model_result_item and ")" in model_result_item:
                # Try parsing as function calls
                result = convert_to_function_call(model_result_item)
                if result:
                    decoded_result = result if isinstance(result, list) else [result]
        except:
            pass
    
    return decoded_result


def evaluate_multi_turn_result(result: Dict, test_case: Dict, ground_truth: Dict,
                               category: str, model_name: str = "unknown") -> Dict:
    """
    Evaluate a multi-turn result following BFCL official logic.
    Based on: benchmarks/bfcl/official_eval/eval_checker/eval_runner.py::multi_turn_runner
    """
    test_id = result.get("id", "unknown")
    
    # Get model output and ground truth - following BFCL official
    multi_turn_model_result_list = result.get("result", result.get("response", []))
    multi_turn_ground_truth_list = ground_truth.get("ground_truth", [])
    
    # Check if model result is valid (from BFCL official)
    if not isinstance(multi_turn_model_result_list, list):
        return {
            "id": test_id,
            "correct": False,
            "error": "Error during inference phase. Model did not output a list of model responses.",
            "error_type": "multi_turn:inference_error"
        }
    
    # Check for force termination - from BFCL official
    if len(multi_turn_model_result_list) != len(multi_turn_ground_truth_list):
        return {
            "id": test_id,
            "correct": False,
            "error": f"Model was force-terminated during inference phase. The length of the model result turns ({len(multi_turn_model_result_list)}) does not match the length of the ground truth turns ({len(multi_turn_ground_truth_list)}).",
            "error_type": "multi_turn:force_terminated"
        }
    
    # Decode the model results - following BFCL official logic
    multi_turn_model_result_list_decoded = []  # list[list[list[str]]]

    for single_turn_model_result_list in multi_turn_model_result_list:
        single_turn_model_result_list_decoded = []

        # Ensure it's a list
        if not isinstance(single_turn_model_result_list, list):
            single_turn_model_result_list = [single_turn_model_result_list]

        # Check if this is already a list of function call strings
        # E.g., ["cd(...)", "mkdir(...)", "mv(...)"]
        if single_turn_model_result_list and all(isinstance(item, str) for item in single_turn_model_result_list):
            # Already decoded format - wrap it in a single sub-step
            single_turn_model_result_list_decoded.append(single_turn_model_result_list)
        else:
            # Original format - need to decode each item
            for model_result_item in single_turn_model_result_list:
                # model_result_item is per step
                try:
                    decoded_result = decode_execute_simplified(model_result_item)
                    if is_empty_execute_response(decoded_result):
                        # Empty output is not considered as a valid function call
                        continue

                    single_turn_model_result_list_decoded.append(decoded_result)

                except Exception as e:
                    # Ignore any failed decoding and continue to the next message
                    # We only care about the decoded function call, not the error message
                    continue

        multi_turn_model_result_list_decoded.append(single_turn_model_result_list_decoded)
    
    # Check using BFCL official multi_turn_checker
    accuracy_checker_result = multi_turn_checker(
        multi_turn_model_result_list_decoded,
        multi_turn_ground_truth_list,
        test_case,
        category,
        model_name,
    )
    
    # Also check irrelevance (when model shouldn't output function calls)
    irrelevance_checker_result = multi_turn_irrelevance_checker(
        multi_turn_model_result_list_decoded,
        multi_turn_ground_truth_list,
    )
    
    # Determine if correct based on both checkers
    is_correct = accuracy_checker_result["valid"] and irrelevance_checker_result["valid"]
    
    # Build error message if not correct
    error_msg = None
    error_type = None
    if not accuracy_checker_result["valid"]:
        error_msg = accuracy_checker_result.get("error_message", "Unknown error")
        error_type = accuracy_checker_result.get("error_type", "multi_turn:unknown")
    elif not irrelevance_checker_result["valid"]:
        error_msg = irrelevance_checker_result.get("error_message", "Unknown error")
        error_type = irrelevance_checker_result.get("error_type", "multi_turn:unknown")
    
    return {
        "id": test_id,
        "correct": is_correct,
        "error": error_msg,
        "error_type": error_type,
        "model_output_decoded": multi_turn_model_result_list_decoded,
        "ground_truth": multi_turn_ground_truth_list
    }


def extract_numeric_id(test_id: str) -> int:
    """Extract numeric ID from test ID like 'simple_4' or 'multiple_3-0-7'"""
    import re
    match = re.search(r'_(\d+)', test_id)
    return int(match.group(1)) if match else 0

def format_tool_call(tool_call):
    """Format a tool call as a single line JSON string"""
    if isinstance(tool_call, dict):
        return json.dumps(tool_call, separators=(',', ':'))
    elif isinstance(tool_call, str):
        try:
            # Try to parse and re-format if it's a JSON string
            parsed = json.loads(tool_call)
            return json.dumps(parsed, separators=(',', ':'))
        except:
            return str(tool_call)
    return str(tool_call)

def evaluate_category(result_file: str, category: str,
                     model_name: Optional[str] = None, show_errors: bool = True) -> Dict[str, Any]:
    """
    Evaluate results for a single category.

    Args:
        result_file: Path to result file
        category: Test category
        model_name: Model name (optional, auto-detected if None)
        show_errors: Whether to show error cases
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n📊 Evaluating category: {category}")
    print(f"📁 File: {result_file}")
    
    # Load data
    results, detected_model = load_model_results(result_file)
    if model_name is None:
        model_name = detected_model or "unknown"
    
    ground_truths = load_ground_truth(category)
    test_cases = load_test_questions(category)
    
    if not ground_truths:
        print(f"  ⚠️  No ground truth available for {category}")
        return {
            "category": category,
            "total": len(results),
            "evaluated": 0,
            "correct": 0,
            "accuracy": 0.0,
            "message": "No ground truth available"
        }
    
    # Evaluate each result
    evaluation_results = []
    error_cases = []
    correct = 0
    total = 0
    
    for result in results:
        test_id = result.get("id", "unknown")
        
        # Skip if no ground truth
        if test_id not in ground_truths:
            continue
        
        ground_truth = ground_truths[test_id]
        test_case = test_cases.get(test_id, {})
        
        # Choose evaluation method based on category
        if category in MULTI_TURN_CATEGORIES:
            eval_result = evaluate_multi_turn_result(
                result, test_case, ground_truth, category, model_name
            )
        else:
            eval_result = evaluate_single_turn_result(
                result, test_case, ground_truth, category, model_name
            )
        
        # Add model output and ground truth to eval result for error cases
        # Keep the original format without double-encoding
        eval_result["model_output"] = result.get("result", [])
        eval_result["ground_truth"] = ground_truth.get("ground_truth", ground_truth.get("result", []))
        
        evaluation_results.append(eval_result)
        total += 1
        if eval_result["correct"]:
            correct += 1
        else:
            # Remove the 'correct' field since it's always False for error cases
            error_case = {k: v for k, v in eval_result.items() if k != "correct"}
            error_cases.append(error_case)
    
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"  Total samples: {len(results)}")
    print(f"  Evaluated: {total}")
    print(f"  Correct: {correct}")
    print(f"  Errors: {total - correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    # Show error cases if requested
    if show_errors and error_cases:
        # Sort error cases by numeric ID
        error_cases.sort(key=lambda x: extract_numeric_id(x["id"]))
        
        print(f"\n❌ Error Cases ({len(error_cases)} total):")
        print("-" * 80)
        
        for error in error_cases:
            print(f"\nCase ID: {error['id']}")
            
            # Format model output
            print(f"Agent output:")
            if isinstance(error['model_output'], list):
                for i, call in enumerate(error['model_output'], 1):
                    # Check if already formatted as string
                    if isinstance(call, str):
                        print(f"  [{i}] {call}")
                    else:
                        print(f"  [{i}] {format_tool_call(call)}")
            else:
                if isinstance(error['model_output'], str):
                    print(f"  {error['model_output']}")
                else:
                    print(f"  {format_tool_call(error['model_output'])}")
            
            # Format ground truth
            print(f"Ground truth:")
            gt = error['ground_truth']
            if not isinstance(gt, list):
                gt = [gt]
            for i, call in enumerate(gt, 1):
                # Check if already formatted as string
                if isinstance(call, str):
                    print(f"  [{i}] {call}")
                else:
                    print(f"  [{i}] {format_tool_call(call)}")
            
            if error.get('error'):
                print(f"Error: {error['error']}")
        
        print("-" * 80)
    
    return {
        "category": category,
        "model": model_name,
        "total": len(results),
        "evaluated": total,
        "correct": correct,
        "accuracy": accuracy,
        "details": error_cases,  # Only return error cases, not all results
        "source_file": result_file  # Keep track of source file for score saving
    }


def format_evaluation_json(evaluations: List[Dict]) -> str:
    """Format evaluation results with compact tool calls."""
    result = "[\n"
    for i, eval_data in enumerate(evaluations):
        if i > 0:
            result += ",\n"
        result += "  {\n"
        result += f'    "category": "{eval_data["category"]}",\n'
        result += f'    "model": "{eval_data["model"]}",\n'
        result += f'    "total": {eval_data["total"]},\n'
        result += f'    "evaluated": {eval_data["evaluated"]},\n'
        result += f'    "correct": {eval_data["correct"]},\n'
        result += f'    "accuracy": {eval_data["accuracy"]},\n'
        result += '    "details": [\n'
        
        for j, detail in enumerate(eval_data["details"]):
            if j > 0:
                result += ",\n"
            result += "      {\n"
            result += f'        "id": "{detail["id"]}",\n'
            result += f'        "error": {json.dumps(detail["error"])},\n'
            result += f'        "error_type": {json.dumps(detail.get("error_type"))},\n'
            
            # Format model_output on single lines
            result += '        "model_output": [\n'
            model_output = detail["model_output"]
            if not isinstance(model_output, list):
                model_output = [model_output]
            for k, call in enumerate(model_output):
                if k > 0:
                    result += ",\n"
                result += f'          {json.dumps(call, separators=(",", ":"))}'
            result += '\n        ],\n'
            
            # Format ground_truth on single lines
            result += '        "ground_truth": [\n'
            gt = detail["ground_truth"]
            if not isinstance(gt, list):
                gt = [gt]
            for k, call in enumerate(gt):
                if k > 0:
                    result += ",\n"
                result += f'          {json.dumps(call, separators=(",", ":"))}'
            result += '\n        ]\n'
            result += "      }"
        
        result += "\n    ]\n"
        result += "  }"
    
    result += "\n]\n"
    return result

def save_evaluation_results(evaluations: List[Dict]):
    """Save evaluation results to score files next to the original result files."""
    
    # Group evaluations by source file
    file_evaluations = {}
    for eval_result in evaluations:
        source_file = eval_result.pop("source_file", None)  # Remove from result dict
        if source_file:
            if source_file not in file_evaluations:
                file_evaluations[source_file] = []
            file_evaluations[source_file].append(eval_result)
    
    # Save score file for each source file
    saved_files = []
    for source_file, evals in file_evaluations.items():
        # Create score filename by replacing extension with _score.json
        source_path = Path(source_file)
        score_file = source_path.parent / f"{source_path.stem}_score.json"
        
        # Write formatted JSON
        with open(score_file, 'w') as f:
            f.write(format_evaluation_json(evals))
        
        saved_files.append(score_file)
        print(f"\n📊 Score saved to: {score_file}")
    
    # Print summary statistics
    overall_correct = 0
    overall_total = 0
    
    for eval_result in evaluations:
        overall_correct += eval_result["correct"]
        overall_total += eval_result["evaluated"]
    
    if overall_total > 0:
        overall_accuracy = (overall_correct / overall_total * 100)
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total samples: {overall_total}")
        print(f"Correct: {overall_correct}")
        print(f"Accuracy: {overall_accuracy:.2f}%")
        print("="*80)
    
    return saved_files


def main(model_names: Optional[List[str]], test_categories: List[str],
         result_dir: str, specific_file: Optional[str] = None):
    """
    Main evaluation function.

    Args:
        model_names: List of model names to evaluate (optional)
        test_categories: List of test categories
        result_dir: Directory containing result files
        specific_file: Specific file to evaluate (optional)
    """
    all_evaluations = defaultdict(list)
    
    # If specific file is provided, evaluate only that file
    if specific_file:
        file_path = Path(specific_file)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return 1

        # Resolve category.
        # Prefer explicit CLI category over filename inference.
        category = None
        explicit_categories = parse_test_category_argument(test_categories or [])
        if len(explicit_categories) == 1:
            category = explicit_categories[0]
        else:
            # Extract category from filename. Match longer names first so
            # "live_multiple" is not swallowed by "multiple".
            filename = file_path.name
            for cat in sorted(ALL_CATEGORIES, key=len, reverse=True):
                if cat in filename:
                    category = cat
                    break
        
        if not category:
            filename = file_path.name
            print(f"⚠️  Could not determine category from filename: {filename}")
            print("Using 'unknown' as category")
            category = "unknown"
        
        print(f"\n📄 Evaluating file: {file_path}")
        print(f"📁 Category: {category}")
        
        eval_result = evaluate_category(str(file_path), category)
        model_name = eval_result.get("model", "unknown")
        all_evaluations[model_name].append(eval_result)
    else:
        # Original logic for directory-based evaluation
        categories = parse_test_category_argument(test_categories)
        
        if not categories:
            print("❌ No valid categories specified")
            return 1
        
        # Convert to Path objects
        result_path = Path(result_dir)
        
        if not result_path.exists():
            print(f"❌ Result directory not found: {result_path}")
            return 1
        
        
        # Find and evaluate result files
        for category in categories:
            # Look for result files matching this category in all subdirectories
            # Pattern: bfcl_official_{category}*.json, bfcl_official_{category}*.jsonl, {category}*.json, {category}*.jsonl
            pattern_files = list(result_path.glob(f"*/bfcl_official_{category}*.json")) + \
                           list(result_path.glob(f"*/bfcl_official_{category}*.jsonl")) + \
                           list(result_path.glob(f"bfcl_official_{category}*.json")) + \
                           list(result_path.glob(f"bfcl_official_{category}*.jsonl")) + \
                           list(result_path.glob(f"*/{category}*.json")) + \
                           list(result_path.glob(f"*/{category}*.jsonl")) + \
                           list(result_path.glob(f"{category}*.json")) + \
                           list(result_path.glob(f"{category}*.jsonl"))
            
            # Remove duplicates and filter out eval files
            result_files = []
            for f in pattern_files:
                if 'eval' not in f.name and 'score' not in f.name and f not in result_files:
                    result_files.append(f)
            
            if not result_files:
                print(f"⚠️  No result files found for category: {category}")
                continue
            
            for result_file in result_files:
                # Detect model name from filename if not specified
                if model_names:
                    # Use specified model names
                    for model_name in model_names:
                        eval_result = evaluate_category(str(result_file), category, model_name)
                        all_evaluations[model_name].append(eval_result)
                else:
                    # Auto-detect from filename or path
                    eval_result = evaluate_category(str(result_file), category)
                    model_name = eval_result.get("model", "unknown")
                    # Try to get model name from path if still unknown
                    if model_name == "unknown" and result_file.parent != result_path:
                        model_name = result_file.parent.name
                    all_evaluations[model_name].append(eval_result)
    
    # Save results for each model
    saved_files = []
    for model_name, evaluations in all_evaluations.items():
        if evaluations:
            files = save_evaluation_results(evaluations)
            saved_files.extend(files)
    
    print(f"\n🏁 Evaluation completed!")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BFCL Evaluation Tool - Evaluate model results against ground truth"
    )
    
    # Arguments following official BFCL conventions
    parser.add_argument(
        "--model", 
        nargs="+", 
        type=str,
        help="Model name(s) to evaluate (optional, auto-detected from filenames if not specified)"
    )
    parser.add_argument(
        "--test-category",
        nargs="+",
        type=str,
        default=["all"],
        help="Test categories to evaluate: all, single_turn, multi_turn, live, or specific categories"
    )
    parser.add_argument(
        "--result-dir",
        default="results",
        type=str,
        help="Directory containing model result files. Can be root dir (results) or specific model dir (results/gpt-4o)"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Specific result file to evaluate (overrides --result-dir and --test-category)"
    )

    args = parser.parse_args()

    # Run evaluation
    exit_code = main(
        args.model,
        args.test_category,
        args.result_dir,
        args.file,
    )
    
    exit(exit_code)

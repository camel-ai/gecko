import json
from typing import Dict, List, Any
from pathlib import Path


def enhance_score_file(score_file_path: str, results_file_path: str = None):
    """
    Enhance the score file with detailed state tracking and better error messages.
    """
    # Load the score file
    with open(score_file_path, 'r') as f:
        score_data = json.load(f)
    
    # If results file is provided, load it for additional context
    results_data = []
    if results_file_path and Path(results_file_path).exists():
        with open(results_file_path, 'r') as f:
            for line in f:
                if line.strip():
                    results_data.append(json.loads(line))
    
    # Create enhanced score format
    enhanced = {
        "summary": {
            "total": score_data["total"],
            "correct": score_data["correct"],
            "accuracy": score_data["accuracy"],
            "accuracy_percentage": f"{score_data['accuracy'] * 100:.2f}%"
        },
        "errors": []
    }
    
    for error in score_data.get("errors", []):
        enhanced_error = {
            "test_id": error["id"],
            "error_type": error["error_type"],
            "error_summary": error["error"],
            "turn_by_turn_analysis": []
        }
        
        # Add turn-by-turn comparison
        model_output = error.get("model_output", [])
        ground_truth = error.get("ground_truth", [])
        
        for turn_idx in range(max(len(model_output), len(ground_truth))):
            turn_analysis = {
                "turn": turn_idx + 1,
                "model_calls": model_output[turn_idx] if turn_idx < len(model_output) else [],
                "expected_calls": ground_truth[turn_idx] if turn_idx < len(ground_truth) else [],
                "differences": []
            }
            
            # Analyze differences
            if turn_idx < len(model_output) and turn_idx < len(ground_truth):
                model_calls = model_output[turn_idx]
                expected_calls = ground_truth[turn_idx]
                
                # Find extra calls in model output
                for call in model_calls:
                    if call not in expected_calls:
                        turn_analysis["differences"].append({
                            "type": "extra_call",
                            "call": call,
                            "impact": analyze_call_impact(call)
                        })
                
                # Find missing calls from model output
                for call in expected_calls:
                    if call not in model_calls:
                        turn_analysis["differences"].append({
                            "type": "missing_call",
                            "call": call,
                            "impact": analyze_call_impact(call)
                        })
                
                # Find calls with different parameters
                for model_call in model_calls:
                    for expected_call in expected_calls:
                        if same_function_different_params(model_call, expected_call):
                            turn_analysis["differences"].append({
                                "type": "parameter_mismatch",
                                "model_call": model_call,
                                "expected_call": expected_call,
                                "impact": analyze_param_difference(model_call, expected_call)
                            })
            
            enhanced_error["turn_by_turn_analysis"].append(turn_analysis)
        
        # Add state evolution if available
        if "state_evolution" in error:
            enhanced_error["state_evolution"] = error["state_evolution"]
        
        enhanced["errors"].append(enhanced_error)
    
    return enhanced


def analyze_call_impact(call: str) -> str:
    """Analyze the impact of a function call."""
    if "cd(" in call:
        return "Changes working directory - affects all subsequent file operations"
    elif "echo(" in call and "\\n" in call:
        return "Content includes newlines - affects file content and word count"
    elif "touch(" in call:
        return "Creates a new file"
    elif "wc(" in call:
        return "Counts words/lines in file"
    else:
        return "General operation"


def same_function_different_params(call1: str, call2: str) -> bool:
    """Check if two calls are to the same function but with different parameters."""
    # Extract function names
    func1 = call1.split('(')[0] if '(' in call1 else call1
    func2 = call2.split('(')[0] if '(' in call2 else call2
    
    if func1 != func2:
        return False
    
    # Check if parameters are different
    return call1 != call2


def analyze_param_difference(call1: str, call2: str) -> str:
    """Analyze the difference in parameters between two function calls."""
    if "echo(" in call1 and "echo(" in call2:
        if "\\n" in call1 and "\\n" not in call2:
            return "Model uses newlines (\\n) while expected uses spaces - affects word count"
        elif "\\n" not in call1 and "\\n" in call2:
            return "Model uses spaces while expected uses newlines - affects formatting"
    
    if "," in call1 and "," in call2:
        # Check for spacing differences
        if ", " in call1 and "," in call2 and ", " not in call2:
            return "Formatting difference: Model includes spaces after commas"
        elif "," in call1 and ", " not in call1 and ", " in call2:
            return "Formatting difference: Model lacks spaces after commas"
    
    return "Parameter values or formatting differ"


def format_enhanced_score(enhanced_data: Dict) -> str:
    """Format the enhanced score data as a readable string."""
    output = []
    output.append("=" * 80)
    output.append("ENHANCED EVALUATION REPORT")
    output.append("=" * 80)
    output.append("")
    
    # Summary
    output.append("📊 SUMMARY")
    output.append("-" * 40)
    output.append(f"Total Tests: {enhanced_data['summary']['total']}")
    output.append(f"Correct: {enhanced_data['summary']['correct']}")
    output.append(f"Accuracy: {enhanced_data['summary']['accuracy_percentage']}")
    output.append("")
    
    # Errors
    if enhanced_data["errors"]:
        output.append("❌ FAILED TESTS")
        output.append("-" * 40)
        
        for error in enhanced_data["errors"]:
            output.append(f"\n🔍 Test ID: {error['test_id']}")
            output.append(f"   Error Type: {error['error_type']}")
            output.append(f"   Summary: {error['error_summary']}")
            output.append("")
            output.append("   Turn-by-Turn Analysis:")
            
            for turn in error["turn_by_turn_analysis"]:
                output.append(f"\n   📍 Turn {turn['turn']}:")
                
                if turn["differences"]:
                    for diff in turn["differences"]:
                        if diff["type"] == "extra_call":
                            output.append(f"      ➕ Extra: {diff['call']}")
                            output.append(f"         Impact: {diff['impact']}")
                        elif diff["type"] == "missing_call":
                            output.append(f"      ➖ Missing: {diff['call']}")
                            output.append(f"         Impact: {diff['impact']}")
                        elif diff["type"] == "parameter_mismatch":
                            output.append(f"      ⚠️  Parameter Mismatch:")
                            output.append(f"         Model: {diff['model_call']}")
                            output.append(f"         Expected: {diff['expected_call']}")
                            output.append(f"         Impact: {diff['impact']}")
                else:
                    output.append("      ✅ No differences")
            
            output.append("")
            output.append("-" * 40)
    
    return "\n".join(output)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        score_file = sys.argv[1]
        results_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        enhanced = enhance_score_file(score_file, results_file)
        
        # Save enhanced score file
        output_path = score_file.replace(".json", "_enhanced.json")
        with open(output_path, 'w') as f:
            json.dump(enhanced, f, indent=2)
        
        # Print readable format
        print(format_enhanced_score(enhanced))
        print(f"\n📝 Enhanced score saved to: {output_path}")
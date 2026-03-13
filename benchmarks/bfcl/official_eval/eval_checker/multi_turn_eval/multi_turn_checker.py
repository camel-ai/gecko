from benchmarks.bfcl.official_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)


def multi_turn_checker(
    multi_turn_model_result_list_decoded: list[list[list[str]]],
    multi_turn_ground_truth_list: list[list[str]],
    test_entry: dict,
    test_category: str,
    model_name: str,
) -> dict:
    initial_config: dict = test_entry["initial_config"]
    involved_classes: list = test_entry["involved_classes"]
    test_entry_id: str = test_entry["id"]
    test_category = test_entry_id.rsplit("_", 1)[0]
    execution_results: list[dict] = []
    all_turn_model_execution_results: list[str] = []

    for turn_index, single_turn_ground_truth_list in enumerate(
        multi_turn_ground_truth_list
    ):
        single_turn_model_response_list = multi_turn_model_result_list_decoded[turn_index]
        single_turn_model_execution_results = []
        single_turn_model_execution_results_uncombined = []
        model_instances = {}

        for single_step_model_response in single_turn_model_response_list:
            single_step_model_execution_results, model_instances = execute_multi_turn_func_call(
                func_call_list=single_step_model_response,
                initial_config=initial_config,
                involved_classes=involved_classes,
                model_name=model_name,
                test_entry_id=test_entry_id,
                long_context=("long_context" in test_category or "composite" in test_category),
                is_eval_run=True,
            )
            single_turn_model_execution_results.extend(single_step_model_execution_results)
            single_turn_model_execution_results_uncombined.append(
                single_step_model_execution_results
            )

        single_turn_ground_truth_execution_results, ground_truth_instances = (
            execute_multi_turn_func_call(
                func_call_list=single_turn_ground_truth_list,
                initial_config=initial_config,
                involved_classes=involved_classes,
                model_name=model_name + "_ground_truth",
                test_entry_id=test_entry_id,
                long_context=("long_context" in test_category or "composite" in test_category),
                is_eval_run=True,
            )
        )

        all_turn_model_execution_results.extend(single_turn_model_execution_results)
        execution_results.append(
            {
                "model": single_turn_model_execution_results_uncombined,
                "ground_truth": single_turn_ground_truth_execution_results,
            }
        )

        if len(single_turn_ground_truth_list) > 0:
            if not single_turn_model_response_list or is_empty_execute_response(
                single_turn_model_response_list
            ):
                return {
                    "valid": False,
                    "error_message": f"Model response list is empty for turn {turn_index}",
                    "error_type": "multi_turn:empty_turn_model_response",
                    "details": {"execution_result": execution_results},
                }

        if not single_turn_ground_truth_list:
            continue

        assert len(model_instances) == len(ground_truth_instances)
        assert set(model_instances.keys()) == set(ground_truth_instances.keys())

        state_check_result = state_checker(model_instances, ground_truth_instances)
        if not state_check_result["valid"]:
            state_check_result["execution_result"] = execution_results
            return state_check_result

        response_check_result = response_checker(
            all_turn_model_execution_results,
            single_turn_ground_truth_execution_results,
            turn_index,
        )
        if not response_check_result["valid"]:
            return response_check_result

    return {"valid": True}


def multi_turn_irrelevance_checker(
    multi_turn_model_result_list_decoded: list[list[list[str]]],
    multi_turn_ground_truth_list: list[list[str]],
) -> dict:
    for turn_index, single_turn_ground_truth_list in enumerate(
        multi_turn_ground_truth_list
    ):
        single_turn_model_response_list = multi_turn_model_result_list_decoded[turn_index]
        if len(single_turn_ground_truth_list) == 0:
            if is_empty_execute_response(single_turn_model_response_list):
                continue
            return {
                "valid": False,
                "error_message": f"Model outputs valid function calls when it should not for turn {turn_index}.",
                "error_type": "multi_turn:irrelevance_error:decoder_success",
                "details": {"model response decoded": single_turn_model_response_list},
            }
    return {"valid": True}


def state_checker(model_instances: dict, ground_truth_instances: dict):
    for class_name, ground_truth_instance in ground_truth_instances.items():
        model_instance = model_instances[class_name]
        valid, differences = _compare_instances(model_instance, ground_truth_instance)

        if not valid:
            model_instance_attributes = {
                key: value
                for key, value in vars(model_instance).items()
                if not key.startswith("_")
            }
            ground_truth_instance_attributes = {
                key: value
                for key, value in vars(ground_truth_instance).items()
                if not key.startswith("_")
            }
            return {
                "valid": False,
                "error_message": f"Model instance for {class_name} does not match the state with ground truth instance.",
                "error_type": "multi_turn:instance_state_mismatch",
                "details": {
                    "differences": differences,
                    "model_instance_state": model_instance_attributes,
                    "ground_truth_instance_state": ground_truth_instance_attributes,
                },
            }

    return {"valid": True}


def response_checker(model_response_list: list, ground_truth_response_list: list, turn_index: int):
    is_subsequence, missing_items = _is_subsequence_unordered(
        ground_truth_response_list, model_response_list
    )
    if not is_subsequence:
        return {
            "valid": False,
            "error_message": f"Model response execution results so far does not contain all the ground truth response execution results for turn {turn_index}.",
            "error_type": "multi_turn:execution_response_mismatch",
            "details": {
                "missing_items": missing_items,
                "model_response (including all previous turns)": model_response_list,
                "ground_truth_response (only the current turn)": ground_truth_response_list,
            },
        }

    return {"valid": True}


def _compare_instances(model_object, ground_truth_object):
    model_dict = {
        key: value for key, value in vars(model_object).items() if not key.startswith("_")
    }
    ground_truth_dict = {
        key: value
        for key, value in vars(ground_truth_object).items()
        if not key.startswith("_")
    }

    differences = {}
    valid = True
    for key in set(model_dict.keys()).union(ground_truth_dict.keys()):
        if model_dict.get(key) != ground_truth_dict.get(key):
            differences[key] = {
                "model": model_dict.get(key),
                "ground_truth": ground_truth_dict.get(key),
            }
            valid = False

    return valid, differences


def _is_subsequence_unordered(target_list: list, candidate_list: list):
    remaining = list(candidate_list)
    missing_items = []
    for item in target_list:
        if item in remaining:
            remaining.remove(item)
        else:
            missing_items.append(item)
    return len(missing_items) == 0, missing_items

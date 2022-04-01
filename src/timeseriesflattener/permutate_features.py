from multiprocessing.sharedctypes import Value
from typing import Callable, Dict, Union
from xmlrpc.client import Boolean


def predictor_dict_has_feature_with_list(dataset_dict: Dict) -> Boolean:
    for dataset_key in dataset_dict.keys():
        if dataset_has_list_in_any_value(dataset_dict[dataset_key]):
            return True

    return False


def dataset_has_list_in_any_value(dataset_dict: Dict) -> Boolean:
    for feature_name in dataset_dict.keys():
        for param_name in dataset_dict[feature_name].keys():
            if isinstance(dataset_dict[feature_name][param_name], list):
                return True

    return False


def dict_has_list_in_any_value(dict: Dict) -> Boolean:
    """
    Checks if a dict has any values that are lists
    """
    for value in dict.values():
        if type(value) == list:
            return True
    return False


def get_first_key_in_dict(dict: Dict):
    return list(dict.keys())[0]


def create_feature_combinations(
    predictor_datasets: Dict[
        str, Dict[str, Dict[str, Union[Callable, float, int, str]]]
    ],
) -> Dict[str, Dict[str, Dict[str, Union[Callable, float, int, str]]]]:
    """
    Generate all permutations of parameters from a dictionary of parameters, where some keys are lists.

    Example:
        >>> predictor_dict = {
        >>>     "prediction_times_df": {
        >>>         "val1": {
        >>>             "lookbehind_days": [1, 30],
        >>>             "resolve_multiple": get_max_value_from_list_of_events,
        >>>             "fallback": 0,
        >>>             "source_values_col_name": "val",
        >>>         }
        >>>     }
        >>> }
        >>> all_features = permutate_features(dict)

    Result:
        >>> all_features = {
        >>>     "prediction_times_df": {
        >>>         "val11": {
        >>>             "lookbehind_days": 1,
        >>>             "resolve_multiple": "get_max_value_from_list_of_events",
        >>>             "fallback": 0,
        >>>             "source_values_col_name": "val",
        >>>         },
        >>>         "val12": {
        >>>             "lookbehind_days": 30,
        >>>             "resolve_multiple": "get_max_value_from_list_of_events",
        >>>             "fallback": 0,
        >>>             "source_values_col_name": "val",
        >>>     }
        >>> }
    """
    output_dict = {}

    if not predictor_dict_has_feature_with_list(predictor_datasets):
        return predictor_datasets
    else:
        for dataset_name in predictor_datasets.keys():
            if not dataset_has_list_in_any_value(predictor_datasets[dataset_name]):
                # If dataset is already processed, just append it to output_dict
                output_dict[dataset_name] = dataset_name.copy()
            else:
                output_dict[dataset_name] = {}

                for param_set_name in predictor_datasets[dataset_name].keys():
                    param_set = predictor_datasets[dataset_name][param_set_name]

                    if not dict_has_list_in_any_value(param_set):
                        # If the param_set doesn't contain any lists in values, append it to output_dict
                        output_dict[dataset_name][param_set_name] = param_set.copy
                    else:
                        for param_name in param_set.keys():
                            param_val = param_set[param_name]

                            hit_param_with_list_as_value = False

                            if isinstance(param_val, list):
                                hit_param_with_list_as_value = True

                                for item in param_val:
                                    i_param_set_name = (
                                        f"{param_set_name}_{param_name}_{item}"
                                    )

                                    i_param_set = param_set.copy()
                                    i_param_set[param_name] = item

                                    output_dict[dataset_name][
                                        i_param_set_name
                                    ] = i_param_set

                            if hit_param_with_list_as_value:
                                break

        return create_feature_combinations(output_dict)

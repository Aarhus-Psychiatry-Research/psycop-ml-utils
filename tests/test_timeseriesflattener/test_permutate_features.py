from timeseriesflattener.permutate_features import *


def test_create_feature_combinations():
    predictor_dict = {
        "prediction_times_df": {
            "val1": {
                "lookbehind_days": [1, 30],
                "resolve_multiple": "get_max_value_from_list_of_events",
                "fallback": 0,
                "source_values_col_name": "val",
            }
        }
    }

    expected_dict = {
        "prediction_times_df": {
            "val1_lookbehind_days_1": {
                "lookbehind_days": 1,
                "resolve_multiple": "get_max_value_from_list_of_events",
                "fallback": 0,
                "source_values_col_name": "val",
            },
            "val1_lookbehind_days_30": {
                "lookbehind_days": 30,
                "resolve_multiple": "get_max_value_from_list_of_events",
                "fallback": 0,
                "source_values_col_name": "val",
            },
        }
    }

    assert (
        create_feature_combinations(predictor_datasets=predictor_dict) == expected_dict
    )


def test_create_multiple_feature_combinations():
    predictor_dict = {
        "prediction_times_df": {
            "val1": {
                "lookbehind_days": [1, 30],
                "resolve_multiple": "get_max_value_from_list_of_events",
                "fallback": [0, 1],
                "source_values_col_name": "val",
            }
        }
    }

    expected_dict = {
        "prediction_times_df": {
            "val1_lookbehind_days_1_fallback_0": {
                "lookbehind_days": 1,
                "resolve_multiple": "get_max_value_from_list_of_events",
                "fallback": 0,
                "source_values_col_name": "val",
            },
            "val1_lookbehind_days_1_fallback_1": {
                "lookbehind_days": 1,
                "resolve_multiple": "get_max_value_from_list_of_events",
                "fallback": 1,
                "source_values_col_name": "val",
            },
            "val1_lookbehind_days_30_fallback_0": {
                "lookbehind_days": 30,
                "resolve_multiple": "get_max_value_from_list_of_events",
                "fallback": 0,
                "source_values_col_name": "val",
            },
            "val1_lookbehind_days_30_fallback_1": {
                "lookbehind_days": 30,
                "resolve_multiple": "get_max_value_from_list_of_events",
                "fallback": 1,
                "source_values_col_name": "val",
            },
        }
    }

    assert (
        create_feature_combinations(predictor_datasets=predictor_dict) == expected_dict
    )


def test_predictor_dict_has_feature_with_list():
    test_pos_predictor_dict = {
        "dataset1": {
            "val1": {
                "lookbehind_days": [1, 30],
                "resolve_multiple": "get_max_value_from_list_of_events",
                "fallback": [0, 1],
                "source_values_col_name": "val",
            }
        }
    }

    assert predictor_dict_has_feature_with_list(test_pos_predictor_dict)

    test_neg_predictor_dict = {
        "dataset1": {
            "val1": {
                "lookbehind_days": 1,
                "resolve_multiple": "get_max_value_from_list_of_events",
                "fallback": 0,
                "source_values_col_name": "val",
            }
        }
    }

    assert not predictor_dict_has_feature_with_list(test_neg_predictor_dict)


def test_dataset_dict_has_feature_with_list():
    test_pos_dataset_dict = {
        "val1": {
            "lookbehind_days": [1],
            "resolve_multiple": "get_max_value_from_list_of_events",
            "fallback": 0,
            "source_values_col_name": "val",
        }
    }

    assert dataset_has_list_in_any_value(test_pos_dataset_dict)

    test_neg_dataset_dict = {
        "val1": {
            "lookbehind_days": 1,
            "resolve_multiple": "get_max_value_from_list_of_events",
            "fallback": 0,
            "source_values_col_name": "val",
        }
    }

    assert not dataset_has_list_in_any_value(test_neg_dataset_dict)


def test_dict_has_list_as_val():
    test_pos_dict = {
        "lookbehind_days": [1, 30],
        "resolve_multiple": "get_max_value_from_list_of_events",
        "fallback": [0, 1],
        "source_values_col_name": "val",
    }

    assert dict_has_list_in_any_value(test_pos_dict)

    test_neg_dict = {
        "lookbehind_days": 1,
        "resolve_multiple": "get_max_value_from_list_of_events",
        "fallback": 0,
        "source_values_col_name": "val",
    }

    assert not dict_has_list_in_any_value(test_neg_dict)

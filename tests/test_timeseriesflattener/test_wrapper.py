from timeseriesflattener.create_feature_combinations import create_feature_combinations
from timeseriesflattener.flattened_dataset import FlattenedDataset
from timeseriesflattener.resolve_multiple_functions import get_max_in_group

from utils_for_testing import *


def test_generate_two_features_from_dict():
    """Test generation of features from a dictionary."""

    prediction_times_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """

    event_times_str = """dw_ek_borger,timestamp,val,
                        1,2021-12-30 00:00:01, 1
                        1,2021-12-29 00:00:02, 2
                        """

    expected_df_str = """dw_ek_borger,timestamp,val_within_1_days,val_within_2_days,val_within_3_days,val_within_4_days
                        1,2021-12-31 00:00:00,1,2,2,2                      
    """

    prediction_times_df = str_to_df(prediction_times_str)
    event_times_df = str_to_df(event_times_str)
    expected_df = str_to_df(expected_df_str)

    flattened_dataset = FlattenedDataset(
        prediction_times_df=prediction_times_df,
        timestamp_col_name="timestamp",
        id_col_name="dw_ek_borger",
        n_workers=4,
    )

    predictor_list = create_feature_combinations(
        [
            {
                "predictor_df": "event_times_df",
                "lookbehind_days": [1, 2, 3, 4],
                "resolve_multiple": "max",
                "fallback": 0,
                "source_values_col_name": "val",
            },
        ]
    )

    flattened_dataset.add_predictors_from_list_of_argument_dictionaries(
        predictor_list=predictor_list,
        predictor_dfs={"event_times_df": event_times_df},
        resolve_multiple_fn_dict={"max": get_max_in_group},
    )

    assert flattened_dataset.df.equals(expected_df)

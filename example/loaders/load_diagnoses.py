from psycopmlutils.loaders.raw.test_feature_combinations import (
    test_that_feature_combinations_return_correct_formatting,
)

if __name__ == "__main__":
    input_dict = [{"predictor_df": "sleep_apnea"}]

    test_that_feature_combinations_return_correct_formatting(
        predictor_dict_list=input_dict,
        n=100,
    )

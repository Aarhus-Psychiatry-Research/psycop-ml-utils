from psycopmlutils.loaders.raw.test_feature_combinations import (
    test_that_feature_combinations_return_values,
)

if __name__ == "__main__":
    input_dict = {"predictor_df": "sleep_apnea"}

    test_that_feature_combinations_return_values(predictor_dict=input_dict, n=100)

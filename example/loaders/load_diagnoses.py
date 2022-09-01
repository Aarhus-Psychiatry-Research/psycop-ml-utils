from psycopmlutils.loaders.raw.check_feature_combination_formatting import (
    check_feature_combinations_return_correct_dfs,
)

if __name__ == "__main__":
    input_dict = [{"predictor_df": "sleep_apnea"}]

    check_feature_combinations_return_correct_dfs(
        predictor_dict_list=input_dict,
        n=100,
    )

from psycopmlutils.data_checks.raw.check_predictor_lists import (
    check_feature_combinations_return_correct_dfs,
)
from psycopmlutils.loaders.raw.load_diagnoses import LoadDiagnoses

if __name__ == "__main__":
    df = sleep_apnea(n=100)

    input_dict = [{"predictor_df": "sleep_apnea"}]

    check_feature_combinations_return_correct_dfs(
        predictor_dict_list=input_dict,
        n=100,
    )

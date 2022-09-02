from psycopmlutils.loaders.raw.check_feature_combination_formatting import (
    check_feature_combinations_return_correct_formatting,
)
from psycopmlutils.loaders.raw.load_diagnoses import LoadDiagnoses

if __name__ == "__main__":
    df = LoadDiagnoses.sleep_apnea(n=100)

    input_dict = [{"predictor_df": "sleep_apnea"}]

    check_feature_combinations_return_correct_formatting(
        predictor_dict_list=input_dict,
        n=100,
    )

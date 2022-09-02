import psycopmlutils.loaders.raw as raw_loaders
from psycopmlutils.loaders.raw.check_feature_combination_formatting import check_raw_df

if __name__ == "__main__":
    df = raw_loaders.load_lab_results.LoadLabResults.unscheduled_p_glc(n=10_000)

    errors = check_raw_df(
        df=df,
        required_columns=["dw_ek_borger", "timestamp", "value"],
        subset_duplicates_columns=["dw_ek_borger", "timestamp", "value"],
    )

    print(errors)

    pass

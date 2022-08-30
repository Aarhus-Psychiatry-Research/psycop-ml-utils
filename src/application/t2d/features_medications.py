from psycopmlutils.timeseriesflattener import create_feature_combinations


def create_medication_feature_combinations(
    LOOKBEHIND_DAYS=[365, 730, 1825, 9999],
    RESOLVE_MULTIPLE=["mean", "max", "min"],
    fallback: any = 0,
):
    return create_feature_combinations(
        [
            {
                "predictor_df": "antipsychotics",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": 0,
            },
        ],
    )

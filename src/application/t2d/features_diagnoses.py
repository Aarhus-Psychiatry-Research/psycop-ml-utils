from psycopmlutils.timeseriesflattener import create_feature_combinations


def create_diag_feature_combinations(
    LOOKBEHIND_DAYS=[365, 730, 1825, 9999],
    RESOLVE_MULTIPLE=["mean", "max", "min"],
    fallback=None,
):
    return create_feature_combinations(
        [
            {
                "predictor_df": "essential_hypertension",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": fallback,
            },
            {
                "predictor_df": "hyperlipidemia",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": fallback,
            },
            {
                "predictor_df": "polycystic_ovarian_syndrome",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": fallback,
            },
            {
                "predictor_df": "sleep_apnea",
                "lookbehind_days": LOOKBEHIND_DAYS,
                "resolve_multiple": RESOLVE_MULTIPLE,
                "fallback": fallback,
            },
        ],
    )

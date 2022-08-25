from psycopmlutils.timeseriesflattener import create_feature_combinations


def create_diag_feature_combinations(
    lookbehind_days=[365, 730, 1825, 9999],
    resolve_multiple=["mean", "max", "min"],
    fallback: any = 0,
):
    """Create diagnosis feature combinations.

    Args:
        lookbehind_days (list, optional): List of lookbehind days. Defaults to [365, 730, 1825, 9999].
        resolve_multiple (list, optional): List of resolve multiple options. Defaults to ["mean", "max", "min"].
        fallback (any, optional): Fallback value. Defaults to 0.

    Returns:
        _type_: _description_
    """
    return create_feature_combinations(
        [
            {
                "predictor_df": "essential_hypertension",
                "lookbehind_days": lookbehind_days,
                "resolve_multiple": resolve_multiple,
                "fallback": fallback,
            },
            {
                "predictor_df": "hyperlipidemia",
                "lookbehind_days": lookbehind_days,
                "resolve_multiple": resolve_multiple,
                "fallback": fallback,
            },
            {
                "predictor_df": "polycystic_ovarian_syndrome",
                "lookbehind_days": lookbehind_days,
                "resolve_multiple": resolve_multiple,
                "fallback": fallback,
            },
            {
                "predictor_df": "sleep_apnea",
                "lookbehind_days": lookbehind_days,
                "resolve_multiple": resolve_multiple,
                "fallback": fallback,
            },
        ],
    )

from typing import Optional


def create_diag_feature_combinations(
    lookbehind_days=[365, 730, 1825, 9999],
    resolve_multiple=["mean", "max", "min"],
    fallback: Optional[any] = 0,
):
    """Create diagnosis feature combinations.

    Args:
        lookbehind_days (list, optional): List of lookbehind days. Defaults to [365, 730, 1825, 9999].
        resolve_multiple (list, optional): List of resolve multiple options. Defaults to ["mean", "max", "min"].
        fallback (any, optional): Fallback value. Defaults to 0.

    Returns:
        _type_: _description_
    """
    return [
        {
            "predictor_df": "essential_hypertension",
            "lookbehind_days": lookbehind_days,
            "resolve_multiple": resolve_multiple,
            "fallback": fallback,
            "allowed_nan_value_prop": 0.0,
        },
        {
            "predictor_df": "hyperlipidemia",
            "lookbehind_days": lookbehind_days,
            "resolve_multiple": resolve_multiple,
            "fallback": fallback,
            "allowed_nan_value_prop": 0.0,
        },
        {
            "predictor_df": "polycystic_ovarian_syndrome",
            "lookbehind_days": lookbehind_days,
            "resolve_multiple": resolve_multiple,
            "fallback": fallback,
            "allowed_nan_value_prop": 0.0,
        },
        {
            "predictor_df": "sleep_apnea",
            "lookbehind_days": lookbehind_days,
            "resolve_multiple": resolve_multiple,
            "fallback": fallback,
            "allowed_nan_value_prop": 0.0,
        },
    ]

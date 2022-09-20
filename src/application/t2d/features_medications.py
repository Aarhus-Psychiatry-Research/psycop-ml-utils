"""Loaders for T2D medication feature spec."""


def get_medication_feature_spec(  # pylint: disable=dangerous-default-value
    lookbehind_days: int = None,
    resolve_multiple: str = None,
    fallback=0,
):
    """Get feature specification for T2D medications.

    Args:
        lookbehind_days (int, optional): Defaults to None.
        resolve_multiple (str, optional): Defaults to None.
        fallback (int, optional): Defaults to 0.

    Returns:
        list: Feature specification.
    """

    if lookbehind_days is None:
        lookbehind_days = [365, 730, 1825, 9999]

    if resolve_multiple is None:
        resolve_multiple = ["mean", "max", "min"]

    return [
        {
            "predictor_df": "antipsychotics",
            "lookbehind_days": lookbehind_days,
            "resolve_multiple": resolve_multiple,
            "fallback": fallback,
            "allowed_nan_value_prop": 0.0,
        },
    ]

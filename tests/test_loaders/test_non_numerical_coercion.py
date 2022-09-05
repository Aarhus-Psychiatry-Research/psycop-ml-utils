import pandas as pd
from utils_for_testing import str_to_df

from psycopmlutils.loaders.non_numerical_coercer import multiply_inequalities_in_df


def test_non_numerical_coercion():
    df_str = """dw_ek_borger,timestamp,value
                1,2021-12-31 00:00:01,>90
                2,2021-12-31 00:00:01,>=90
                3,2021-12-31 00:00:01,<90
                4,2021-12-31 00:00:01,<=90
                5,2021-12-31 00:00:01,"<1,2"
            """

    expected_df_str = """dw_ek_borger,timestamp,value
                1,2021-12-31 00:00:01,135.0
                2,2021-12-31 00:00:01,108.0
                3,2021-12-31 00:00:01,60.0
                4,2021-12-31 00:00:01,72.0
                5,2021-12-31 00:00:01,1.0
            """

    df = str_to_df(df_str, convert_str_to_float=False)
    expected_df = str_to_df(expected_df_str, convert_str_to_float=False)

    df = multiply_inequalities_in_df(df)

    for col in df.columns:
        pd.testing.assert_series_equal(df[col], expected_df[col], check_dtype=False)

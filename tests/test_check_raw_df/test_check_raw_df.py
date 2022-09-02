from utils_for_testing import check_any_item_in_list_has_str, str_to_df

from psycopmlutils.loaders.raw.check_raw_df import check_raw_df


def test_raw_df_has_rows():
    df_str = """dw_ek_borger,timestamp,value
            """

    df = str_to_df(df_str)

    assert check_any_item_in_list_has_str(
        l=check_raw_df(df)[0],
        str_="No rows returned",
    )


def test_raw_df_has_required_cols():
    df_str = """dw_ek_borger,timstamp,value
            """

    df = str_to_df(df_str)

    assert check_any_item_in_list_has_str(l=check_raw_df(df)[0], str_="not in columns")


def test_raw_df_has_datetime_formatting():
    df_str = """dw_ek_borger,timestamp,value
                1,2021-01-01 00:00:00,1
            """

    df = str_to_df(df_str, convert_timestamp_to_datetime=False)

    assert check_any_item_in_list_has_str(
        l=check_raw_df(df)[0],
        str_="invalid datetime",
    )


def test_raw_df_has_expected_val_dtype():
    df_str = """dw_ek_borger,timestamp,value
                1,2021-01-01 00:00:00,a
            """

    df = str_to_df(df_str)

    assert check_any_item_in_list_has_str(l=check_raw_df(df)[0], str_="invalid dtype")


def test_raw_df_has_invalid_na_prop():
    df_str = """dw_ek_borger,timestamp,value
                1,2021-01-01 00:00:00,np.nan
            """

    df = str_to_df(df_str)

    assert check_any_item_in_list_has_str(l=check_raw_df(df)[0], str_="NaN")


def test_raw_df_has_duplicates():
    df_str = """dw_ek_borger,timestamp,value
                1,2021-01-01 00:00:00,np.nan
                1,2021-01-01 00:00:00,np.nan
            """

    df = str_to_df(df_str)

    assert check_any_item_in_list_has_str(l=check_raw_df(df)[0], str_="NaN")

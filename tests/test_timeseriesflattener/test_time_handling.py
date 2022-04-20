from timeseriesflattener.flattened_dataset import *

from utils_for_testing import *


def test_str_conversion():
    prediction_times_df_str = """dw_ek_borger,timestamp,
                            1,2021-12-31 00:00:00
                            """

    df_prediction_times = str_to_df(prediction_times_df_str)

    pass

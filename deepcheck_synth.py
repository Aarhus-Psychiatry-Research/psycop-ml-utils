import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity

from psycopmlutils.timeseriesflattener.data_integrity import (
    label_integrity_checks, label_split_checks)

if __name__ == "__main__":
    df = pd.read_csv(
        "https://raw.githubusercontent.com/Aarhus-Psychiatry-Research/psycop-t2d/main/tests/test_data/synth_prediction_data.csv"
    )
    df = df.drop(["Unnamed: 0", "citizen_ids", "timestamp_outcome"], axis=1)

    ds = Dataset(
        df,
        datetime_name="timestamp",
        label="outc_dichotomous_t2d_within_30_days_max_fallback_0",
    )

    integ_suite = label_integrity_checks()
    suite_result = integ_suite.run(ds)

    suite_result.show()

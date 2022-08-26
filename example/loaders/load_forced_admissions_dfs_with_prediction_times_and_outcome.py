import pandas as pd
from wasabi import msg

from psycopmlutils.loaders.sql_load import sql_load
from psycopmlutils.utils import data_loaders


class LoadForced:
    @data_loaders.register("forced_admissions_admissions")
    def forced_admissions_admissions():

        df = sql_load(
            "SELECT * FROM [fct].[psycop_fa_outcomes_all_disorders_tvangsindlaeg_Indlagt_2y_0f_2015-2021]",
            database="USR_PS_FORSK",
            chunksize=None,
        )

        df = df[["dw_ek_borger", "datotid_slut", "six_month"]]

        df.rename(
            columns={
                "datotid_slut": "timestamp",
                "six_month": "forced_admission_withn_6_months",
            },
            inplace=True,
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        msg.good(
            "Finished loading data frame for forced admissions with prediction times and outcome for all outpatient visits",
        )

        return df.reset_index(drop=True)

    @data_loaders.register("forced_admissions_outpatient_visits")
    def forced_admnissions_outpatient_visits():

        df = sql_load(
            "SELECT * FROM [fct].[psycop_fa_outcomes_all_disorders_tvangsindlaeg_Ambulant_2y_0f_2015-2021]",
            database="USR_PS_FORSK",
            chunksize=None,
        )

        df = df[["dw_ek_borger", "datotid_predict", "six_month"]]

        df.rename(
            columns={
                "datotid_predict": "timestamp",
                "six_month": "forced_admission_withn_6_months",
            },
            inplace=True,
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        msg.good(
            "Finished loading data frame for forced admissions with prediction times and outcome for all admissions",
        )

        return df.reset_index(drop=True)


if __name__ == "__main__":
    df = LoadForced.forced_admissions_admissions()

    pass

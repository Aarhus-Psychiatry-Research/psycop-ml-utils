import pandas as pd
from loaders import sql_load
from wasabi import msg


class LoadVisits:
    def physical_visits_to_psychiatry(frac=None):
        msg.info("Loading physical visits to psychiatry")

        view = "[FOR_besoeg_fysiske_fremmoeder_inkl_2021_feb2022]"
        sql = f"SELECT dw_ek_borger, datotid_start FROM [fct].{view} WHERE besoeg=1"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None)

        if frac is not None:
            df = df.sample(frac=frac)

        df.rename(columns={"datotid_start": "timestamp"}, inplace=True)

        msg.good("Loaded physical visits")
        return df.reset_index(drop=True)

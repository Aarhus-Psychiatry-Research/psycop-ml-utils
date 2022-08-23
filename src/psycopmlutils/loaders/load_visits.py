import pandas as pd
from wasabi import msg

from psycopmlutils.loaders.sql_load import sql_load


class LoadVisits:
    def physical_visits_to_psychiatry():
        # msg.info("Loading physical visits to psychiatry")

        # SHAK = 6600 ≈ in psychiatry
        d = {
            "LPR3": {
                "view": "[FOR_LPR3kontakter_psyk_somatik_inkl_2021_feb2022]",
                "datetime_col": "datotid_lpr3kontaktstart",
                "location_col": "shakkode_lpr3kontaktansvarlig",
                "where": "LIKE '6600%' AND pt_type in ('Ambulant', 'Akut ambulant', 'Indlæggelse')",
            },
            "LPR2_outpatient": {
                "view": "[FOR_besoeg_psyk_somatik_LPR2_inkl_2021_feb2022]",
                "datetime_col": "datotid_start",
                "location_col": "shakafskode",
                "where": "LIKE '6600%' AND psykambbesoeg = 1",
            },
            "LPR2_acute_outpatient": {
                "view": "[FOR_akutambulantekontakter_psyk_somatik_LPR2_inkl_2021_feb2022]",
                "datetime_col": "datotid_start",
                "location_col": "afsnit_stam",
                "where": " LIKE '6600%'",
            },
            "LPR2_admissions": {
                "view": "[FOR_indlaeggelser_psyk_somatik_LPR2_inkl_2021_feb2022]",
                "datetime_col": "datotid_indlaeggelse",
                "location_col": "shakKode_kontaktansvarlig",
                "where": "LIKE '6600%'",
            },
        }

        dfs = []

        for k, v in d.items():
            cols = f"{v['datetime_col']}, dw_ek_borger"

            sql = f"SELECT {cols} FROM [fct].{v['view']}"

            if "where" in v:
                sql += f" WHERE {v['location_col']} {v['where']}"

            df = sql_load(sql, database="USR_PS_FORSK", chunksize=None)
            df.rename(columns={v["datetime_col"]: "timestamp"}, inplace=True)

            dfs.append(df)

        # Concat the list of dfs
        output_df = pd.concat(dfs)
        msg.good("Loaded physical visits")

        return output_df.reset_index(drop=True)

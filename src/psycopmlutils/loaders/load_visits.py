from wasabi import msg

from psycopmlutils.loaders.sql_load import sql_load


class LoadVisits:
    def physical_visits_to_psychiatry():
        # msg.info("Loading physical visits to psychiatry")

        # LPR3
        d = {
            "LPR3": {
                "view": "[FOR_LPR3kontakter_psyk_somatik_inkl_2021_feb2022]",
                "columns": "dw_ek_borger, datotid_start",
                "where": "(SHAK=6600 and pt_type in ('indlæggelse', 'akut ambulant', 'ambulant')",
            },
            "LPR2_outpatient": {
                "view": "[FOR_besoeg_psyk_somatik_LPR2_inkl_2021_feb2022]",
                "columns": "dw_ek_borger, datotid_start",
                "where": "(SHAK=6600 and pt_type = 'psykambbesoeg')",
            },
            "LPR2_acute_outpatient": {
                "view": "[FOR_besoeg_psyk_somatik_LPR2_inkl_2021_feb2022]",
                "columns": "dw_ek_borger, datotid_start",
                "where": "SHAK=6600",
            },
            "LPR2_admissions": {
                "view": "[FOR_indlaeggelser_psyk_somatik_LPR2_inkl_2021_feb2022]",
                "columns": "dw_ek_borger, datotid_start",
                "where": "SHAK=6600",
            },
        }

        dfs = []

        for k, v in d.items:
            sql = f"SELECT {v.columns} FROM [fct].{v.view} WHERE {v.where}"
            df = sql_load(sql, database="USR_PS_FORSK", chunksize=None)
            df.rename(columns={"datotid_start": "timestamp"}, inplace=True)

            dfs.append(df)

        output_df = dfs.concat(axis=0)
        msg.good("Loaded physical visits")

        return output_df.reset_index(drop=True)

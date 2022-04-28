from typing import List, Union
import pandas as pd
from loaders.sql_load import sql_load
from loaders.str_utils import create_cols_for_unique_vals_at_depth
from wasabi import msg


class LoadDiagnoses:
    def diagnoses_from_physical_visits(
        icd_str: Union[List[str], str],
        new_col_str: str = None,
        depth: int = None,
        wildcard_icd_10_end: bool = False,
    ) -> pd.DataFrame:
        """Load diagnoses from all physical visits

        Args:
            icd_str (Union[List[str], str]): Substring(s) to match diagnoses for. Matches any diagnoses, whether a-diagnosis, b-diagnosis etc.
            new_col_str (str, optional): Name of new column string. Defaults to None.
            depth (int, optional): At which level to generate combinations. E.g. if depth = 3, A0004 and A0001 will both be A000,
                whereas depth = 4 would result in two different columns.
            depth (int, optional): At which level to generate combinations. E.g. if depth = 3, A0004 and A0001 will both be A000,
                whereas depth = 4 would result in two different columns.

        Returns:
            pd.DataFrame: _description_
        """

        if isinstance(icd_str, list):
            if new_col_str is None:
                raise ValueError(
                    "new_col_str is None while icd_str is a list. Must specify a name for the new column when aggregating multiple diagnoses."
                )

        print_str = f"diagnoses matching NPU-code {icd_str}"
        msg.info(f"Loading {print_str}")

        diagnoses_source_table_info = {
            "lpr3": {
                "fct": "FOR_LPR3kontakter_psyk_somatik_inkl_2021",
                "source_timestamp_col_name": "datotid_lpr3kontaktstart",
            },
            "lpr2_inpatient": {
                "fct": "FOR_indlaeggelser_psyk_somatik_LPR2_inkl_2021",
                "source_timestamp_col_name": "datotid_indlaeggelse",
            },
            "lpr2_outpatient": {
                "fct": "FOR_besoeg_psyk_somatik_LPR2_inkl_2021",
                "source_timestamp_col_name": "datotid_start",
            },
        }

        dfs = [
            LoadDiagnoses._load_diagnoses(
                icd_str=icd_str,
                depth=depth,
                new_col_str=new_col_str,
                wildcard_icd_10_end=wildcard_icd_10_end,
                **kwargs,
            )
            for source_name, kwargs in diagnoses_source_table_info.items()
        ]

        df = pd.concat(dfs)

        msg.good(f"Loaded {print_str}")
        return df.reset_index(drop=True)

    def _load_diagnoses(
        icd_str: Union[List[str], str],
        source_timestamp_col_name: str,
        fct: str,
        new_col_str: str = None,
        depth: int = None,
        wildcard_icd_10_end: bool = True,
    ) -> pd.DataFrame:
        """Load the visits that have diagnoses that match icd_str from the beginning of their adiagnosekode string.
        Aggregates all that match.

        Args:
            icd_str (Union[List[str], str]): Substring(s) to match diagnoses for. Matches any diagnoses, whether a-diagnosis, b-diagnosis etc.
            source_timestamp_col_name (str): Name of the timestamp column in the SQL table.
            view (str): Which view to use, e.g. "FOR_Medicin_ordineret_inkl_2021_feb2022"
            new_col_str (str, optional): Name of new column string. Defaults to None.
            depth (int, optional): At which level to generate combinations. E.g. if depth = 3, A0004 and A0001 will both be A000,
                whereas depth = 4 would result in two different columns.
            wildcard_icd_10_end (bool, optional): Whether to match on icd_str*. Defaults to true.

        Returns:
            pd.DataFrame: A pandas dataframe with dw_ek_borger, timestamp and new_col_str = 1
        """
        fct = f"[{fct}]"

        # Add a % at the end of the SQL match as a wildcard, so e.g. F20 matches F200.
        sql_ending = "%" if wildcard_icd_10_end else ""

        if isinstance(icd_str, list):
            match_col_sql_strings = [
                f"lower(diagnosegruppestreng) LIKE lower('%{diag_str}{sql_ending}')"
                for diag_str in icd_str
            ]

            match_col_sql_str = " OR ".join(match_col_sql_strings)
        else:
            match_col_sql_str = (
                f"lower(diagnosegruppestreng) LIKE lower('%{icd_str}{sql_ending})'"
            )

        sql = f"SELECT dw_ek_borger, {source_timestamp_col_name}, diagnosegruppestreng FROM [fct].{fct} WHERE ({match_col_sql_str})"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None)

        if new_col_str is None:
            new_col_str = icd_str

        # Handle depth
        if depth is None:
            df[new_col_str] = 1
        else:
            df["icd_match"] = df["diagnosegruppestreng"].str.extract(
                rf"({icd_str}[^#]*)", expand=False
            )

            df = create_cols_for_unique_vals_at_depth(
                df=df, source_col_name="icd_match", depth=depth
            )

            df.drop(["icd_match"], axis=1, inplace=True)

        df.drop(["diagnosegruppestreng"], axis="columns", inplace=True)

        return df.rename(
            columns={
                source_timestamp_col_name: "timestamp",
            }
        )

    def t2d_times():
        from pathlib import Path

        msg.info("Loading t2d event times")

        full_csv_path = Path(
            r"C:\Users\adminmanber\Desktop\manber-t2d\csv\first_t2d_diagnosis.csv"
        )

        df = pd.read_csv(str(full_csv_path))
        df = df[["dw_ek_borger", "datotid_first_t2d_diagnosis"]]
        df["val"] = 1

        df.rename(columns={"datotid_first_t2d_diagnosis": "timestamp"}, inplace=True)

        msg.good("Finished loading t2d event times")
        return df.reset_index(drop=True)

    def essential_hypertension():
        return LoadDiagnoses.diagnoses_from_physical_visits(
            icd_str="I109", wildcard_icd_10_end=False
        )

    def hyperlipidemia():
        return LoadDiagnoses.diagnoses_from_physical_visits(
            icd_str=[
                "E780",
                "E785",
            ],  # Only these two, as the others are exceedingly rare
            new_col_str="hyperlipidemia",
            wildcard_icd_10_end=False,
        )

    def liverdisease_uns():
        return LoadDiagnoses.diagnoses_from_physical_visits(
            icd_str="K769",
            wildcard_icd_10_end=False,
        )

    def polycystic_overian_syndrom():
        return LoadDiagnoses.diagnoses_from_physical_visits(
            icd_str="E282",
            wildcard_icd_10_end=False,
        )

    def sleep_apnea():
        return LoadDiagnoses.diagnoses_from_physical_visits(
            icd_str=["G473", "G4732"],
            wildcard_icd_10_end=False,
        )

    def sleep_problems_unspecified():
        return LoadDiagnoses.diagnoses_from_physical_visits(
            icd_str="G479",
            wildcard_icd_10_end=False,
        )
    

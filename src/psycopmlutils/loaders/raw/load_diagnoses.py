from typing import List, Optional, Union

import pandas as pd

from psycopmlutils.loaders.raw.sql_load import sql_load
from psycopmlutils.utils import data_loaders


class LoadDiagnoses:
    def concat_from_physical_visits(
        icd_codes: List[str],
        output_col_name: str,
        wildcard_icd_10_end: Optional[bool] = False,
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load all diagnoses matching any icd_code in icd_codes. Create
        output_col_name and set to 1.

        Args:
            icd_codes (List[str]): List of icd_codes. # noqa: DAR102
            output_col_name (str): Output column name
            wildcard_icd_10_end (bool, optional): Whether to match on icd_codes* or icd_codes. Defaults to False.
            n: Number of rows to return. Defaults to None.

        Returns:
            pd.DataFrame
        """

        diagnoses_source_table_info = {
            "lpr3": {
                "fct": "FOR_LPR3kontakter_psyk_somatik_inkl_2021_feb2022",
                "source_timestamp_col_name": "datotid_lpr3kontaktstart",
            },
            "lpr2_inpatient": {
                "fct": "FOR_indlaeggelser_psyk_somatik_LPR2_inkl_2021_feb2022",
                "source_timestamp_col_name": "datotid_indlaeggelse",
            },
            "lpr2_acute_outpatient": {
                "fct": "FOR_akutambulantekontakter_psyk_somatik_LPR2_inkl_2021_feb2022",
                "source_timestamp_col_name": "datotid_start",
            },
            "lpr2_outpatient": {
                "fct": "FOR_besoeg_psyk_somatik_LPR2_inkl_2021_feb2022",
                "source_timestamp_col_name": "datotid_start",
            },
        }

        # Using ._load is faster than from_physical_visits since it can process all icd_codes in the SQL request at once,
        # rather than processing one at a time and aggregating.
        dfs = [
            LoadDiagnoses._load(
                icd_code=icd_codes,
                output_col_name=output_col_name,
                wildcard_icd_code=wildcard_icd_10_end,
                n=n,
                **kwargs,
            )
            for source_name, kwargs in diagnoses_source_table_info.items()
        ]

        df = pd.concat(dfs).drop_duplicates(
            subset=["dw_ek_borger", "timestamp", "value"],
            keep="first",
        )
        return df.reset_index(drop=True)

    def from_physical_visits(
        icd_code: str,
        output_col_name: Optional[str] = "value",
        n: Optional[int] = None,
        wildcard_icd_code: Optional[bool] = False,
    ) -> pd.DataFrame:
        """Load diagnoses from all physical visits. If icd_code is a list, will
        aggregate as one column (e.g. ["E780", "E785"] into a
        ypercholesterolemia column).

        Args:
            icd_code (str): Substring to match diagnoses for. Matches any diagnoses, whether a-diagnosis, b-diagnosis etc. # noqa: DAR102
            output_col_name (str, optional): Name of new column string. Defaults to "value".
            n: Number of rows to return. Defaults to None.
            wildcard_icd_code (bool, optional): Whether to match on icd_code*. Defaults to False.

        Returns:
            pd.DataFrame
        """

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
            LoadDiagnoses._load(
                icd_code=icd_code,
                output_col_name=output_col_name,
                wildcard_icd_code=wildcard_icd_code,
                n=n,
                **kwargs,
            )
            for source_name, kwargs in diagnoses_source_table_info.items()
        ]

        df = pd.concat(dfs)

        return df.reset_index(drop=True)

    def _load(
        icd_code: Union[List[str], str],
        source_timestamp_col_name: str,
        fct: str,
        output_col_name: Optional[str] = None,
        wildcard_icd_code: Optional[bool] = True,
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load the visits that have diagnoses that match icd_code from the
        beginning of their adiagnosekode string. Aggregates all that match.

        Args:
            icd_code (Union[List[str], str]): Substring(s) to match diagnoses for. # noqa: DAR102
                Matches any diagnoses, whether a-diagnosis, b-diagnosis etc. If a list is passed, will
                count a diagnosis as a match if any of the icd_codes in the list match.
            source_timestamp_col_name (str): Name of the timestamp column in the SQL
                view.
            fct (str): Name of the SQL view to load from.
            output_col_name (str, optional): Name of new column string. Defaults to
                None.
            wildcard_icd_code (bool, optional): Whether to match on icd_code*.
                Defaults to true.
            n: Number of rows to return. Defaults to None.

        Returns:
            pd.DataFrame: A pandas dataframe with dw_ek_borger, timestamp and
                output_col_name = 1
        """
        fct = f"[{fct}]"

        # Must be able to split a string like this:
        #   A:DF431#+:ALFC3#B:DF329
        # Which means that if wildcard_icd_10_end is False, we must match on icd_code# or icd_code followed by nothing.
        # If it's true, we can match on icd_code*.

        # Handle if there are multiple ICD codes to count together.
        if isinstance(icd_code, list):
            match_col_sql_strings = []

            for code_str in icd_code:
                if wildcard_icd_code:
                    match_col_sql_strings.append(
                        f"lower(diagnosegruppestreng) LIKE '%{code_str.lower()}%'",
                    )
                else:
                    # If the string is at the end of diagnosegruppestreng, it doesn't end with a hashtag
                    match_col_sql_strings.append(
                        f"lower(diagnosegruppestreng) LIKE '%{code_str.lower()}'",
                    )

                    # But if it is at the end, it does
                    match_col_sql_strings.append(
                        f"lower(diagnosegruppestreng) LIKE '%{code_str.lower()}#%'",
                    )

            match_col_sql_str = " OR ".join(match_col_sql_strings)
        else:
            if wildcard_icd_code:
                match_col_sql_str = (
                    f"lower(diagnosegruppestreng) LIKE '%{icd_code.lower()}%'"
                )

            else:
                match_col_sql_str = f"lower(diagnosegruppestreng) LIKE '%{icd_code.lower()}' OR lower(diagnosegruppestreng) LIKE '%{icd_code.lower()}#%'"

        sql = (
            f"SELECT dw_ek_borger, {source_timestamp_col_name}, diagnosegruppestreng"
            + f" FROM [fct].{fct} WHERE {source_timestamp_col_name} IS NOT NULL AND ({match_col_sql_str})"
        )

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None, n=n)

        if output_col_name is None:
            output_col_name = icd_code

        df[output_col_name] = 1

        df.drop(["diagnosegruppestreng"], axis="columns", inplace=True)

        return df.rename(
            columns={
                source_timestamp_col_name: "timestamp",
            },
        )

    @data_loaders.register("essential_hypertension")
    def essential_hypertension(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="I109",
            wildcard_icd_code=False,
            n=n,
        )

    @data_loaders.register("hyperlipidemia")
    def hyperlipidemia(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=[
                "E780",
                "E785",
            ],  # Only these two, as the others are exceedingly rare
            wildcard_icd_code=False,
            n=n,
        )

    @data_loaders.register("liverdisease_unspecified")
    def liverdisease_unspecified(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="K769",
            wildcard_icd_code=False,
            n=n,
        )

    @data_loaders.register("polycystic_ovarian_syndrome")
    def polycystic_ovarian_syndrome(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="E282",
            wildcard_icd_code=False,
            n=n,
        )

    @data_loaders.register("sleep_apnea")
    def sleep_apnea(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["G473", "G4732"],
            wildcard_icd_code=False,
            n=n,
        )

    @data_loaders.register("sleep_problems_unspecified")
    def sleep_problems_unspecified(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="G479",
            wildcard_icd_code=False,
            n=n,
        )

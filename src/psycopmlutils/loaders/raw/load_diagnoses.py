from typing import List, Optional, Union

import pandas as pd

from psycopmlutils.loaders.raw.sql_load import sql_load
from psycopmlutils.utils import data_loaders


class LoadDiagnoses:
    def concat_from_physical_visits(
        icd_codes: List[str],
        output_col_name: str,
        wildcard_icd_code: Optional[bool] = False,
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load all diagnoses matching any icd_code in icd_codes. Create
        output_col_name and set to 1.

        Args:
            icd_codes (List[str]): List of icd_codes. # noqa: DAR102
            output_col_name (str): Output column name
            wildcard_icd_code (bool, optional): Whether to match on icd_codes* or icd_codes. Defaults to False.
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
                wildcard_icd_code=wildcard_icd_code,
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

        n_per_df = int(n / len(diagnoses_source_table_info))

        dfs = [
            LoadDiagnoses._load(
                icd_code=icd_code,
                output_col_name=output_col_name,
                wildcard_icd_code=wildcard_icd_code,
                n=n_per_df,
                **kwargs,
            )
            for source_name, kwargs in diagnoses_source_table_info.items()
        ]

        df = pd.concat(dfs).drop_duplicates(
            subset=["dw_ek_borger", "timestamp", "value"],
            keep="first",
        )

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
        # Which means that if wildcard_icd_code is False, we must match on icd_code# or icd_code followed by nothing.
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
        
    # Psychiatric diagnoses

    # data loaders for all diagnoses in the f0-chapter (organic mental disorders)
    @data_loaders.register("f0_disorders")
    def f0_disorders(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="f0",
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("dementia")
    def dementia(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f00", "f01", "f02", "f03", "f04"],
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("delirium")
    def delirium(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="f05",
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("miscellaneous organic mental disorders")
    def misc_organic_mental_disorders(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f06", "f07", "f09"],
            wildcard_icd_code=True,
            n=n,
        )

    # data loaders for all diagnoses in the f1-chapter (mental and behavioural disorders due to psychoactive substance use)
    @data_loaders.register("f1_disorders")
    def f1_disorders(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="f1",
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("alcohol and tobacco dependencies")
    def alcohol_and_tobacco(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f10", "f17"],
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("opioid and sedative dependencies")
    def opioids_and_sedatives(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f11", "f13"],
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("cannabinoid and hallucinogen dependencies")
    def cannabinoids_and_hallucinogens(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f12", "f16"],
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("stimulant dependencies")
    def stimulants(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f14", "f15"],
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("miscellaneous drug dependencies")
    def misc_drugs(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f18", "f19"],
            wildcard_icd_code=True,
            n=n,
        )

    # data loaders for all diagnoses in the f2-chapter (schizophrenia, schizotypal and delusional disorders)

    @data_loaders.register("f2_disorders")
    def f2_disorders(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="f2",
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("schizophrenia")
    def schizophrenia(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f20", "f21"],
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("schizoaffective")
    def schizoaffective(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="f25",
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("miscellaneous psychotic disorders")
    def misc_psychosis(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f22", "f23", "f24", "f26", "f27", "f28", "f29"],
            wildcard_icd_code=True,
            n=n,
        )

    # data loaders for all diagnoses in the f3-chapter (mood (affective) disorders).

    @data_loaders.register("f3_disorders")
    def f3_disorders(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="f3",
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("manic and bipolar")
    def manic_and_bipolar(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f30", "f31"],
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("depressive disorders")
    def depressive_disorders(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f32", "f33", "f34", "f38"],
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("miscellaneous affective disorders")
    def misc_affective_disorders(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f38", "f39"],
            wildcard_icd_code=True,
            n=n,
        )

    # data loaders for all diagnoses in the f4-chapter (neurotic, stress-related and somatoform disorders).

    @data_loaders.register("f4_disorders")
    def f4_disorders(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="f4",
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("phobic,anxiety and ocd")
    def phobic_and_anxiety(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f40", "f41", "f42"],
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("reaction to severe stress and adjustment disorders")
    def stress_and_adjustment(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="f43",
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("dissociative, somatoform and miscellaneous")
    def dissociative_somatoform_and_misc(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f44", "f45", "f48"],
            wildcard_icd_code=True,
            n=n,
        )

    # data loaders for all diagnoses in the f5-chapter (behavioural syndromes associated with physiological disturbances and physical factors).

    @data_loaders.register("f5_disorders")
    def f5_disorders(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="f5",
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("eating disorders")
    def eating_disorders(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="f50",
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("sleeping and sexual disorders")
    def sleeping_and_sexual_disorders(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f51", "f52"],
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("miscellaneous f5 disorders")
    def misc_f5(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f53", "f54", "f55", "f59"],
            wildcard_icd_code=True,
            n=n,
        )

    # data loaders for all diagnoses in the f6-chapter (disorders of adult personality and behaviour).
    @data_loaders.register("f6_disorders")
    def f6_disorders(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="f6",
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("cluster_a")
    def cluster_a(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f600", "f601"],
            wildcard_icd_code=False,
            n=n,
        )

    @data_loaders.register("cluster_b")
    def cluster_b(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f602", "f603", "f604"],
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("cluster_c")
    def cluster_c(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f605", "f606", "f607"],
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("miscellaneous personality disorders")
    def misc_personality_disorders(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f608", "f609", "f61", "f62", "f63", "f68", "f69"],
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("sexual disorders")
    def misc_personality(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f65", "f66"],
            wildcard_icd_code=True,
            n=n,
        )

        # should we exclude sexual identity disorders? f64

    # data loaders for all diagnoses in the f7-chapter (mental retardation).
    @data_loaders.register("f7_disorders")
    def f7_disorders(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="f7",
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("mild mental retardation")
    def mild_mental_retardation(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="f70",
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("moderate mental retardation")
    def moderate_mental_retardation(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="f71",
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("severe mental retardation")
    def severe_mental_retardation(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f72", "f73"],
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("miscellaneous mental retardation disorders")
    def misc_mental_retardation(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f78", "f79"],
            wildcard_icd_code=True,
            n=n,
        )

    # data loaders for all diagnoses in the f8-chapter (disorders of psychological development).
    @data_loaders.register("f8_disorders")
    def f8_disorders(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="f8",
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("pervasive developmental disorders")
    def pervasive_developmental_disorders(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="f84",
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("miscellaneous f8 disorders")
    def misc_f8(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f80", "f81", "f82", "f83", "f88", "f89"],
            wildcard_icd_code=True,
            n=n,
        )

    # data loaders for all diagnoses in the f9-chapter (child and adolescent disorders).
    @data_loaders.register("f9_disorders")
    def f9_disorders(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="f9",
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("hyperkinetic disorders")
    def hyperkinetic_disorders(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code="f90",
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("behavioural disorders")
    def behavioural_disorders(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f91", "f92", "f93", "f94"],
            wildcard_icd_code=True,
            n=n,
        )

    @data_loaders.register("tics and miscellaneous f9")
    def tics_and_misc(n: Optional[int] = None) -> pd.DataFrame:
        return LoadDiagnoses.from_physical_visits(
            icd_code=["f95", "f98"],
            wildcard_icd_code=True,
            n=n,
        )

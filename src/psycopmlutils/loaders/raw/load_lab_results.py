from typing import Dict, Optional

import pandas as pd

from psycopmlutils.loaders.non_numerical_coercer import multiply_inequalities_in_df
from psycopmlutils.loaders.raw.sql_load import sql_load
from psycopmlutils.utils import data_loaders


class LoadLabResults:
    def concatenate_blood_samples(
        blood_sample_ids: list,
        n: Optional[int] = None,
        values_to_load: str = "all",
    ) -> pd.DataFrame:
        """Concatenate multiple blood_sample_ids (typically NPU-codes) into one
        column. Makes sense for similar blood samples with different NPU codes,
        e.g. a scheduled blood-glucose at 9 a.m. has a different NPU-code from
        one at 11 a.m.

        Args:
            blood_sample_ids (list): List of blood_sample_id, typically an NPU-codes. # noqa: DAR102
            n (int): Number of rows to return. Defaults to None.
            values_to_load (str): Which values to load. Takes either "numerical", "numerical_and_coerce", "cancelled" or "all". Defaults to "all".

        Returns:
            pd.DataFrame
        """
        allowed_values_to_load = [
            "numerical",
            "numerical_and_coerce",
            "cancelled",
            "all",
        ]

        if values_to_load not in allowed_values_to_load:
            raise ValueError(f"values_to_load must be one of {allowed_values_to_load}")

        n_per_df = int(n / len(blood_sample_ids))

        dfs = [
            LoadLabResults.blood_sample(
                blood_sample_id=f"{id}",
                n=n_per_df,
                values_to_load=values_to_load,
            )
            for id in blood_sample_ids
        ]

        return (
            pd.concat(dfs, axis=0)
            .drop_duplicates(
                subset=["timestamp", "dw_ek_borger", "value"],
                keep="first",
            )
            .reset_index(drop=True)
        )

    def blood_sample(
        blood_sample_id: str,
        n: Optional[int] = None,
        values_to_load: str = "all",
    ) -> pd.DataFrame:
        """Load a blood sample.

        Args:
            blood_sample_id (str): The blood_sample_id, typically an NPU code. # noqa: DAR102
            n: Number of rows to return. Defaults to None.
            values_to_load (str): Which values to load. Takes either "numerical", "numerical_and_coerce", "cancelled" or "all". Defaults to "all".

        Returns:
            pd.DataFrame
        """
        view = "[FOR_labka_alle_blodprover_inkl_2021_feb2022]"

        allowed_values_to_load = [
            "numerical",
            "numerical_and_coerce",
            "cancelled",
            "all",
        ]

        dfs = []

        if values_to_load not in allowed_values_to_load:
            raise ValueError(
                f"values_to_load must be one of {allowed_values_to_load}, not {values_to_load}",
            )

        fn_dict = {
            "coerce": LoadLabResults.load_non_numerical_values_and_coerce_inequalities,
            "numerical": LoadLabResults.load_numerical_values,
            "cancelled": LoadLabResults.load_cancelled,
            "all": LoadLabResults.load_all_values,
        }

        for k in fn_dict.keys():
            if k in values_to_load:
                dfs.append(fn_dict[k](blood_sample_id=blood_sample_id, n=n, view=view))

        # Concatenate dfs
        if len(dfs) > 1:
            df = pd.concat(dfs)
        else:
            df = dfs[0]

        return df.reset_index(drop=True).drop_duplicates(
            subset=["dw_ek_borger", "timestamp", "value"],
            keep="first",
        )

    def load_non_numerical_values_and_coerce_inequalities(
        blood_sample_id: str,
        n: int,
        view: str,
        ineq2mult: Dict[str, float] = None,
    ) -> pd.DataFrame:
        """Load non-numerical values for a blood sample.

        Args:
            blood_sample_id (str): The blood_sample_id, typically an NPU code.  # noqa: DAR102
            n: Number of rows to return. Defaults to None.
            view (str): The view to load from.
            ineq2mult (Dict[str, float]): A dictionary mapping inequalities to a multiplier. Defaults to None.

        Returns:
            pd.DataFrame: A dataframe with the non-numerical values.
        """
        cols = "dw_ek_borger, datotid_sidstesvar, svar"
        sql = f"SELECT {cols} FROM [fct].{view} WHERE npukode = '{blood_sample_id}' AND numerisksvar IS NULL AND (left(Svar,1) = '>' OR left(Svar, 1) = '<')"
        df = sql_load(
            sql,
            database="USR_PS_FORSK",
            chunksize=None,
            n=n,
        )

        df.rename(
            columns={"datotid_sidstesvar": "timestamp", "svar": "value"},
            inplace=True,
        )

        if ineq2mult:
            return multiply_inequalities_in_df(df, ineq2mult=ineq2mult)
        else:
            return multiply_inequalities_in_df(df)

    def load_numerical_values(blood_sample_id: str, n: int, view: str) -> pd.DataFrame:
        """Load numerical values for a blood sample.

        Args:
            blood_sample_id (str): The blood_sample_id, typically an NPU code.  # noqa: DAR102
            n (int): Number of rows to return. Defaults to None.
            view (str): The view to load from.

        Returns:
            pd.DataFrame: A dataframe with the numerical values.
        """

        cols = "dw_ek_borger, datotid_sidstesvar, numerisksvar"
        sql = f"SELECT {cols} FROM [fct].{view} WHERE npukode = '{blood_sample_id}' AND numerisksvar IS NOT NULL"
        df = sql_load(
            sql,
            database="USR_PS_FORSK",
            chunksize=None,
            n=n,
        )

        df.rename(
            columns={"datotid_sidstesvar": "timestamp", "numerisksvar": "value"},
            inplace=True,
        )

        return df

    def load_cancelled(blood_sample_id: str, n: int, view: str) -> pd.DataFrame:
        """Load cancelled samples for a blood sample.

        Args:
            blood_sample_id (str): The blood_sample_id, typically an NPU code.  # noqa: DAR102
            n: Number of rows to return. Defaults to None.
            view (str): The view to load from.

        Returns:
            pd.DataFrame: A dataframe with the timestamps for cancelled values.
        """
        cols = "dw_ek_borger, datotid_sidstesvar"
        sql = f"SELECT {cols} FROM [fct].{view} WHERE npukode = '{blood_sample_id}' AND Svar == 'Aflyst' AND (left(Svar,1) == '>' OR left(Svar, 1) == '<')"

        df = sql_load(
            sql,
            database="USR_PS_FORSK",
            chunksize=None,
            n=n,
        )

        # Create the value column == 1, since all timestamps here are from cancelled blood samples
        df["value"] = 1

        df.rename(
            columns={"datotid_sidstesvar": "timestamp"},
            inplace=True,
        )

        return df

    def load_all_values(blood_sample_id: str, n: int, view: str) -> pd.DataFrame:
        """Load all samples for a blood sample.

        Args:
            blood_sample_id (str): The blood_sample_id, typically an NPU code.  # noqa: DAR102
            n: Number of rows to return. Defaults to None.
            view (str): The view to load from.

        Returns:
            pd.DataFrame: A dataframe with all values.
        """
        cols = "dw_ek_borger, datotid_sidstesvar, svar"
        sql = f"SELECT {cols} FROM [fct].{view} WHERE npukode = '{blood_sample_id}'"

        df = sql_load(
            sql,
            database="USR_PS_FORSK",
            chunksize=None,
            n=n,
        )

        df.rename(
            columns={"datotid_sidstesvar": "timestamp", "svar": "value"},
            inplace=True,
        )

        return df

    @data_loaders.register("hba1c")
    def hba1c(n: Optional[int] = None, values_to_load: str = "all") -> pd.DataFrame:
        return LoadLabResults.blood_sample(
            blood_sample_id="NPU27300",
            n=n,
            values_to_load=values_to_load,
        )

    @data_loaders.register("scheduled_glc")
    def scheduled_glc(
        n: Optional[int] = None,
        values_to_load: str = "all",
    ) -> pd.DataFrame:
        npu_suffixes = [
            "08550",
            "08551",
            "08552",
            "08553",
            "08554",
            "08555",
            "08556",
            "08557",
            "08558",
            "08559",
            "08560",
            "08561",
            "08562",
            "08563",
            "08564",
            "08565",
            "08566",
            "08567",
            "08893",
            "08894",
            "08895",
            "08896",
            "08897",
            "08898",
            "08899",
            "08900",
            "08901",
            "08902",
            "08903",
            "08904",
            "08905",
            "08906",
            "08907",
            "08908",
            "08909",
            "08910",
            "08911",
            "08912",
            "08913",
            "08914",
            "08915",
            "08916",
        ]

        blood_sample_ids = [f"NPU{suffix}" for suffix in npu_suffixes]

        return LoadLabResults.concatenate_blood_samples(
            blood_sample_ids=blood_sample_ids,
            n=n,
            values_to_load=values_to_load,
        )

    @data_loaders.register("unscheduled_p_glc")
    def unscheduled_p_glc(
        n: Optional[int] = None,
        values_to_load: str = "all",
    ) -> pd.DataFrame:
        npu_suffixes = [
            "02192",
            "21533",
            "21531",
        ]

        dnk_suffixes = ["35842"]

        blood_sample_ids = [f"NPU{suffix}" for suffix in npu_suffixes]
        blood_sample_ids += [f"DNK{suffix}" for suffix in dnk_suffixes]

        return LoadLabResults.concatenate_blood_samples(
            blood_sample_ids=blood_sample_ids,
            n=n,
            values_to_load=values_to_load,
        )

    @data_loaders.register("triglycerides")
    def triglycerides(
        n: Optional[int] = None,
        values_to_load: str = "all",
    ) -> pd.DataFrame:
        return LoadLabResults.blood_sample(
            blood_sample_id="NPU04094",
            n=n,
            values_to_load=values_to_load,
        )

    @data_loaders.register("fasting_triglycerides")
    def fasting_triglycerides(
        n: Optional[int] = None,
        values_to_load: str = "all",
    ) -> pd.DataFrame:
        return LoadLabResults.blood_sample(
            blood_sample_id="NPU03620",
            n=n,
            values_to_load=values_to_load,
        )

    @data_loaders.register("hdl")
    def hdl(n: Optional[int] = None, values_to_load: str = "all") -> pd.DataFrame:
        return LoadLabResults.blood_sample(
            blood_sample_id="NPU01567",
            n=n,
            values_to_load=values_to_load,
        )

    @data_loaders.register("ldl")
    def ldl(n: Optional[int] = None, values_to_load: str = "all") -> pd.DataFrame:
        return LoadLabResults.concatenate_blood_samples(
            blood_sample_ids=["NPU01568", "AAB00101"],
            n=n,
            values_to_load=values_to_load,
        )

    @data_loaders.register("fasting_ldl")
    def fasting_ldl(
        n: Optional[int] = None,
        values_to_load: str = "all",
    ) -> pd.DataFrame:
        return LoadLabResults.concatenate_blood_samples(
            blood_sample_ids=["NPU10171", "AAB00102"],
            n=n,
            values_to_load=values_to_load,
        )

    @data_loaders.register("alat")
    def alat(n: Optional[int] = None, values_to_load: str = "all") -> pd.DataFrame:
        return LoadLabResults.blood_sample(
            blood_sample_id="NPU19651",
            n=n,
            values_to_load=values_to_load,
        )

    @data_loaders.register("asat")
    def asat(n: Optional[int] = None, values_to_load: str = "all") -> pd.DataFrame:
        return LoadLabResults.blood_sample(
            blood_sample_id="NPU19654",
            n=n,
            values_to_load=values_to_load,
        )

    @data_loaders.register("lymphocytes")
    def lymphocytes(
        n: Optional[int] = None,
        values_to_load: str = "all",
    ) -> pd.DataFrame:
        return LoadLabResults.blood_sample(
            blood_sample_id="NPU02636",
            n=n,
            values_to_load=values_to_load,
        )

    @data_loaders.register("leukocytes")
    def leukocytes(
        n: Optional[int] = None,
        values_to_load: str = "all",
    ) -> pd.DataFrame:
        return LoadLabResults.blood_sample(
            blood_sample_id="NPU02593",
            n=n,
            values_to_load=values_to_load,
        )

    @data_loaders.register("crp")
    def crp(n: Optional[int] = None, values_to_load: str = "all") -> pd.DataFrame:
        return LoadLabResults.blood_sample(
            blood_sample_id="NPU19748",
            n=n,
            values_to_load=values_to_load,
        )

    @data_loaders.register("creatinine")
    def creatinine(
        n: Optional[int] = None,
        values_to_load: str = "all",
    ) -> pd.DataFrame:
        return LoadLabResults.concatenate_blood_samples(
            blood_sample_ids=["NPU18016", "ASS00355", "ASS00354"],
            n=n,
        )

    @data_loaders.register("egfr")
    def egfr(n: Optional[int] = None, values_to_load: str = "all") -> pd.DataFrame:
        return LoadLabResults.concatenate_blood_samples(
            blood_sample_ids=["DNK35302", "DNK35131", "AAB00345", "AAB00343"],
            n=n,
            values_to_load=values_to_load,
        )

    @data_loaders.register("albumine_creatinine_ratio")
    def albumine_creatinine_ratio(
        n: Optional[int] = None,
        values_to_load: str = "all",
    ) -> pd.DataFrame:
        return LoadLabResults.blood_sample(
            blood_sample_id="NPU19661",
            n=n,
            values_to_load=values_to_load,
        )

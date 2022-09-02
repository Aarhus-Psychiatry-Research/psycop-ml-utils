from typing import Optional

import pandas as pd

from psycopmlutils.loaders.raw.sql_load import sql_load
from psycopmlutils.utils import data_loaders


class LoadLabResults:
    def blood_sample(
        blood_sample_id: str,
        numerical_only: bool = False,
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load a blood sample.

        Args:
            blood_sample_id (str): The blood_sample_id, typically an NPU code. # noqa: DAR102
            numerical_only (bool): Whether to only return numerical values. Defaults toFalse.
            n: Number of rows to return. Defaults to None.

        Returns:
            pd.DataFrame
        """
        view = "[FOR_labka_alle_blodprover_inkl_2021_feb2022]"

        if numerical_only:
            val_col = "numerisksvar"
            where = f" AND {val_col} IS NOT NULL"
        else:
            val_col = "svar"
            where = ""

        columns = f"dw_ek_borger, datotid_sidstesvar, {val_col}"

        sql = f"SELECT {columns} FROM [fct].{view} WHERE npukode = '{blood_sample_id}' {where}"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None, n=n)

        df.rename(
            columns={"datotid_sidstesvar": "timestamp", val_col: "value"},
            inplace=True,
        )

        # Convert all values matching X,Y to X.Y and then to float
        if not numerical_only:
            df["value"] = df["value"].str.replace(",", ".")

        return df.reset_index(drop=True)

    def _concatenate_blood_samples(
        blood_sample_ids: list,
        n: Optional[int] = None,
        numerical_only: bool = False,
    ) -> pd.DataFrame:
        """Concatenate multiple blood_sample_ids (typically NPU-codes) into one
        column.

        Args:
            blood_sample_ids (list): List of blood_sample_id, typically an NPU-codes. # noqa: DAR102
            n (int, optional): Number of rows to return. Defaults to None.
            numerical_only (bool): Whether to only return numerical values. Defaults to False.

        Returns:
            pd.DataFrame
        """
        n_per_df = int(n / len(blood_sample_ids))

        dfs = [
            LoadLabResults.blood_sample(
                blood_sample_id=f"{id}",
                n=n_per_df,
                numerical_only=numerical_only,
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

    @data_loaders.register("hba1c")
    def hba1c(n: Optional[int] = None) -> pd.DataFrame:
        return LoadLabResults.blood_sample(
            blood_sample_id="NPU27300",
            n=n,
            numerical_only=True,
        )

    @data_loaders.register("scheduled_glc")
    def scheduled_glc(n: Optional[int] = None) -> pd.DataFrame:
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

        return LoadLabResults._concatenate_blood_samples(
            blood_sample_ids=blood_sample_ids,
            n=n,
        )

    @data_loaders.register("unscheduled_p_glc")
    def unscheduled_p_glc(n: Optional[int] = None) -> pd.DataFrame:
        npu_suffixes = [
            "02192",
            "21533",
            "21531",
        ]

        dnk_suffixes = ["35842"]

        blood_sample_ids = [f"NPU{suffix}" for suffix in npu_suffixes]
        blood_sample_ids += [f"DNK{suffix}" for suffix in dnk_suffixes]

        return LoadLabResults._concatenate_blood_samples(
            blood_sample_ids=blood_sample_ids,
            n=n,
        )

    @data_loaders.register("triglycerides")
    def triglycerides(n: Optional[int] = None) -> pd.DataFrame:
        return LoadLabResults.blood_sample(blood_sample_id="NPU04094", n=n)

    @data_loaders.register("fasting_triglycerides")
    def fasting_triglycerides(n: Optional[int] = None) -> pd.DataFrame:
        return LoadLabResults.blood_sample(blood_sample_id="NPU03620", n=n)

    @data_loaders.register("hdl")
    def hdl(n: Optional[int] = None) -> pd.DataFrame:
        return LoadLabResults.blood_sample(blood_sample_id="NPU01567", n=n)

    @data_loaders.register("ldl")
    def ldl(n: Optional[int] = None) -> pd.DataFrame:
        return LoadLabResults._concatenate_blood_samples(
            blood_sample_ids=["NPU01568", "AAB00101"],
            n=n,
        )

    @data_loaders.register("fasting_ldl")
    def fasting_ldl(n: Optional[int] = None) -> pd.DataFrame:
        return LoadLabResults._concatenate_blood_samples(
            blood_sample_ids=["NPU10171", "AAB00102"],
            n=n,
        )

    @data_loaders.register("alat")
    def alat(n: Optional[int] = None) -> pd.DataFrame:
        return LoadLabResults.blood_sample(blood_sample_id="NPU19651", n=n)

    @data_loaders.register("asat")
    def asat(n: Optional[int] = None) -> pd.DataFrame:
        return LoadLabResults.blood_sample(blood_sample_id="NPU19654", n=n)

    @data_loaders.register("lymphocytes")
    def lymphocytes(n: Optional[int] = None) -> pd.DataFrame:
        return LoadLabResults.blood_sample(blood_sample_id="NPU02636", n=n)

    @data_loaders.register("leukocytes")
    def leukocytes(n: Optional[int] = None) -> pd.DataFrame:
        return LoadLabResults.blood_sample(blood_sample_id="NPU02593", n=n)

    @data_loaders.register("crp")
    def crp(n: Optional[int] = None) -> pd.DataFrame:
        return LoadLabResults.blood_sample(blood_sample_id="NPU19748", n=n)

    @data_loaders.register("creatinine")
    def creatinine(n: Optional[int] = None) -> pd.DataFrame:
        return LoadLabResults._concatenate_blood_samples(
            blood_sample_ids=["NPU18016", "ASS00355", "ASS00354"],
            n=n,
        )

    @data_loaders.register("egfr")
    def egfr(n: Optional[int] = None) -> pd.DataFrame:
        return LoadLabResults._concatenate_blood_samples(
            blood_sample_ids=["DNK35302", "DNK35131", "AAB00345", "AAB00343"],
            n=n,
        )

    @data_loaders.register("albumine_creatinine_ratio")
    def albumine_creatinine_ratio(n: Optional[int] = None) -> pd.DataFrame:
        return LoadLabResults.blood_sample(blood_sample_id="NPU19661", n=n)

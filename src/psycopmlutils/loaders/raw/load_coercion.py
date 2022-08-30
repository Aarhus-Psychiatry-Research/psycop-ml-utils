from typing import Optional

import pandas as pd

from psycopmlutils.loaders.raw.sql_load import sql_load
from psycopmlutils.utils import data_loaders


class LoadCoercion:
    @data_loaders.register("coercion")
    def coercion(
        coercion_type: Optional[str] = None,
        reason_for_coercion: Optional[str] = None,
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load coercion data. Defaults to entire coercion data view.

        Args:
            coercion_type (str): Type of coercion, e.g. 'tvangsindlæggelse', 'bæltefiksering'. Defaults to None. # noqa: DAR102
            reason_for_coercion (str): Reason for coercion, e.g. 'farlighed'. Defaults to None.
            n: Number of rows to return. Defaults to None.

        Returns:
            pd.DataFrame
        """
        view = "[FOR_tvang_alt_hele_kohorten_inkl_2021]"

        sql = f"SELECT dw_ek_borger, datotid_start_sei, varighed FROM [fct].{view}"

        if coercion_type and reason_for_coercion is None:

            sql += f"WHERE typetekst_sei = '{coercion_type}'"

        if coercion_type is None and reason_for_coercion:

            sql += f"WHERE begrundtekst_sei = '{reason_for_coercion}'"

        if coercion_type and reason_for_coercion:

            sql += f"WHERE typetekst_sei = '{coercion_type}' AND begrundtekst_sei = '{reason_for_coercion}'"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None, n=n)

        df.rename(
            columns={"datotid_start_sei": "timestamp", "varighed": "value"},
            inplace=True,
        )

        return df.reset_index(drop=True)

    def _aggregate_coercion(
        coercion_types_list: list,
        subset_by: Optional[int] = "both",
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Aggregate multiple types of coercion with multiple reasons into one
        column.

        Args:
            coercion_types_list (list): List of lists each containing a string with coercion_type and a string with reason_for_coercion # noqa: DAR102
            subset_by (str): String indicating whether data is being subset based on coercion type, reason or both
            n (int, optional): Number of rows to return. Defaults to None.

        Returns:
            pd.DataFrame
        """
        subset_choices = ["type", "reason", "both"]

        if subset_by not in subset_choices:
            raise ValueError(
                "Invalid subset_by argument. Expected one of: %s" % subset_choices,
            )

        if subset_by == "both":
            dfs = [
                LoadCoercion.coercion(
                    coercion_type=id[0],
                    reason_for_coercion=id[1],
                    n=n,
                )
                for id in coercion_types_list
            ]

        if subset_by == "type":
            dfs = [
                LoadCoercion.coercion(coercion_type=id[0], n=n)
                for id in coercion_types_list
            ]

        if subset_by == "reason":
            dfs = [
                LoadCoercion.coercion(reason_for_coercion=id[0], n=n)
                for id in coercion_types_list
            ]

        return pd.concat(dfs, axis=0).reset_index(drop=True)

    @data_loaders.register("coercion_dangerous")
    def coercion_dangerous(n: Optional[int] = None) -> pd.DataFrame:

        coercion_types_lists = [
            ["Bælte", "Farlighed"],
            ["Remme", "Farlighed"],
            ["Fastholden", "Farlighed"],
            ["Handsker", "Farlighed"],
            ["Tvangstilbageholdelse", "På grund af farlighed"],
        ]

        return LoadCoercion._aggregate_coercion(
            coercion_types_list=coercion_types_lists,
            subset_by="both",
            n=n,
        )

    @data_loaders.register("bælte")
    def bælte(n: Optional[int] = None) -> pd.DataFrame:

        return LoadCoercion.coercion(
            coercion_type="Bælte",
            n=n,
        )

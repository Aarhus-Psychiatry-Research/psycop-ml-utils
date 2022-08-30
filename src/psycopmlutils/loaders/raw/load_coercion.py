from typing import Optional

import pandas as pd

from psycopmlutils.loaders.raw.sql_load import sql_load
from psycopmlutils.utils import data_loaders


class LoadCoercion:
    def coercion(
        coercion_type: Optional[str] = None,
        reason_for_coercion: Optional[str] = None,
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load coarcion data.

        Args:
            coercion_type (str): Type of coercion, e.g. 'tvangsindlæggelse', 'bæltefiksering'. # noqa: DAR102
            reason_for_coercion (str): Reason for coercion, e.g. 'farlighed'. # noqa: DAR102
            n: Number of rows to return. Defaults to None.

        Returns:
            pd.DataFrame
        """
        view = "[FOR_tvang_alt_hele_kohorten_inkl_2021]"

        if coercion_type is None and reason_for_coercion is None:

            sql = f"SELECT dw_ek_borger, datotid_start_sei, varighed FROM [fct].{view}"

        elif coercion_type is not None and reason_for_coercion is None:

            sql = f"SELECT dw_ek_borger, datotid_start_sei, varighed FROM [fct].{view} WHERE typetekst_sei = '{coercion_type}'"

        else:

            sql = f"SELECT dw_ek_borger, datotid_start_sei, varighed FROM [fct].{view} WHERE typetekst_sei = '{coercion_type}' AND begrundtekst_sei = '{reason_for_coercion}'"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None, n=n)

        df.rename(
            columns={"datotid_start_sei": "timestamp", "varighed": "value"},
            inplace=True,
        )

        # msg.good(f"Loaded {print_str}")
        return df.reset_index(drop=True)

    def _aggregate_coercion(
        coercion_types_lists: list,
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Aggregate multiple types of coercion with multiple reasons into one
        data frame column.

        Args:
            coercion_types_lists (list): List of lists each containing a string with coercion_type and a string with reason_for_coercion # noqa: DAR102
            n (int, optional): Number of rows to return. Defaults to None.

        Returns:
            pd.DataFrame
        """
        dfs = [
            LoadCoercion.coercion(coercion_type=id[0], reason_for_coercion=id[1], n=n)
            for id in coercion_types_lists
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
            coercion_types_lists=coercion_types_lists,
            n=n,
        )

from typing import Dict, List, Optional

import pandas as pd

from psycopmlutils.loaders.raw.sql_load import sql_load
from psycopmlutils.utils import data_loaders


class LoadCoercion:
    @data_loaders.register("coercion_duration")
    def coercion_duration(
        coercion_type: Optional[str] = None,
        reason_for_coercion: Optional[str] = None,
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load coercion data. By default returns entire coercion data view
        with duration in hours as the value column.

        Args:
            coercion_type (str): Type of coercion, e.g. 'tvangsindlæggelse', 'bæltefiksering'. Defaults to None. # noqa: DAR102
            reason_for_coercion (str): Reason for coercion, e.g. 'farlighed'. Defaults to None.
            n: Number of rows to return. Defaults to None which returns entire coercion data view.

        Returns:
            pd.DataFrame
        """
        view = "[FOR_tvang_alt_hele_kohorten_inkl_2021]"

        sql = f"SELECT dw_ek_borger, datotid_start_sei, varighed_timer_sei FROM [fct].{view}"

        if coercion_type and reason_for_coercion is None:

            sql += f"WHERE typetekst_sei = '{coercion_type}'"

        if coercion_type is None and reason_for_coercion:

            sql += f"WHERE begrundtekst_sei = '{reason_for_coercion}'"

        if coercion_type and reason_for_coercion:

            sql += f"WHERE typetekst_sei = '{coercion_type}' AND begrundtekst_sei = '{reason_for_coercion}'"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None, n=n)

        df.rename(
            columns={"datotid_start_sei": "timestamp", "varighed_timer_sei": "value"},
            inplace=True,
        )

        return df.reset_index(drop=True)

    def _concatenate_coercion(
        coercion_types_list: List[Dict[str, str]],
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Aggregate multiple types of coercion with multiple reasons into one
        column.

        Args:
            coercion_types_list (list): List of dictionaries containing a 'coercion_type' key and a 'reason_for_coercion' key. If keys not in dicts, they are set to None # noqa: DAR102
            n (int, optional): Number of rows to return. Defaults to None.

        Returns:
            pd.DataFrame
        """

        for d in coercion_types_list:  # Make sure proper keys are given
            if "coercion_type" not in d and "reason_for_coercion" not in d:
                raise KeyError(
                    f'{d} does not contain either "coercion_type"  or "reason_for_coercion". At least one is required.',
                )
            if "coercion_type" not in d:
                d["coercion_type"] = None
            if "reason_for_coercion" not in d:
                d["reason_for_coercion"] = None

        dfs = [
            LoadCoercion.coercion_duration(
                coercion_type=d["coercion_type"],
                reason_for_coercion=d["reason_for_coercion"],
                n=n,
            )
            for d in coercion_types_list
        ]

        return pd.concat(dfs, axis=0).reset_index(drop=True)

    @data_loaders.register("farlighed")
    def farlighed(n: Optional[int] = None) -> pd.DataFrame:

        coercion_types_list = [
            {
                "reason_for_coercion": "Farlighed",
            },
            {
                "reason_for_coercion": "På grund af farlighed",
            },
        ]

        return LoadCoercion._concatenate_coercion(
            coercion_types_list=coercion_types_list,
            n=n,
        )

    @data_loaders.register("urolig_tilstand")
    def urolig_tilstand(n: Optional[int] = None) -> pd.DataFrame:

        return LoadCoercion.coercion_duration(
            reason_for_coercion="Urolig tilstand",
            n=n,
        )

    @data_loaders.register("anden_begrundelse")
    def anden_begrundelse(n: Optional[int] = None) -> pd.DataFrame:

        return LoadCoercion.coercion_duration(
            reason_for_coercion="Anden begrundelse",
            n=n,
        )

    @data_loaders.register("magtanvendelse")
    def magtanvendelse(n: Optional[int] = None) -> pd.DataFrame:

        coercion_types_list = [
            {
                "coercion_type": "Bælte",
            },
            {
                "coercion_type": "Remme",
            },
            {
                "coercion_type": "Fastholden",
            },
            {
                "coercion_type": "Beroligende medicin",
            },
            {
                "coercion_type": "Døraflåsning",
            },
            {
                "coercion_type": "Personlig afskærmning over 24 timer",
            },
            {
                "coercion_type": "Handsker",
            },
        ]

        return LoadCoercion._concatenate_coercion(
            coercion_types_list=coercion_types_list,
            n=n,
        )

    @data_loaders.register("frihedsberøvelser")
    def frihedsberøvelser(n: Optional[int] = None) -> pd.DataFrame:

        coercion_types_list = [
            {
                "coercion_type": "Tvangsindlæggelse",
            },
            {
                "coercion_type": "Tvangstilbageholdelse",
            },
        ]

        return LoadCoercion._concatenate_coercion(
            coercion_types_list=coercion_types_list,
            n=n,
        )

    @data_loaders.register("tvangsbehandlinger")
    def tvangsbehandlinger(n: Optional[int] = None) -> pd.DataFrame:

        coercion_types_list = [
            {
                "coercion_type": "Af legemlig lidelse",
            },
            {
                "coercion_type": "Medicinering",
            },
            {
                "coercion_type": "Ernæring",
            },
            {
                "coercion_type": "ECT",
            },
        ]

        return LoadCoercion._concatenate_coercion(
            coercion_types_list=coercion_types_list,
            n=n,
        )

    @data_loaders.register("bælte")
    def bælte(n: Optional[int] = None) -> pd.DataFrame:

        return LoadCoercion.coercion_duration(
            coercion_type="Bælte",
            n=n,
        )

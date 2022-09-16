from typing import Dict, List, Optional

import pandas as pd

from psycopmlutils.loaders.raw.sql_load import sql_load
from psycopmlutils.utils import data_loaders


class LoadCoercion:
    @data_loaders.register("coercion_duration")
    def coercion_duration(
        coercion_type: Optional[str] = None,
        reason_for_coercion: Optional[str] = None,
        n_rows: Optional[int] = None,
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

        sql = f"SELECT dw_ek_borger, datotid_start_sei, varighed_timer_sei FROM [fct].{view} WHERE datotid_start_sei IS NOT NULL"

        if coercion_type and reason_for_coercion is None:

            sql += f" AND typetekst_sei = '{coercion_type}'"

        if coercion_type is None and reason_for_coercion:

            sql += f" AND begrundtekst_sei = '{reason_for_coercion}'"

        if coercion_type and reason_for_coercion:

            sql += f" AND typetekst_sei = '{coercion_type}' AND begrundtekst_sei = '{reason_for_coercion}'"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None, n_rows=n_rows)

        df.rename(
            columns={"datotid_start_sei": "timestamp", "varighed_timer_sei": "value"},
            inplace=True,
        )

        return df.reset_index(drop=True)

    def _concatenate_coercion(
        coercion_types_list: List[Dict[str, str]],
        n_rows: Optional[int] = None,
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
                n_rows=n_rows,
            )
            for d in coercion_types_list
        ]

        return pd.concat(dfs, axis=0).reset_index(drop=True)

    ### REASON (begrundtekst_sei) ###

    @data_loaders.register("farlighed")
    def farlighed(n_rows: Optional[int] = None) -> pd.DataFrame:

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
            n_rows=n_rows,
        )

    @data_loaders.register("urolig_tilstand")
    def urolig_tilstand(n_rows: Optional[int] = None) -> pd.DataFrame:

        return LoadCoercion.coercion_duration(
            reason_for_coercion="Urolig tilstand",
            n_rows=n_rows,
        )

    @data_loaders.register("anden_begrundelse")
    def anden_begrundelse(n_rows: Optional[int] = None) -> pd.DataFrame:

        return LoadCoercion.coercion_duration(
            reason_for_coercion="Anden begrundelse",
            n_rows=n_rows,
        )

    @data_loaders.register("af helbredsmæssige grunde")
    def af_helbredsmæssige_grunde(n_rows: Optional[int] = None) -> pd.DataFrame:

        return LoadCoercion.coercion_duration(
            reason_for_coercion=" Af helbredsmæssige grunde",
            n_rows=n_rows,
        )

    @data_loaders.register("nærliggende eller væsentlig fare for patienten eller andre")
    def nærliggende_fare(n_rows: Optional[int] = None) -> pd.DataFrame:

        return LoadCoercion.coercion_duration(
            reason_for_coercion="Nærliggende eller væsentlig fare for patienten eller andre",
            n_rows=n_rows,
        )

    ### GENERAL TYPE (tabeltekst) ###

    @data_loaders.register("magtanvendelse")
    def magtanvendelse(n_rows: Optional[int] = None) -> pd.DataFrame:

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
            n_rows=n_rows,
        )

    @data_loaders.register("frihedsberøvelser")
    def frihedsberøvelser(n_rows: Optional[int] = None) -> pd.DataFrame:

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
            n_rows=n_rows,
        )

    @data_loaders.register("tvangsbehandlinger")
    def tvangsbehandlinger(n_rows: Optional[int] = None) -> pd.DataFrame:

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
            n_rows=n_rows,
        )

    ### SPECIFIC TYPE (typetekst_sei) ###

    @data_loaders.register("bælte")
    def bælte(n_rows: Optional[int] = None) -> pd.DataFrame:

        return LoadCoercion.coercion_duration(
            coercion_type="Bælte",
            n_rows=n_rows,
        )

    @data_loaders.register("remme")
    def remme(n_rows: Optional[int] = None) -> pd.DataFrame:

        return LoadCoercion.coercion_duration(
            coercion_type="Remme",
            n_rows=n_rows,
        )

    @data_loaders.register("fastholden")
    def fastholden(n_rows: Optional[int] = None) -> pd.DataFrame:

        return LoadCoercion.coercion_duration(
            coercion_type="Fastholden",
            n_rows=n_rows,
        )

    @data_loaders.register("beroligende medicin")
    def beroligende_medicin(n_rows: Optional[int] = None) -> pd.DataFrame:

        return LoadCoercion.coercion_duration(
            coercion_type="Beroligende medicin",
            n_rows=n_rows,
        )

    @data_loaders.register("handsker")
    def handsker(n_rows: Optional[int] = None) -> pd.DataFrame:

        return LoadCoercion.coercion_duration(
            coercion_type="Handsker",
            n_rows=n_rows,
        )

    @data_loaders.register("tvangsindlæggelse")
    def tvangsindlæggelse(n_rows: Optional[int] = None) -> pd.DataFrame:

        return LoadCoercion.coercion_duration(
            coercion_type="Tvangsindlæggelse",
            n_rows=n_rows,
        )

    @data_loaders.register("tvangstilbageholdelse ")
    def tvangstilbageholdelse(n_rows: Optional[int] = None) -> pd.DataFrame:

        return LoadCoercion.coercion_duration(
            coercion_type="Tvangstilbageholdelse",
            n_rows=n_rows,
        )

    @data_loaders.register("medicinering")
    def medicinering(n_rows: Optional[int] = None) -> pd.DataFrame:

        return LoadCoercion.coercion_duration(
            coercion_type="Medicinering",
            n_rows=n_rows,
        )

    @data_loaders.register("ect")
    def ect(n_rows: Optional[int] = None) -> pd.DataFrame:

        return LoadCoercion.coercion_duration(
            coercion_type="ECT",
            n_rows=n_rows,
        )

    @data_loaders.register("ernæring")
    def ernæring(n_rows: Optional[int] = None) -> pd.DataFrame:

        return LoadCoercion.coercion_duration(
            coercion_type="Ernæring",
            n_rows=n_rows,
        )

"""Loaders for medications."""
from typing import Optional

import pandas as pd
from wasabi import msg

from psycopmlutils.loaders.raw.sql_load import sql_load
from psycopmlutils.utils import data_loaders

# pylint: disable=missing-function-docstring


def _load_one_source(
    atc_code: str,
    source_timestamp_col_name: str,
    view: str,
    output_col_name: Optional[str] = None,
    wildcard_icd_code: Optional[bool] = False,
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load the prescribed medications that match atc. If wildcard_icd_code,
    match from atc_code*. Aggregates all that match. Beware that data is
    incomplete prior to sep. 2016 for prescribed medications.

    Args:
        atc_code (str): ATC string to match on. # noqa: DAR102
        source_timestamp_col_name (str): Name of the timestamp column in the SQL
            table.
        view (str): Which view to use, e.g.
            "FOR_Medicin_ordineret_inkl_2021_feb2022"
        output_col_name (str, optional): Name of new column string. Defaults to
            None.
        wildcard_icd_code (bool, optional): Whether to match on atc_code* or
            atc_code.
        n_rows (int, optional): Number of rows to return. Defaults to None.

    Returns:
        pd.DataFrame: A pandas dataframe with dw_ek_borger, timestamp and
            output_col_name = 1
    """

    if wildcard_icd_code:
        end_of_sql = "%"
    else:
        end_of_sql = ""  # noqa

    view = f"[{view}]"
    sql = (
        f"SELECT dw_ek_borger, {source_timestamp_col_name}, atc FROM [fct].{view}"
        + f" WHERE {source_timestamp_col_name} IS NOT NULL AND (lower(atc)) LIKE lower('{atc_code}{end_of_sql}')"
    )

    df = sql_load(sql, database="USR_PS_FORSK", chunksize=None, n_rows=n_rows)

    if output_col_name is None:
        output_col_name = atc_code

    df[output_col_name] = 1

    df.drop(["atc"], axis="columns", inplace=True)

    return df.rename(
        columns={
            source_timestamp_col_name: "timestamp",
        },
    )


def load(
    atc_code: str,
    output_col_name: Optional[str] = None,
    load_prescribed: Optional[bool] = False,
    load_administered: Optional[bool] = True,
    wildcard_icd_code: Optional[bool] = True,
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load medications. Aggregates prescribed/administered if both true. If
    wildcard_icd_code, match from atc_code*. Aggregates all that match. Beware
    that data is incomplete prior to sep. 2016 for prescribed medications.

    Args:
        atc_code (str): ATC-code prefix to load. Matches atc_code_prefix*. # noqa: DAR102
            Aggregates all.
        output_col_name (str, optional): Name of output_col_name. Contains 1 if
            atc_code matches atc_code_prefix, 0 if not.Defaults to
            {atc_code_prefix}_value.
        load_prescribed (bool, optional): Whether to load prescriptions. Defaults to
            False. Beware incomplete until sep 2016.
        load_administered (bool, optional): Whether to load administrations.
            Defaults to True.
        wildcard_icd_code (bool, optional): Whether to match on atc_code* or
            atc_code.
        n_rows (int, optional): Number of rows to return. Defaults to None.

    Returns:
        pd.DataFrame: Cols: dw_ek_borger, timestamp, {atc_code_prefix}_value = 1
    """

    if load_prescribed:
        msg.warn(
            "Beware, there are missing prescriptions until september 2016. "
            "Hereafter, data is complete. See the wiki (OBS: Medication) for more details.",
        )

    df = pd.DataFrame()

    if load_prescribed:
        df_medication_prescribed = _load_one_source(
            atc_code=atc_code,
            source_timestamp_col_name="datotid_ordinationstart",
            view="FOR_Medicin_ordineret_inkl_2021_feb2022",
            output_col_name=output_col_name,
            wildcard_icd_code=wildcard_icd_code,
            n_rows=n_rows,
        )

        df = pd.concat([df, df_medication_prescribed])

    if load_administered:
        df_medication_administered = _load_one_source(
            atc_code=atc_code,
            source_timestamp_col_name="datotid_administration_start",
            view="FOR_Medicin_administreret_inkl_2021_feb2022",
            output_col_name=output_col_name,
            wildcard_icd_code=wildcard_icd_code,
            n_rows=n_rows,
        )
        df = pd.concat([df, df_medication_administered])

    if output_col_name is None:
        output_col_name = atc_code

    df.rename(
        columns={
            atc_code: "value",
        },
        inplace=True,
    )

    return df.reset_index(drop=True).drop_duplicates(
        subset=["dw_ek_borger", "timestamp", "value"],
        keep="first",
    )


def concat_medications(
    output_col_name: str,
    atc_code_prefixes: list[str],
    n_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Aggregate multiple blood_sample_ids (typically NPU-codes) into one
    column.

    Args:
        output_col_name (str): Name for new column.  # noqa: DAR102
        atc_code_prefixes (list[str]): list of atc_codes.
        n_rows (int, optional): Number of atc_codes to aggregate. Defaults to None.

    Returns:
        pd.DataFrame
    """
    dfs = [
        load(
            atc_code=f"{id}",
            output_col_name=output_col_name,
            n_rows=n_rows,
        )
        for id in atc_code_prefixes
    ]

    return (
        pd.concat(dfs, axis=0)
        .drop_duplicates(
            subset=["dw_ek_borger", "timestamp", "value"],
            keep="first",
        )
        .reset_index(drop=True)
    )


# data_loaders primarly used in psychiatry
@data_loaders.register("antipsychotics")
def antipsychotics(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N05A",
        load_prescribed=True,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


# 1. generation antipsychotics [flupentixol, pimozid, haloperidol, zuclopenthixol, melperon,pipamperon, chlorprotixen]
@data_loaders.register("first_gen_antipsychotics")
def first_gen_antipsychotics(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code=[
            "N05AF01",
            "N05AG02",
            "N05AD01",
            "N05AF05",
            "N05AD03",
            "N05AD05",
            "N05AF03",
        ],
        load_prescribed=True,
        load_administered=True,
        wildcard_icd_code=False,
        n_rows=n_rows,
    )


# 2. generation antipsychotics [amisulpride, aripiprazole,asenapine, brexpiprazole, cariprazine, lurasidone, olanzapine, paliperidone, Quetiapine, risperidone, sertindol]
@data_loaders.register("second_gen_antipsychotics")
def second_gen_antipsychotics(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code=[
            "N05AL05",
            "N05AX12",
            "N05AH05",
            "N05AX16",
            "N05AX15",
            "N05AE02",
            "N05AE05",
            "N05AH03",
            "N05AX13",
            "N05AH04",
            "N05AX08",
            "N05AE04",
            "N05AE03",
        ],
        load_prescribed=True,
        load_administered=True,
        wildcard_icd_code=False,
        n_rows=n_rows,
    )


@data_loaders.register("olanzapine")
def olanzapine(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N05AH03",
        load_prescribed=True,
        load_administered=True,
        wildcard_icd_code=False,
        n_rows=n_rows,
    )


@data_loaders.register("clozapine")
def clozapine(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N05AH02",
        load_prescribed=True,
        load_administered=True,
        wildcard_icd_code=False,
        n_rows=n_rows,
    )


@data_loaders.register("anxiolytics")
def anxiolytics(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N05B",
        load_prescribed=False,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("hypnotics and sedatives")
def hypnotics(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N05C",
        load_prescribed=False,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("antidepressives")
def antidepressives(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N06A",
        load_prescribed=False,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


# SSRIs [escitalopram, citalopram, fluvoxamin, fluoxetin, paroxetin]
@data_loaders.register("ssri")
def ssri(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code=["N06AB10", "N06AB04", "N06AB08", "N06AB03", "N06AB05", "N06AB06"],
        load_prescribed=True,
        load_administered=True,
        wildcard_icd_code=False,
        n_rows=n_rows,
    )


# SNRIs [duloxetin, venlafaxin]
@data_loaders.register("snri")
def snri(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code=["N06AX21", "N06AX16"],
        load_prescribed=True,
        load_administered=True,
        wildcard_icd_code=False,
        n_rows=n_rows,
    )


# TCAs [amitryptilin, clomipramin, imipramin, nortriptyline, dosulepin]
@data_loaders.register("tca")
def tca(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code=["N06AA09", "N06AA04", "N06AA02", "N06AA10", "N06AA16"],
        load_prescribed=True,
        load_administered=True,
        wildcard_icd_code=False,
        n_rows=n_rows,
    )


@data_loaders.register("lithium")
def lithium(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N05AN01",
        load_prescribed=True,
        load_administered=True,
        wildcard_icd_code=False,
        n_rows=n_rows,
    )


@data_loaders.register("hyperactive disorders medications")
def hyperactive_disorders_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N06B",
        load_prescribed=False,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("dementia medications")
def dementia_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N06D",
        load_prescribed=False,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("anti-epileptics")
def anti_epileptics(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N03",
        load_prescribed=False,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


# medications used in alcohol abstinence treatment [thiamin, b-combin, klopoxid, fenemal]
@data_loaders.register("alcohol_abstinence")
def alcohol_abstinence(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code=["A11DA01", "A11EA", "N05BA02", "N03AA02"],
        load_prescribed=True,
        load_administered=True,
        wildcard_icd_code=False,
        n_rows=n_rows,
    )


# data loaders for medications primarily used outside psychiatry
@data_loaders.register("alimentary_tract_and_metabolism_medications")
def alimentary_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="A",
        load_prescribed=False,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("blood_and_blood_forming_organs_medications")
def blood_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="B",
        load_prescribed=False,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("cardiovascular_medications")
def cardiovascular_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="C",
        load_prescribed=False,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("dermatologicals")
def dermatological_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="D",
        load_prescribed=False,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("genito_urinary_system_and_sex_hormones_medications")
def genito_sex_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="G",
        load_prescribed=False,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("systemic_hormonal_preparations")
def hormonal_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="H",
        load_prescribed=False,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("antiinfectives")
def antiinfectives(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="J",
        load_prescribed=False,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("antineoplastic")
def antineoplastic(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="L",
        load_prescribed=False,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("musculoskeletal_medications")
def musculoskeletal_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="M",
        load_prescribed=False,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("nervous_system_medications")
def nervous_system_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N",
        load_prescribed=False,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("analgesics")
def analgesic(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="N02",
        load_prescribed=False,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("antiparasitic")
def antiparasitic(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="P",
        load_prescribed=False,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("respiratory_medications")
def respiratory_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="R",
        load_prescribed=False,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("sensory_organs_medications")
def sensory_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="S",
        load_prescribed=False,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


@data_loaders.register("various_medications")
def various_medications(n_rows: Optional[int] = None) -> pd.DataFrame:
    return load(
        atc_code="V",
        load_prescribed=False,
        load_administered=True,
        wildcard_icd_code=True,
        n_rows=n_rows,
    )


__all__ = [
    "alimentary_medications",
    "anti_epileptics",
    "antidepressives",
    "antiinfectives",
    "antineoplastic",
    "antiparasitic",
    "antipsychotics",
    "anxiolytics",
    "blood_medications",
    "cardiovascular_medications",
    "concat_medications",
    "dementia_medications",
    "dermatological_medications",
    "genito_sex_medications",
    "hormonal_medications",
    "hyperactive_disorders_medications",
    "hypnotics",
    "load",
    "musculoskeletal_medications",
    "nervous_system_medications",
    "respiratory_medications",
    "sensory_medications",
    "various_medications",
]

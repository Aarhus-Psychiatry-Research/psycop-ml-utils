from venv import create
from numpy import source
import pandas as pd
from loaders.sql_load import sql_load
from loaders.str_utils import create_cols_for_unique_vals_at_depth
from wasabi import msg


class LoadMedications:
    def medication(
        atc_str: str,
        new_col_str=None,
        load_prescribed=True,
        load_administered=True,
        depth: int = None,
    ) -> pd.DataFrame:
        """Load medications. Aggregates prescribed/administered if both true.

        Args:
            atc_str (str): ATC-code prefix to load. Matches atc_str*. Aggregates all.
            new_col_str (_type_, optional): Name of new_col_str. Defaults to atc_str_val.
            load_prescribed (bool, optional): Whether to load prescriptions. Defaults to True. Beware incomplete until sep 2016.
            load_administered (bool, optional): Whether to load administrations. Defaults to True.
            depth (int, optional): At which level to generate combinations. E.g. if depth = 3, A0004 and A0001 will both be A000,
                whereas depth = 4 would result in two different columns.

        Returns:
            pd.DataFrame: Cols: dw_ek_borger, timestamp, atc_str_val = 1
        """
        print_str = f"medications matching NPU-code {atc_str}"
        msg.info(f"Loading {print_str}")

        df = pd.DataFrame()

        if load_prescribed:
            df_medication_prescribed = LoadMedications._load_medications(
                atc_str=atc_str,
                source_timestamp_col_name="datotid_ordinationstart",
                view="FOR_Medicin_ordineret_inkl_2021_feb2022",
                new_col_str=new_col_str,
                depth=depth,
            )
            df = pd.concat([df, df_medication_prescribed])

        if load_administered:
            df_medication_administered = LoadMedications._load_medications(
                atc_str=atc_str,
                source_timestamp_col_name="datotid_administration_start",
                view="FOR_Medicin_administreret_inkl_2021_feb2022",
                new_col_str=new_col_str,
                depth=depth,
            )
            df = pd.concat([df, df_medication_administered])

        if new_col_str == None:
            new_col_str = atc_str

        df.rename(
            columns={
                atc_str: f"{new_col_str}_val",
            },
            inplace=True,
        )

        msg.good(f"Loaded {print_str}")
        return df.reset_index(drop=True)

    def _load_medications(
        atc_str: str,
        source_timestamp_col_name: str,
        view: str,
        new_col_str: str = None,
        depth: int = None,
    ) -> pd.DataFrame:
        """Load the prescribed medications that match atc from the beginning of their atc string.
        Aggregates all that match. Beware that data is incomplete prior to sep. 2016.

        Args:
            atc_str (str): ATC string to match on.
            source_timestamp_col_name (str): Name of the timestamp column in the SQL table.
            view (str): Which view to use, e.g. "FOR_Medicin_ordineret_inkl_2021_feb2022"
            new_col_str (str, optional): Name of new column string. Defaults to None.
            depth (int, optional): At which level to generate combinations. E.g. if depth = 3, A0004 and A0001 will both be A000,
                whereas depth = 4 would result in two different columns.

        Returns:
            pd.DataFrame: A pandas dataframe with dw_ek_borger, timestamp and new_col_str = 1
        """
        view = f"[{view}]"
        sql = f"SELECT dw_ek_borger, {source_timestamp_col_name}, atc FROM [fct].{view} WHERE (lower(atc)) LIKE lower('{atc_str}%')"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None)

        if new_col_str is None:
            new_col_str = atc_str

        # Handle depth
        if depth is None:
            df[new_col_str] = 1
        else:
            df = create_cols_for_unique_vals_at_depth(
                df=df, source_col_name="atc", depth=depth
            )

        df.drop(["atc"], axis="columns", inplace=True)

        return df.rename(
            columns={
                source_timestamp_col_name: "timestamp",
            }
        )

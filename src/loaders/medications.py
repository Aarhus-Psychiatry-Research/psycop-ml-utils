import pandas as pd
from loaders import sql_load
from wasabi import msg


class LoadMedications:
    def medication(
        atc_str: str, new_col_str=None, load_prescribed=True, load_administered=True
    ) -> pd.DataFrame:
        """Load medications. Aggregates prescribed/administered if both true.

        Args:
            atc_str (str): ATC-code prefix to load. Matches atc_str*. Aggregates all.
            new_col_str (_type_, optional): Name of new_col_str. Defaults to atc_str_val.
            load_prescribed (bool, optional): Whether to load prescriptions. Defaults to True. Beware incomplete until sep 2016.
            load_administered (bool, optional): Whether to load administrations. Defaults to True.

        Returns:
            pd.DataFrame: Cols: dw_ek_borger, timestamp, atc_str_val = 1
        """
        print_str = f"medications matching NPU-code {atc_str}"
        msg.info(f"Loading {print_str}")

        df = pd.DataFrame()

        if load_prescribed:
            df = pd.concat(
                [df, LoadMedications._load_medication_prescriptions(atc_str=atc_str)]
            )

        if load_administered:
            df = pd.concat(
                [df, LoadMedications._load_medication_administrations(atc_str=atc_str)]
            )

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

    def _load_medication_administrations(
        atc_str: str, new_col_str: str = None
    ) -> pd.DataFrame:
        """Load the administered medications that match atc from the beginning of their atc string.
        Aggregates all that match.

        Args:
            atc_str (str): ATC string to match on.
            new_col_str (str, optional): Name of new column string. Defaults to None.

        Returns:
            pd.DataFrame: A pandas dataframe with dw_ek_borger, timestamp and new_col_str = 1
        """
        view = "[FOR_Medicin_administreret_inkl_2021_feb2022]"
        sql = f"SELECT dw_ek_borger, datotid_administration_start, atc FROM [fct].{view} WHERE (lower(atc)) LIKE lower('{atc_str}%')"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None)

        df.drop("atc", axis="columns", inplace=True)

        if new_col_str is None:
            new_col_str = atc_str

        df[new_col_str] = 1

        return df.rename(
            columns={
                "datotid_administration_start": "timestamp",
            }
        )

    def _load_medication_prescriptions(
        atc_str: str, new_col_str: str = None
    ) -> pd.DataFrame:
        """Load the prescribed medications that match atc from the beginning of their atc string.
        Aggregates all that match. Beware that data is incomplete prior to sep. 2016.

        Args:
            atc_str (str): ATC string to match on.
            new_col_str (str, optional): Name of new column string. Defaults to None.

        Returns:
            pd.DataFrame: A pandas dataframe with dw_ek_borger, timestamp and new_col_str = 1
        """
        view = "[FOR_Medicin_ordineret_inkl_2021_feb2022]"
        sql = f"SELECT dw_ek_borger, datotid_ordinationstart, atc FROM [fct].{view} WHERE (lower(atc)) LIKE lower('{atc_str}%')"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None)

        df.drop("atc", axis="columns", inplace=True)

        if new_col_str is None:
            new_col_str = atc_str

        df[new_col_str] = 1

        return df.rename(
            columns={
                "datotid_ordinationstart": "timestamp",
            }
        )

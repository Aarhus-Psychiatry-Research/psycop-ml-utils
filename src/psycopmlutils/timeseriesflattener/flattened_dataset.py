import datetime as dt
from datetime import timedelta
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from catalogue import Registry  # noqa
from pandas import DataFrame
from wasabi import Printer, msg

from psycopmlutils.timeseriesflattener.resolve_multiple_functions import resolve_fns
from psycopmlutils.utils import (
    data_loaders,
    df_contains_duplicates,
    generate_feature_colname,
)


class FlattenedDataset:
    """Turn a set of time-series into tabular prediction-time data."""

    def __init__(
        self,
        prediction_times_df: DataFrame,
        id_col_name: Optional[str] = "dw_ek_borger",
        timestamp_col_name: Optional[str] = "timestamp",
        min_date: Optional[pd.Timestamp] = None,
        n_workers: Optional[int] = 60,
        predictor_col_name_prefix: Optional[str] = "pred",
        outcome_col_name_prefix: Optional[str] = "outc",
    ):
        """Class containing a time-series, flattened. A 'flattened' version is
        a tabular representation for each prediction time.

        A prediction time is every timestamp where you want your model to issue a prediction.

        E.g if you have a prediction_times_df:

        id_col_name | timestamp_col_name
        1           | 2022-01-10
        1           | 2022-01-12
        1           | 2022-01-15

        And a time-series of blood-pressure values as an outcome:
        id_col_name | timestamp_col_name | blood_pressure_value
        1           | 2022-01-09         | 120
        1           | 2022-01-14         | 140

        Then you can "flatten" the outcome into a new df, with a row for each of your prediction times:
        id_col_name | timestamp_col_name | latest_blood_pressure_within_24h
        1           | 2022-01-10         | 120
        1           | 2022-01-12         | NA
        1           | 2022-01-15         | 140

        Args:
            prediction_times_df (DataFrame): Dataframe with prediction times, required cols: patient_id, .
            timestamp_col_name (str, optional): Column name name for timestamps. Is used across outcomes and predictors. Defaults to "timestamp".
            min_date (Optional[pd.Timestamp], optional): Drop all prediction times before this date. Defaults to None.
            id_col_name (str, optional): Column namn name for patients ids. Is used across outcome and predictors. Defaults to "dw_ek_borger".
            predictor_col_name_prefix (str, optional): Prefix for predictor col names. Defaults to "pred_".
            outcome_col_name_prefix (str, optional): Prefix for outcome col names. Defaults to "outc_".
            n_workers (int): Number of subprocesses to spawn for parallellisation. Defaults to 60.
        """
        self.n_workers = n_workers

        self.timestamp_col_name = timestamp_col_name
        self.id_col_name = id_col_name
        self.pred_time_uuid_col_name = "prediction_time_uuid"
        self.predictor_col_name_prefix = predictor_col_name_prefix
        self.outcome_col_name_prefix = outcome_col_name_prefix
        self.min_date = min_date

        self.df = prediction_times_df

        # Check that colnames are present
        for col_name in [self.timestamp_col_name, self.id_col_name]:
            if col_name not in self.df.columns:
                raise ValueError(
                    f"{col_name} does not exist in prediction_times_df, change the df or set another argument",
                )

        # Check for duplicates
        if df_contains_duplicates(
            df=self.df,
            col_subset=[self.id_col_name, self.timestamp_col_name],
        ):
            raise ValueError(
                "Duplicate patient/timestamp combinations in prediction_times_df, aborting",
            )

        # Check timestamp col type
        timestamp_col_type = type(self.df[self.timestamp_col_name][0]).__name__

        if timestamp_col_type not in ["Timestamp"]:
            try:
                self.df[self.timestamp_col_name] = pd.to_datetime(
                    self.df[self.timestamp_col_name],
                )
            except Exception:
                raise ValueError(
                    f"prediction_times_df: {self.timestamp_col_name} is of type {timestamp_col_type}, and could not be converted to 'Timestamp' from Pandas. Will cause problems. Convert before initialising FlattenedDataset.",
                )

        # Drop prediction times before min_date
        if min_date is not None:
            self.df = self.df[self.df[self.timestamp_col_name] > self.min_date]

        # Create pred_time_uuid_columne
        self.df[self.pred_time_uuid_col_name] = self.df[self.id_col_name].astype(
            str,
        ) + self.df[self.timestamp_col_name].dt.strftime("-%Y-%m-%d-%H-%M-%S")

        self.loaders_catalogue = data_loaders

    def add_temporal_predictors_from_list_of_argument_dictionaries(
        self,
        predictors: List[Dict[str, str]],
        predictor_dfs: Dict[str, DataFrame] = None,
        resolve_multiple_fns: Optional[Dict[str, Callable]] = None,
    ):
        """Add predictors to the flattened dataframe from a list.

        Args:
            predictors (List[Dict[str, str]]): A list of dictionaries describing the prediction_features you'd like to generate.
            predictor_dfs (Dict[str, DataFrame], optional): If wanting to pass already resolved dataframes.
                By default, you should add your dataframes to the @data_loaders registry.
                Then the the predictor_df value in the predictor dict will map to a callable which returns the dataframe.
                Optionally, you can map the string to a dataframe in predictor_dfs.
            resolve_multiple_fns (Union[str, Callable], optional): If wanting to use manually defined resolve_multiple strategies
                I.e. ones that aren't in the resolve_fns catalogue require a dictionary mapping the
                resolve_multiple string to a Callable object. Defaults to None.

        Example:
            >>> predictor_list = [
            >>>     {
            >>>         "predictor_df": "df_name",
            >>>         "lookbehind_days": 1,
            >>>         "resolve_multiple": "resolve_multiple_strat_name",
            >>>         "fallback": 0,
            >>>         "source_values_col_name": "val",
            >>>     },
            >>>     {
            >>>         "predictor_df": "df_name",
            >>>         "lookbehind_days": 1,
            >>>         "resolve_multiple_fns": "min",
            >>>         "fallback": 0,
            >>>         "source_values_col_name": "val",
            >>>     }
            >>> ]
            >>> predictor_dfs = {"df_name": df_object}
            >>> resolve_multiple_strategies = {"resolve_multiple_strat_name": resolve_multiple_func}

            >>> dataset.add_predictors_from_list(
            >>>     predictor_list=predictor_list,
            >>>     predictor_dfs=predictor_dfs,
            >>>     resolve_multiple_fn_dict=resolve_multiple_strategies,
            >>> )
        """
        processed_arg_dicts = []

        dicts_found_in_predictor_dfs = []

        # Replace strings with objects as relevant
        for arg_dict in predictors:

            # If resolve_multiple is a string, see if possible to resolve to a Callable
            # Actual resolving is handled in resolve_multiple_values_within_interval_days
            # To preserve str for column name generation
            if isinstance(arg_dict["resolve_multiple"], str):
                # Try from resolve_multiple_fns
                resolved_func = False
                if resolve_multiple_fns is not None:
                    try:
                        resolved_func = resolve_multiple_fns.get(
                            [arg_dict["resolve_multiple"]],
                        )
                    except Exception:
                        pass

                try:
                    resolved_func = resolve_fns.get(arg_dict["resolve_multiple"])
                except Exception:
                    pass

                if not isinstance(resolved_func, Callable):
                    raise ValueError(
                        "resolve_function neither is nor resolved to a Callable",
                    )

            # Rename arguments for create_flattened_df_for_val
            arg_dict["values_df"] = arg_dict["predictor_df"]
            arg_dict["interval_days"] = arg_dict["lookbehind_days"]
            arg_dict["direction"] = "behind"
            arg_dict["id_col_name"] = self.id_col_name
            arg_dict["timestamp_col_name"] = self.timestamp_col_name
            arg_dict["pred_time_uuid_col_name"] = self.pred_time_uuid_col_name
            arg_dict["new_col_name_prefix"] = self.predictor_col_name_prefix

            if "new_col_name" not in arg_dict.keys():
                arg_dict["new_col_name"] = arg_dict["values_df"]

            # Resolve values_df to either a dataframe from predictor_dfs_dict or a callable from the registr
            loader_fns = self.loaders_catalogue.get_all()
            try:
                if predictor_dfs is not None:
                    if arg_dict["values_df"] in predictor_dfs:
                        if arg_dict["values_df"] not in dicts_found_in_predictor_dfs:
                            dicts_found_in_predictor_dfs.append(arg_dict["values_df"])
                            msg.info(f"Found {arg_dict['values_df']} in predictor_dfs")

                        arg_dict["values_df"] = predictor_dfs[
                            arg_dict["values_df"]
                        ].copy()
                    else:
                        arg_dict["values_df"] = loader_fns[arg_dict["values_df"]]
                elif predictor_dfs is None:
                    arg_dict["values_df"] = loader_fns[arg_dict["values_df"]]
            except Exception:
                # Error handling in _validate_processed_arg_dicts
                # to handle in bulk
                pass

            required_keys = [
                "values_df",
                "direction",
                "interval_days",
                "resolve_multiple",
                "fallback",
                "new_col_name",
                "new_col_name_prefix",
            ]

            if "values_to_load" in arg_dict:
                required_keys.append("values_to_load")

            processed_arg_dicts.append(
                select_and_assert_keys(dictionary=arg_dict, key_list=required_keys),
            )

        # Validate dicts before starting pool, saves time if errors!
        self._validate_processed_arg_dicts(processed_arg_dicts)

        pool = Pool(self.n_workers)

        flattened_predictor_dfs = pool.map(
            self._flatten_temporal_values_to_df_wrapper,
            processed_arg_dicts,
        )

        flattened_predictor_dfs = [
            df.set_index(self.pred_time_uuid_col_name) for df in flattened_predictor_dfs
        ]

        msg.info("Feature generation complete, concatenating")
        concatenated_dfs = pd.concat(
            flattened_predictor_dfs,
            axis=1,
        ).reset_index()

        self.df = pd.merge(
            self.df,
            concatenated_dfs,
            how="left",
            on=self.pred_time_uuid_col_name,
            suffixes=("", ""),
            validate="1:1",
        )

        self.df = self.df.copy()

    def _validate_processed_arg_dicts(self, arg_dicts: list):
        warnings = []

        for d in arg_dicts:
            if not isinstance(d["values_df"], (DataFrame, Callable)):
                warnings.append(
                    f"values_df resolves to neither a Callable nor a DataFrame in {d}",
                )

            if not (d["direction"] == "ahead" or d["direction"] == "behind"):
                warnings.append(f"direction is neither ahead or behind in {d}")

            if not isinstance(d["interval_days"], (int, float)):
                warnings.append(f"interval_days is neither an int nor a float in {d}")

        if len(warnings) != 0:
            raise ValueError(
                f"Didn't generate any features because: {warnings}",
            )

    def _flatten_temporal_values_to_df_wrapper(self, kwargs_dict: Dict) -> DataFrame:
        """Wrap flatten_temporal_values_to_df with kwargs for multithreading
        pool.

        Args:
            kwargs_dict (Dict): Dictionary of kwargs

        Returns:
            DataFrame: DataFrame generates with create_flattened_df
        """
        return self.flatten_temporal_values_to_df(
            prediction_times_with_uuid_df=self.df[
                [
                    self.pred_time_uuid_col_name,
                    self.id_col_name,
                    self.timestamp_col_name,
                ]
            ],
            id_col_name=self.id_col_name,
            timestamp_col_name=self.timestamp_col_name,
            pred_time_uuid_col_name=self.pred_time_uuid_col_name,
            **kwargs_dict,
        )

    def add_age(
        self,
        id2date_of_birth: DataFrame,
        date_of_birth_col_name: Optional[str] = "date_of_birth",
    ):
        """Add age at prediction time to each prediction time.

        Args:
            id2date_of_birth (DataFrame): Two columns, id and date_of_birth.
            date_of_birth_col_name (str, optional): Name of the date_of_birth column in id2date_of_birth.
            Defaults to "date_of_birth".

        Raises:
            ValueError: _description_
        """
        if id2date_of_birth[date_of_birth_col_name].dtype != "<M8[ns]":
            try:
                id2date_of_birth[date_of_birth_col_name] = pd.to_datetime(
                    id2date_of_birth[date_of_birth_col_name],
                    format="%Y-%m-%d",
                )
            except Exception:
                raise ValueError(
                    f"Conversion of {date_of_birth_col_name} to datetime failed, doesn't match format %Y-%m-%d. Recommend converting to datetime before adding.",
                )

        self.add_static_info(
            info_df=id2date_of_birth,
            input_col_name=date_of_birth_col_name,
        )

        age = (
            (
                self.df[self.timestamp_col_name]
                - self.df[f"{self.predictor_col_name_prefix}_{date_of_birth_col_name}"]
            ).dt.days
            / (365.25)
        ).round(2)

        self.df.drop(
            f"{self.predictor_col_name_prefix}_{date_of_birth_col_name}",
            axis=1,
            inplace=True,
        )

        self.df[f"{self.predictor_col_name_prefix}_age_in_years"] = age

    def add_static_info(
        self,
        info_df: DataFrame,
        prefix: Optional[str] = "self.predictor_col_name_prefix",
        input_col_name: Optional[str] = None,
        output_col_name: Optional[str] = None,
    ):
        """Add static info to each prediction time, e.g. age, sex etc.

        Args:
            info_df (DataFrame): Contains an id_column and a value column.
            prefix (str, optional): Prefix for column. Defaults to self.predictor_col_name_prefix.
            input_col_name (str, optional): Column names for the values you want to add. Defaults to "value".
            output_col_name (str, optional): Name of the output column. Defaults to None.
        """

        value_col_name = [col for col in info_df.columns if col not in self.id_col_name]

        # Try to infer value col name if not provided
        if input_col_name is None:
            if len(value_col_name) == 1:
                value_col_name = value_col_name[0]
            elif len(value_col_name) > 1:
                raise ValueError(
                    f"Only one value column can be added to static info, found multiple: {value_col_name}",
                )
            elif len(value_col_name) == 0:
                raise ValueError("No value column found in info_df, please check.")
        else:
            value_col_name = input_col_name

        # Find output_col_name
        if prefix == "self.predictor_col_name_prefix":
            prefix = self.predictor_col_name_prefix

        if output_col_name is None:
            output_col_name = f"{prefix}_{value_col_name}"
        else:
            output_col_name = f"{prefix}_{output_col_name}"

        df = pd.DataFrame(
            {
                self.id_col_name: info_df[self.id_col_name],
                output_col_name: info_df[value_col_name],
            },
        )

        self.df = pd.merge(
            self.df,
            df,
            how="left",
            on=self.id_col_name,
            suffixes=("", ""),
            validate="m:1",
        )

    def add_temporal_outcome(
        self,
        outcome_df: DataFrame,
        lookahead_days: float,
        resolve_multiple: Union[Callable, str],
        fallback: float,
        incident: Optional[bool] = False,
        new_col_name: Optional[str] = "value",
        dichotomous: Optional[bool] = False,
    ):
        """Add an outcome-column to the dataset.

        Args:
            outcome_df (DataFrame): A table in wide format. Required columns: patient_id, timestamp, value.
            lookahead_days (float): How far ahead to look for an outcome in days. If none found, use fallback.
            resolve_multiple (Callable, str): How to handle multiple values within the lookahead window. Takes either i) a function that takes a list as an argument and returns a float, or ii) a str mapping to a callable from the resolve_multiple_fn catalogue.
            fallback (float): What to do if no value within the lookahead.
            incident (Optional[bool], optional): Whether looking for an incident outcome. If true, removes all prediction times after the outcome time. Defaults to false.
            new_col_name (str): Name to use for new col. Automatically generated as '{new_col_name}_within_{lookahead_days}_days'. Defaults to "value".
            dichotomous (bool, optional): Whether the outcome is dichotomous. Allows computational shortcuts, making adding an outcome _much_ faster. Defaults to False.
        """
        prediction_timestamp_col_name = f"{self.timestamp_col_name}_prediction"
        outcome_timestamp_col_name = f"{self.timestamp_col_name}_outcome"
        if incident:
            df = pd.merge(
                self.df,
                outcome_df,
                how="left",
                on=self.id_col_name,
                suffixes=("_prediction", "_outcome"),
                validate="m:1",
            )

            df = df.drop(
                df[
                    df[outcome_timestamp_col_name] < df[prediction_timestamp_col_name]
                ].index,
            )

            if dichotomous:
                full_col_str = f"{self.outcome_col_name_prefix}_dichotomous_{new_col_name}_within_{lookahead_days}_days_{resolve_multiple}_fallback_{fallback}"

                df[full_col_str] = (
                    df[prediction_timestamp_col_name] + timedelta(days=lookahead_days)
                    > df[outcome_timestamp_col_name]
                ).astype(int)

            df.rename(
                {prediction_timestamp_col_name: "timestamp"},
                axis=1,
                inplace=True,
            )
            df.drop([outcome_timestamp_col_name], axis=1, inplace=True)

            df.drop(["value"], axis=1, inplace=True)

            self.df = df

        if not (dichotomous and incident):
            self.add_temporal_col_to_flattened_dataset(
                values_df=outcome_df,
                direction="ahead",
                interval_days=lookahead_days,
                resolve_multiple=resolve_multiple,
                fallback=fallback,
                new_col_name=new_col_name,
            )

    def add_temporal_predictor(
        self,
        predictor_df: DataFrame,
        lookbehind_days: float,
        resolve_multiple: Union[Callable, str],
        fallback: float,
        new_col_name: str = None,
    ):
        """Add a column with predictor values to the flattened dataset (e.g.
        "average value of bloodsample within n days").

        Args:
            predictor_df (DataFrame): A table in wide format. Required columns: patient_id, timestamp, value.
            lookbehind_days (float): How far behind to look for a predictor value in days. If none found, use fallback.
            resolve_multiple (Callable, str): How to handle multiple values within the lookbehind window. Takes either i) a function that takes a list as an argument and returns a float, or ii) a str mapping to a callable from the resolve_multiple_fn catalogue.
            fallback (float): What to do if no value within the lookahead.
            new_col_name (str): Name to use for new col. Automatically generated as '{new_col_name}_within_{lookahead_days}_days'.
        """
        self.add_temporal_col_to_flattened_dataset(
            values_df=predictor_df,
            direction="behind",
            interval_days=lookbehind_days,
            resolve_multiple=resolve_multiple,
            fallback=fallback,
            new_col_name=new_col_name,
        )

    def add_temporal_col_to_flattened_dataset(
        self,
        values_df: Union[DataFrame, str],
        direction: str,
        interval_days: float,
        resolve_multiple: Union[Callable, str],
        fallback: float,
        new_col_name: Optional[str] = None,
    ):
        """Add a column to the dataset (either predictor or outcome depending
        on the value of "direction").

        Args:
            values_df (DataFrame): A table in wide format. Required columns: patient_id, timestamp, value.
            direction (str): Whether to look "ahead" or "behind".
            interval_days (float): How far to look in direction.
            resolve_multiple (Callable, str): How to handle multiple values within interval_days. Takes either i) a function that takes a list as an argument and returns a float, or ii) a str mapping to a callable from the resolve_multiple_fn catalogue.
            fallback (float): What to do if no value within the lookahead.
            new_col_name (str): Name to use for new column. Automatically generated as '{new_col_name}_within_{lookahead_days}_days'.
        """
        timestamp_col_type = type(values_df[self.timestamp_col_name][0]).__name__

        if timestamp_col_type not in ["Timestamp"]:
            raise ValueError(
                f"{self.timestamp_col_name} is of type {timestamp_col_type}, not 'Timestamp' from Pandas. Will cause problems. Convert before initialising FlattenedDataset.",
            )

        if direction == "behind":
            new_col_name_prefix = self.predictor_col_name_prefix
        elif direction == "ahead":
            new_col_name_prefix = self.outcome_col_name_prefix

        df = FlattenedDataset.flatten_temporal_values_to_df(
            prediction_times_with_uuid_df=self.df[
                [
                    self.id_col_name,
                    self.timestamp_col_name,
                    self.pred_time_uuid_col_name,
                ]
            ],
            values_df=values_df,
            direction=direction,
            interval_days=interval_days,
            resolve_multiple=resolve_multiple,
            fallback=fallback,
            new_col_name=new_col_name,
            id_col_name=self.id_col_name,
            timestamp_col_name=self.timestamp_col_name,
            pred_time_uuid_col_name=self.pred_time_uuid_col_name,
            new_col_name_prefix=new_col_name_prefix,
        )

        self.df = pd.merge(
            self.df,
            df,
            how="left",
            on=self.pred_time_uuid_col_name,
            validate="1:1",
        )

    @staticmethod
    def flatten_temporal_values_to_df(
        prediction_times_with_uuid_df: DataFrame,
        values_df: Union[Callable, DataFrame],
        direction: str,
        interval_days: float,
        resolve_multiple: Union[Callable, str],
        fallback: Union[float, str],
        id_col_name: str,
        timestamp_col_name: str,
        pred_time_uuid_col_name: str,
        new_col_name: str,
        new_col_name_prefix: Optional[str] = None,
        values_to_load: Optional[str] = None,
    ) -> DataFrame:

        """Create a dataframe with flattened values (either predictor or
        outcome depending on the value of "direction").

        Args:
            prediction_times_with_uuid_df (DataFrame): Dataframe with id_col and
                timestamps for each prediction time.
            values_df (Union[Callable, DataFrame]): A dataframe or callable resolving to
                a dataframe containing id_col, timestamp and value cols.
            direction (str): Whether to look "ahead" or "behind" the prediction time.
            interval_days (float): How far to look in each direction.
            resolve_multiple (Union[Callable, str]): How to handle multiple values
                within interval_days. Takes either
                i) a function that takes a list as an argument and returns a float, or
                ii) a str mapping to a callable from the resolve_multiple_fn catalogue.
            fallback (Union[float, str]): Which value to put if no value within the
                lookahead. "NaN" for Pandas NA.
            id_col_name (str): Name of id_column in prediction_times_with_uuid_df and
                values_df. Required because this is a static method.
            timestamp_col_name (str): Name of timestamp column in
                prediction_times_with_uuid_df and values_df. Required because this is a
                static method.
            pred_time_uuid_col_name (str): Name of uuid column in
                prediction_times_with_uuid_df. Required because this is a static method.
            new_col_name (str): Name of new column in returned
                dataframe.
            new_col_name_prefix (str, optional): Prefix to use for new column name.
                Defaults to None.
            values_to_load (str, optional): Which values to load from lab results.
                Takes either "numerical", "numerical_and_coerce", "cancelled" or "all".
                Defaults to None.

        Returns:
            DataFrame
        """
        msg = Printer(timestamp=True)

        # Rename column
        if new_col_name is None:
            raise ValueError("No name for new colum")

        full_col_str = generate_feature_colname(
            prefix=new_col_name_prefix,
            out_col_name=new_col_name,
            interval_days=interval_days,
            resolve_multiple=resolve_multiple,
            fallback=fallback,
            values_to_load=values_to_load,
        )

        # Resolve values_df if not already a dataframe.
        if isinstance(values_df, Callable):
            if values_to_load:
                msg.info(f"Loading values for {full_col_str}")
                values_df = values_df(values_to_load=values_to_load)
            else:
                values_df = values_df()

        if not isinstance(values_df, DataFrame):
            raise ValueError("values_df is not a dataframe")

        for col_name in [timestamp_col_name, id_col_name]:
            if col_name not in values_df.columns:
                raise ValueError(
                    f"{col_name} does not exist in df_prediction_times, change the df or set another argument",
                )

        # Generate df with one row for each prediction time x event time combination
        # Drop dw_ek_borger for faster merge
        df = pd.merge(
            left=prediction_times_with_uuid_df,
            right=values_df,
            how="left",
            on=id_col_name,
            suffixes=("_pred", "_val"),
            validate="m:m",
        ).drop("dw_ek_borger", axis=1)

        # Drop prediction times without event times within interval days
        df = FlattenedDataset.drop_records_outside_interval_days(
            df,
            direction=direction,
            interval_days=interval_days,
            timestamp_pred_colname="timestamp_pred",
            timestamp_value_colname="timestamp_val",
        )

        # Add back prediction times that don't have a value, and fill them with fallback
        df = FlattenedDataset.add_back_prediction_times_without_value(
            df=df,
            pred_times_with_uuid=prediction_times_with_uuid_df,
            pred_time_uuid_colname=pred_time_uuid_col_name,
        ).fillna(fallback)

        df["timestamp_val"].replace({fallback: pd.NaT}, inplace=True)

        df = FlattenedDataset.resolve_multiple_values_within_interval_days(
            resolve_multiple=resolve_multiple,
            df=df,
            timestamp_col_name=timestamp_col_name,
            pred_time_uuid_colname=pred_time_uuid_col_name,
        )

        # If resolve_multiple generates empty values,
        # e.g. when there is only one prediction_time within look_ahead window for slope calculation,
        # replace with NaN

        try:
            df["value"].replace({np.NaN: fallback}, inplace=True)
        except KeyError:
            print(full_col_str)
            print(df.columns)

        df.rename(
            {"value": full_col_str},
            axis=1,
            inplace=True,
        )

        # msg.good(f"Returning flattened dataframe with {full_col_str}")

        cols_to_return = [pred_time_uuid_col_name, full_col_str]

        return df[cols_to_return]

    @staticmethod
    def add_back_prediction_times_without_value(
        df: DataFrame,
        pred_times_with_uuid: DataFrame,
        pred_time_uuid_colname: str,
    ) -> DataFrame:
        """Ensure all prediction times are represented in the returned
        dataframe.

        Args:
            df (DataFrame): Dataframe with prediction times but without uuid.
            pred_times_with_uuid (DataFrame): Dataframe with prediction times and uuid.
            pred_time_uuid_colname (str): Name of uuid column in both df and pred_times_with_uuid.

        Returns:
            DataFrame: A merged dataframe with all prediction times.
        """
        return pd.merge(
            pred_times_with_uuid,
            df,
            how="left",
            on=pred_time_uuid_colname,
            suffixes=("", "_temp"),
        ).drop(["timestamp_pred"], axis=1)

    @staticmethod
    def resolve_multiple_values_within_interval_days(
        resolve_multiple: Callable,
        df: DataFrame,
        timestamp_col_name: str,
        pred_time_uuid_colname: str,
    ) -> DataFrame:
        """Apply the resolve_multiple function to prediction_times where there
        are multiple values within the interval_days lookahead.

        Args:
            resolve_multiple (Callable): Takes a grouped df and collapses each group to one record (e.g. sum, count etc.).
            df (DataFrame): Source dataframe with all prediction time x val combinations.
            timestamp_col_name (str): Name of timestamp column in df.
            pred_time_uuid_colname (str): Name of uuid column in df.

        Returns:
            DataFrame: DataFrame with one row pr. prediction time.
        """
        # Convert timestamp val to numeric that can be used for resolve_multiple functions
        df["timestamp_val"] = df["timestamp_val"].map(dt.datetime.toordinal)

        # Sort by timestamp_pred in case resolve_multiple needs dates
        df = df.sort_values(by=timestamp_col_name).groupby(pred_time_uuid_colname)

        if isinstance(resolve_multiple, str):
            resolve_multiple = resolve_fns.get(resolve_multiple)

        if isinstance(resolve_multiple, Callable):
            df = resolve_multiple(df).reset_index()
        else:
            raise ValueError("resolve_multiple must be or resolve to a Callable")

        return df

    @staticmethod
    def drop_records_outside_interval_days(
        df: DataFrame,
        direction: str,
        interval_days: float,
        timestamp_pred_colname: str,
        timestamp_value_colname: str,
    ) -> DataFrame:
        """Keep only rows where timestamp_value is within interval_days in
        direction of timestamp_pred.

        Args:
            direction (str): Whether to look ahead or behind.
            interval_days (float): How far to look
            df (DataFrame): Source dataframe
            timestamp_pred_colname (str): Name of timestamp column for predictions in df.
            timestamp_value_colname (str): Name of timestamp column for values in df.

        Raises:
            ValueError: If direction is niether ahead nor behind.

        Returns:
            DataFrame
        """
        df["time_from_pred_to_val_in_days"] = (
            (df[timestamp_value_colname] - df[timestamp_pred_colname])
            / (np.timedelta64(1, "s"))
            / 86_400
        )
        # Divide by 86.400 seconds/day

        if direction == "ahead":
            df["is_in_interval"] = (
                df["time_from_pred_to_val_in_days"] <= interval_days
            ) & (df["time_from_pred_to_val_in_days"] > 0)
        elif direction == "behind":
            df["is_in_interval"] = (
                df["time_from_pred_to_val_in_days"] >= -interval_days
            ) & (df["time_from_pred_to_val_in_days"] < 0)
        else:
            raise ValueError("direction can only be 'ahead' or 'behind'")

        return df[df["is_in_interval"]].drop(
            ["is_in_interval", "time_from_pred_to_val_in_days"],
            axis=1,
        )


def select_and_assert_keys(dictionary: Dict, key_list: List[str]) -> Dict:
    """Keep only the keys in the dictionary that are in key_order, and orders
    them as in the lsit.

    Args:
        dictionary (Dict): Dictionary to process
        key_list (List[str]): List of keys to keep

    Returns:
        Dict: Dict with only the selected keys
    """
    for key in key_list:
        if key not in dictionary:
            raise KeyError(f"{key} not in dict")

    return {key: dictionary[key] for key in key_list if key in dictionary}

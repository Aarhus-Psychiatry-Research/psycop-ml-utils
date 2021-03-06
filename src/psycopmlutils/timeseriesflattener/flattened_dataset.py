from datetime import timedelta
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from catalogue import Registry  # noqa
from pandas import DataFrame
from wasabi import msg

from psycopmlutils.timeseriesflattener.resolve_multiple_functions import resolve_fns
from psycopmlutils.utils import data_loaders


class FlattenedDataset:
    """Turn a set of time-series into tabular prediction-time data."""

    def __init__(
        self,
        prediction_times_df: DataFrame,
        id_col_name: str = "dw_ek_borger",
        timestamp_col_name: str = "timestamp",
        min_date: Optional[pd.Timestamp] = None,
        n_workers: int = 60,
        predictor_col_name_prefix: str = "pred",
        outcome_col_name_prefix: str = "outc",
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

            if "source_values_col_name" not in arg_dict.keys():
                arg_dict["source_values_col_name"] = "value"

            # Resolve values_df to either a dataframe from predictor_dfs_dict or a callable from the registry
            if predictor_dfs is None:
                predictor_dfs = self.loaders_catalogue.get_all()
            else:
                predictor_dfs = {
                    **predictor_dfs,
                    **self.loaders_catalogue.get_all(),
                }

            try:
                arg_dict["values_df"] = predictor_dfs[arg_dict["values_df"]]
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
                "source_values_col_name",
                "new_col_name_prefix",
            ]

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
        )

        self.df = self.df.copy()

    def _validate_processed_arg_dicts(self, arg_dicts: list):
        warn = False

        for d in arg_dicts:
            if not isinstance(d["values_df"], (DataFrame, Callable)):
                msg.warn(
                    f"values_df resolves to neither a Callable nor a DataFrame in {d}",
                )
                warn = True

            if not (d["direction"] == "ahead" or d["direction"] == "behind"):
                msg.warn(f"direction is neither ahead or behind in {d}")
                warn = True

            if not isinstance(d["interval_days"], (int, float)):
                msg.warn(f"interval_days is neither an int nor a float in {d}")
                warn = True

        if warn:
            raise ValueError(
                "Errors in argument dictionaries, didn't generate any features.",
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
            prediction_times_with_uuid_df=self.df,
            id_col_name=self.id_col_name,
            timestamp_col_name=self.timestamp_col_name,
            pred_time_uuid_col_name=self.pred_time_uuid_col_name,
            **kwargs_dict,
        )

    def add_age(
        self,
        id2date_of_birth: DataFrame,
        date_of_birth_col_name: str = "date_of_birth",
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
        prefix: str = "self.predictor_col_name_prefix",
        input_col_name: str = None,
        output_col_name: str = None,
    ):
        """Add static info to each prediction time, e.g. age, sex etc.

        Args:
            predictor_df (DataFrame): Contains an id_column and a value column.
            prefix (str): Prefix for column. Defaults to self.predictor_col_name_prefix.
            value_col_name (str, optional): Column names for the values you want to add. Defaults to "value".
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
        )

    def add_temporal_outcome(
        self,
        outcome_df: DataFrame,
        lookahead_days: float,
        resolve_multiple: Union[Callable, str],
        fallback: float,
        incident: Optional[bool] = False,
        outcome_df_values_col_name: str = "value",
        new_col_name: str = "value",
        is_fallback_prop_warning_threshold: float = 0.9,
        keep_outcome_timestamp: bool = True,
        dichotomous: bool = False,
    ):
        """Add an outcome-column to the dataset.

        Args:
            outcome_df (DataFrame): A table in wide format. Required columns: patient_id, timestamp, value.
            lookahead_days (float): How far ahead to look for an outcome in days. If none found, use fallback.
            resolve_multiple (Callable, str): How to handle multiple values within the lookahead window. Takes either i) a function that takes a list as an argument and returns a float, or ii) a str mapping to a callable from the resolve_multiple_fn catalogue.
            fallback (float): What to do if no value within the lookahead.
            incident (Optional[bool], optional): Whether looking for an incident outcome. If true, removes all prediction times after the outcome time. Defaults to false.
            outcome_df_values_col_name (str): Column name for the outcome values in outcome_df, e.g. whether a patient has t2d or not at the timestamp. Defaults to "value".
            new_col_name (str): Name to use for new col. Automatically generated as '{new_col_name}_within_{lookahead_days}_days'. Defaults to "value".
            is_fallback_prop_warning_threshold (float, optional): Triggers a ValueError if proportion of
                prediction_times that receive fallback is larger than threshold.
                Indicates unlikely to be a learnable feature. Defaults to 0.9.
            keep_outcome:timestamp (bool, optional): Whether to keep the timestamp for the outcome value as a separate column. Defaults to False.
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
                source_values_col_name=outcome_df_values_col_name,
                is_fallback_prop_warning_threshold=is_fallback_prop_warning_threshold,
                keep_val_timestamp=keep_outcome_timestamp,
                new_col_name=new_col_name,
            )

    def add_temporal_predictor(
        self,
        predictor_df: DataFrame,
        lookbehind_days: float,
        resolve_multiple: Union[Callable, str],
        fallback: float,
        source_values_col_name: str = "value",
        new_col_name: str = None,
    ):
        """Add a column with predictor values to the flattened dataset (e.g.
        "average value of bloodsample within n days").

        Args:
            predictor_df (DataFrame): A table in wide format. Required columns: patient_id, timestamp, value.
            lookbehind_days (float): How far behind to look for a predictor value in days. If none found, use fallback.
            resolve_multiple (Callable, str): How to handle multiple values within the lookbehind window. Takes either i) a function that takes a list as an argument and returns a float, or ii) a str mapping to a callable from the resolve_multiple_fn catalogue.
            fallback (List[str]): What to do if no value within the lookahead.
            source_values_col_name (str): Column name for the predictor values in predictor_df, e.g. the patient's most recent blood-sample value. Defaults to "value".
            new_col_name (str): Name to use for new col. Automatically generated as '{new_col_name}_within_{lookahead_days}_days'.
        """
        self.add_temporal_col_to_flattened_dataset(
            values_df=predictor_df,
            direction="behind",
            interval_days=lookbehind_days,
            resolve_multiple=resolve_multiple,
            fallback=fallback,
            source_values_col_name=source_values_col_name,
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
        source_values_col_name: str = "value",
        is_fallback_prop_warning_threshold: float = 0.9,
        keep_val_timestamp: bool = False,
    ):
        """Add a column to the dataset (either predictor or outcome depending
        on the value of "direction").

        Args:
            values_df (DataFrame): A table in wide format. Required columns: patient_id, timestamp, value.
            direction (str): Whether to look "ahead" or "behind".
            interval_days (float): How far to look in direction.
            resolve_multiple (Callable, str): How to handle multiple values within interval_days. Takes either i) a function that takes a list as an argument and returns a float, or ii) a str mapping to a callable from the resolve_multiple_fn catalogue.
            fallback (List[str]): What to do if no value within the lookahead.
            new_col_name (str): Name to use for new column. Automatically generated as '{new_col_name}_within_{lookahead_days}_days'.
            source_values_col_name (str, optional): Column name of the values column in values_df. Defaults to "val".
            is_fallback_prop_warning_threshold (float, optional): Triggers a ValueError if proportion of
                prediction_times that receive fallback is larger than threshold.
                Indicates unlikely to be a learnable feature. Defaults to 0.9.
            keep_val_timestamp (bool, optional): Whether to keep the timestamp for the temporal value as a separate column. Defaults to False.
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
            prediction_times_with_uuid_df=self.df,
            values_df=values_df,
            direction=direction,
            interval_days=interval_days,
            resolve_multiple=resolve_multiple,
            fallback=fallback,
            new_col_name=new_col_name,
            id_col_name=self.id_col_name,
            timestamp_col_name=self.timestamp_col_name,
            pred_time_uuid_col_name=self.pred_time_uuid_col_name,
            source_values_col_name=source_values_col_name,
            is_fallback_prop_warning_threshold=is_fallback_prop_warning_threshold,
            keep_val_timestamp=keep_val_timestamp,
            new_col_name_prefix=new_col_name_prefix,
        )

        self.df = pd.merge(self.df, df, how="left", on=self.pred_time_uuid_col_name)

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
        source_values_col_name: str = "value",
        is_fallback_prop_warning_threshold: float = 0.9,
        low_variance_threshold: float = 0.01,
        keep_val_timestamp: bool = False,
        new_col_name_prefix: str = None,
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
            source_values_col_name (str, optional): Name of column containing values in
                values_df. Defaults to "value".
            is_fallback_prop_warning_threshold (float, optional): Triggers a ValueError
                if proportion of prediction_times that receive fallback is larger than
                threshold. Indicates unlikely to be a learnable feature. Defaults to
                0.9.
            keep_val_timestamp (bool, optional): Whether to keep the timestamp for the
                temporal value as a separate column. Defaults to False.


        Returns:
            DataFrame
        """
        # Rename column
        if new_col_name is None:
            raise ValueError("No name for new colum")

        full_col_str = f"{new_col_name_prefix}_{new_col_name}_within_{interval_days}_days_{resolve_multiple}_fallback_{fallback}"

        # Resolve values_df if not already a dataframe.
        if isinstance(values_df, Callable):
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
            prediction_times_with_uuid_df,
            values_df,
            how="left",
            on=id_col_name,
            suffixes=("_pred", "_val"),
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

        df = FlattenedDataset.resolve_multiple_values_within_interval_days(
            resolve_multiple=resolve_multiple,
            df=df,
            timestamp_col_name=timestamp_col_name,
            pred_time_uuid_colname=pred_time_uuid_col_name,
        )

        df.rename(
            {"value": full_col_str},
            axis=1,
            inplace=True,
        )

        if direction == "ahead":
            if is_fallback_prop_warning_threshold is not None:
                prop_of_values_that_are_fallback = (
                    df[df[full_col_str] == fallback].shape[0] / df.shape[0]
                )

                if (
                    prop_of_values_that_are_fallback
                    > is_fallback_prop_warning_threshold
                ):
                    msg.warn(
                        f"""{full_col_str}: Beware, {prop_of_values_that_are_fallback*100}% of rows contain the fallback value, indicating that it is unlikely to be a learnable feature. Consider redefining. You can generate the feature anyway by passing an is_fallback_prop_warning_threshold argument with a higher threshold or None.""",
                    )

            if low_variance_threshold is not None:
                variance_as_fraction_of_mean = (
                    df[full_col_str].var() / df[full_col_str].mean()
                )
                if variance_as_fraction_of_mean < low_variance_threshold:
                    msg.warn(
                        f"""{full_col_str}: Beware, variance / mean < low_variance_threshold ({variance_as_fraction_of_mean} < {low_variance_threshold}), indicating high risk of overfitting. Consider redefining. You can generate the feature anyway by passing an low_variance_threshold argument with a lower threshold or None.""",
                    )

        msg.good(f"Returning flattened dataframe with {full_col_str}")

        cols_to_return = [pred_time_uuid_col_name, full_col_str]

        if keep_val_timestamp:
            timestamp_val_name = f"timestamp_{full_col_str}"

            df.rename(
                {"timestamp_val": timestamp_val_name},
                axis=1,
                inplace=True,
            )

            cols_to_return.append(timestamp_val_name)

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
            df (DataFrame):

        Returns:
            DataFrame:
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

        Returns:
            DataFrame: DataFrame with one row pr. prediction time.
        """
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
            timestamp_pred_colname (str):
            timestamp_value_colname (str):

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
        dict (Dict): Dictionary to process
        keys_to_keep (List[str]): List of keys to keep

    Returns:
        Dict: Dict with only the selected keys
    """
    for key in key_list:
        if key not in dictionary:
            raise KeyError(f"{key} not in dict")

    return {key: dictionary[key] for key in key_list if key in dictionary}

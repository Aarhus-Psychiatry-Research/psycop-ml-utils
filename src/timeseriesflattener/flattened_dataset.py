from typing import Callable, Dict, List, Union, Tuple
from pandas import DataFrame
from datetime import datetime


class FlattenedDataset:
    def __init__(
        self,
        prediction_times_df: DataFrame,
        timestamp_colname: str = "timestamp",
        id_colname: str = "dw_ek_borger",
    ):
        """Class containing a time-series, flattened.

        Args:
            prediction_times_df (DataFrame): Dataframe with prediction times.
            timestamp_colname (str, optional): Colname for timestamps. Is used across outcomes and predictors. Defaults to "timestamp".
            id_colname (str, optional): Colname for patients ids. Is used across outcome and predictors. Defaults to "dw_ek_borger".
        """
        self.df_prediction_times = prediction_times_df
        self.timestamp_colname = timestamp_colname
        self.id_colname = id_colname

        self.df = self.df_prediction_times

    def add_outcome(
        self,
        outcome_df: DataFrame,
        lookahead_days: float,
        resolve_multiple: Callable,
        fallback: float,
        source_values_colname: str = "val",
        new_col_name: str = None,
    ):
        """Adds an outcome-column to the dataset

        Args:
            outcome_df (DataFrame): Cols: dw_ek_borger, datotid, value if relevant.
            lookahead_days (float): How far ahead to look for an outcome in days. If none found, use fallback.
            resolve_multiple (Callable): How to handle multiple values within the lookahead window. Takes a a function that takes a list as an argument and returns a float.
            fallback (float): What to do if no value within the lookahead.
            source_values_colname (str): Colname for outcome values in outcome_df. Defaults to "val".
            new_col_name (str): Name to use for new col. Automatically generated as '{new_col_name}_within_{lookahead_days}_days'.
        """

        self._add_col(
            values_df=outcome_df,
            direction="ahead",
            interval_days=lookahead_days,
            resolve_multiple=resolve_multiple,
            fallback=fallback,
            new_col_name=new_col_name,
            source_values_colname=source_values_colname,
        )

    def add_predictor(
        self,
        predictor_df: DataFrame,
        lookbehind_days: float,
        resolve_multiple: str,
        fallback: float,
        source_values_colname: str = "val",
        new_col_name: str = None,
    ):
        """Adds a predictor-column to the dataset

        Args:
            predictor_df (DataFrame): Cols: dw_ek_borger, datotid, value if relevant.
            lookbehind_days (float): How far behind to look for a predictor value in days. If none found, use fallback.
            resolve_multiple (Callable): How to handle multiple values within the lookbehind window. Takes a a function that takes a list as an argument and returns a float.
            fallback (List[str]): What to do if no value within the lookahead.
            new_col_name (str): Name to use for new col. Automatically generated as '{new_col_name}_within_{lookahead_days}_days'.
        """

        self._add_col(
            values_df=predictor_df,
            direction="behind",
            interval_days=lookbehind_days,
            resolve_multiple=resolve_multiple,
            fallback=fallback,
            new_col_name=new_col_name,
            source_values_colname=source_values_colname,
        )

    def _add_col(
        self,
        values_df: DataFrame,
        direction: str,
        interval_days: float,
        resolve_multiple: str,
        fallback: float,
        new_col_name: str,
        source_values_colname: str = "val",
    ):
        """Adds a value-column to the dataset

        Args:
            values_df (DataFrame): Cols: dw_ek_borger, datotid, value.
            direction (str): Whether to look "ahead" or "behind".
            interval_days (float): How far to look in direction.
            resolve_multiple (Callable): How to handle multiple values within the lookbehind window. Takes a a function that takes a list as an argument and returns a float.
            fallback (List[str]): What to do if no value within the lookahead.
            new_col_name (str): Name to use for new col. Automatically generated as '{new_col_name}_within_{lookahead_days}_days'.
            source_values_colname (str, optional): Values colname in the values_df. Defaults to "values".
        """

        values_dict = self._events_to_dict_by_patient(
            df=values_df,
            values_colname=source_values_colname,
        )

        new_col = self.df_prediction_times.apply(
            lambda row: self._flatten_events_for_prediction_time(
                direction=direction,
                prediction_timestamp=row[self.timestamp_colname],
                val_dict=values_dict,
                interval_days=interval_days,
                id=row[self.id_colname],
                resolve_multiple=resolve_multiple,
                fallback=fallback,
            ),
            axis=1,
        )

        if new_col_name is None:
            new_col_name = source_values_colname

        self.df[f"{new_col_name}_within_{interval_days}_days"] = new_col

    def _events_to_dict_by_patient(
        self,
        df: DataFrame,
        values_colname: str,
    ) -> Dict[str, List[Tuple[Union[datetime, float]]]]:
        """
        Generate a dict of events grouped by patient_id

        Args:
            df (DataFrame): Dataframe to come from
            values_colname (str): Column name for event values

        Returns:
            Dict[str, List[Tuple[Union[datetime, float]]]]:
                                    {
                                        patientid1: [(timestamp11, val11), (timestamp12, val12)],
                                        patientid2: [(timestamp21, val21), (timestamp22, val22)]
                                    }
        """

        return (
            df.groupby(self.id_colname)
            .apply(
                lambda row: tuple(
                    [
                        list(event)
                        for event in zip(
                            row[self.timestamp_colname], row[values_colname]
                        )
                    ]
                )
            )
            .to_dict()
        )

    def _get_events_within_n_days(
        self,
        direction: str,
        prediction_timestamp: datetime,
        val_dict: Dict[str, List[Tuple[Union[datetime, float]]]],
        interval_days: float,
        id: int,
    ) -> List:
        """Gets a list of values that are within interval_days in direction from prediction_timestamp for id.

        Args:
            direction (str): Whether to look ahead or behind.
            prediction_timestamp (timestamp):
            val_dict (Dict[str, List[Tuple[Union[datetime, float]]]]): A dict containing the timestamps and vals for the events.
                Shaped like {patient_id: [(timestamp1: val1), (timestamp2: val2)]}
            interval_days (int): How far to look in direction.
            id (int): Patient id

        Returns:
            list: [datetime, value]
        """

        events_within_n_days = []

        for event in val_dict[id]:
            event_timestamp = event[0]

            if is_within_n_days(
                direction=direction,
                prediction_timestamp=prediction_timestamp,
                event_timestamp=event_timestamp,
                interval_days=interval_days,
            ):
                events_within_n_days.append(event)

        return events_within_n_days

    def _flatten_events_for_prediction_time(
        self,
        direction: str,
        prediction_timestamp: str,
        val_dict: Dict[str, List[Tuple[Union[datetime, float]]]],
        interval_days: float,
        resolve_multiple: Callable,
        fallback: list,
        id: int,
    ) -> float:
        """Takes a list of events and turns them into a single value for a prediction_time
        given a set of conditions.

        Args:
            direction (str): Whether to look ahead or behind from the prediction time.
            prediction_timestamp (str): The timestamp to anchor on.
            val_dict (Dict[str, List[Tuple[Union[datetime, float]]]]): A dict containing the timestamps and vals for the events.
                Shaped like {patient_id: [(timestamp1: val1), (timestamp2: val2)]}
            interval_days (float): How many days to look in direction for events.
            resolve_multiple (str): How to handle multiple events within interval_days.
            fallback (list): How to handle no events within interval_days.
            id (int): Which patient ID to flatten events for.

        Returns:
            float: Value for each prediction_time.
        """
        events = self._get_events_within_n_days(
            direction=direction,
            prediction_timestamp=prediction_timestamp,
            val_dict=val_dict,
            interval_days=interval_days,
            id=id,
        )

        if len(events) == 0:
            return fallback
        elif len(events) == 1:
            event_val = events[0][1]
            return event_val
        elif len(events) > 1:
            return resolve_multiple(events)


def is_within_n_days(
    direction: str,
    prediction_timestamp: datetime,
    event_timestamp: datetime,
    interval_days: float,
) -> bool:
    """Looks interval_days in direction from prediction_timestamp.
    Returns true if event_timestamp is within interval_days.

    Args:
        direction: Whether to look ahead or behind
        prediction_timestamp (timestamp): timestamp for prediction
        event_timestamp (timestamp): timestamp for event
        interval_days (int): How far to look in direction

    Returns:
        boolean
    """

    difference_in_days = (
        event_timestamp - prediction_timestamp
    ).total_seconds() / 86400
    # Use .seconds instead of .days to get fractions of a day

    if direction == "ahead":
        is_in_interval = difference_in_days <= interval_days and difference_in_days > 0
    elif direction == "behind":
        is_in_interval = difference_in_days >= -interval_days and difference_in_days < 0
    else:
        raise ValueError("direction can only be 'ahead' or 'behind'")

    return is_in_interval
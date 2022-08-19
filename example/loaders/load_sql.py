import psycopmlutils.loaders
from psycopmlutils.loaders.sql_load import sql_load

if __name__ == "__main__":
    view = "[FOR_kohorte_demografi_inkl_2021_feb2022]"
    sql = "SELECT dw_ek_borger FROM [fct]." + view
    kohorte_demografi_ids = sql_load(
        sql,
        chunksize=None,
        format_timestamp_cols_to_datetime=False,
    )

    prediction_times = psycopmlutils.loaders.LoadVisits.physical_visits_to_psychiatry()

    # Find how many unique values there are in prediction times dw_ek_borger column
    n_prediction_time_ids = prediction_times["dw_ek_borger"].nunique()
    n_demografi_ids = kohorte_demografi_ids["dw_ek_borger"].nunique()

    print(f"Missing {n_demografi_ids/n_prediction_time_ids}% of ids")

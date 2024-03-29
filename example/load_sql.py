"""Example of how to load IDs from sql."""

from psycop_ml_utils.sql.loader import sql_load

if __name__ == "__main__":
    VIEW = "[psycop_t2d_train]"
    SQL = "SELECT * FROM [fct]." + VIEW
    df = sql_load(SQL, chunksize=None, format_timestamp_cols_to_datetime=False)

from psycopmlutils.loaders import LoadVisits

if __name__ == "__main__":
    df = LoadVisits.physical_visits_from_psychiatry(
        where_clause="dw_ek_borger = '5296254'",
    )

    # Sort dataframe by timestamp
    df.sort_values(by="timestamp", inplace=True)

    # Only one duplicate visit, why does this result in duplicates?

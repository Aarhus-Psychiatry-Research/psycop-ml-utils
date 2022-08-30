from psycopmlutils.loaders.raw import LoadVisits

if __name__ == "__main__":
    df = LoadVisits.physical_visits_to_psychiatry(
        #  where_clause="dw_ek_borger = '5296254'",
    )

    # Only one duplicate visit, why does this result in duplicates

    # Get duplicates from df
    duplicate_series = df.duplicated(subset=["timestamp", "dw_ek_borger"], keep=False)
    duplicates = df[duplicate_series].sort_values(by=["dw_ek_borger"])

    pass

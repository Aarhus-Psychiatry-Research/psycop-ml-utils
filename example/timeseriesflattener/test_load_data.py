from loaders.diagnoses import LoadDiagnoses
from wasabi import msg

if __name__ == "__main__":
    df = LoadDiagnoses.diagnoses_from_physical_visits(
        icd_str=["E780", "E785"],
        new_col_str="hypercholesterolemia",
        wildcard_icd_10_end=False,
    )

    msg.info(f"Columns: {df.columns}")

    pass

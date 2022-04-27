from loaders.diagnoses import LoadDiagnoses
from wasabi import msg

if __name__ == "__main__":
    df = LoadDiagnoses.diagnoses_from_physical_visits(icd_str="F20", depth=3)

    msg.info(f"Columns: {df.columns}")

    pass

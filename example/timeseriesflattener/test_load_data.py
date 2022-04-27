from loaders.medications import LoadMedications
from wasabi import msg

if __name__ == "__main__":
    df = LoadMedications.medication(atc_str="A10")

    msg.info(f"Columns: {df.columns}")

    pass

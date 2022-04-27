from load_data import LoadData
from wasabi import msg

if __name__ == "__main__":
    df = LoadData.medication(atc_str="A10")

    msg.info(f"Columns: {df.columns}")

    pass

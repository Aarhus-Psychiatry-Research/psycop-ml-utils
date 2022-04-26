from load_data import LoadData
from wasabi import msg

if __name__ == "__main__":
    df = LoadData.egfr()

    msg.info(f"Columns: {df.columns}")

    pass

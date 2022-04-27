import pandas as pd
from loaders import sql_load
from sqlalchemy import true
from wasabi import msg


class LoadData:
    def physical_visits(frac=None):
        msg.info("Loading physical visits")

        view = "[FOR_besoeg_fysiske_fremmoeder_inkl_2021_feb2022]"
        sql = f"SELECT dw_ek_borger, datotid_start FROM [fct].{view} WHERE besoeg=1"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None)

        if frac is not None:
            df = df.sample(frac=frac)

        df.rename(columns={"datotid_start": "timestamp"}, inplace=True)

        msg.good("Loaded physical visits")
        return df.reset_index(drop=True)

    def t2d_times():
        from pathlib import Path

        msg.info("Loading t2d event times")

        full_csv_path = Path(
            r"C:\Users\adminmanber\Desktop\manber-t2d\csv\first_t2d_diagnosis.csv"
        )

        df = pd.read_csv(str(full_csv_path))
        df = df[["dw_ek_borger", "datotid_first_t2d_diagnosis"]]
        df["val"] = 1

        df.rename(columns={"datotid_first_t2d_diagnosis": "timestamp"}, inplace=True)

        msg.good("Finished loading t2d event times")
        return df.reset_index(drop=True)

    def birthdays():
        msg.info("Loading birthdays")

        view = "[FOR_kohorte_demografi_inkl_2021_feb2022]"
        sql = f"SELECT dw_ek_borger, foedselsdato FROM [fct].{view}"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None)

        msg.good("Loaded birthdays")
        return df.reset_index(drop=True)

    def sex():
        msg.info("Loading sexes")

        view = "[FOR_kohorte_demografi_inkl_2021_feb2022]"
        sql = f"SELECT dw_ek_borger, koennavn FROM [fct].{view}"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None)

        df.loc[df["koennavn"] == "Mand", "koennavn"] = 1
        df.loc[df["koennavn"] == "Kvinde", "koennavn"] = 0

        df.rename(
            columns={"koennavn": "male"},
            inplace=True,
        )

        msg.good("Loaded sexes")
        return df.reset_index(drop=True)

    def hba1c():
        return LoadData.blood_sample(blood_sample_id="NPU27300", new_col_str="hba1c")

    def scheduled_glc():
        npu_suffixes = [
            "08550",
            "08551",
            "08552",
            "08553",
            "08554",
            "08555",
            "08556",
            "08557",
            "08558",
            "08559",
            "08560",
            "08561",
            "08562",
            "08563",
            "08564",
            "08565",
            "08566",
            "08567",
            "08893",
            "08894",
            "08895",
            "08896",
            "08897",
            "08898",
            "08899",
            "08900",
            "08901",
            "08902",
            "08903",
            "08904",
            "08905",
            "08906",
            "08907",
            "08908",
            "08909",
            "08910",
            "08911",
            "08912",
            "08913",
            "08914",
            "08915",
            "08916",
        ]

        blood_sample_ids = [f"NPU{suffix}" for suffix in npu_suffixes]

        return LoadData.aggregate_blood_samples(
            blood_sample_ids=blood_sample_ids, new_col_str="scheduled_p_glc"
        )

    def unscheduled_p_glc():
        npu_suffixes = [
            "02192",
            "21533",
            "21531",
        ]

        dnk_suffixes = ["35842"]

        blood_sample_ids = [f"NPU{suffix}" for suffix in npu_suffixes]
        blood_sample_ids += [f"DNK{suffix}" for suffix in dnk_suffixes]

        return LoadData.aggregate_blood_samples(
            blood_sample_ids=blood_sample_ids, new_col_str="unscheduled_p_glc"
        )

    def triglycerides():
        return LoadData.blood_sample(
            blood_sample_id="NPU04094", new_col_str="triglyceride"
        )

    def fasting_triglycerides():
        return LoadData.blood_sample(
            blood_sample_id="NPU03620", new_col_str="fasting_triglyceride"
        )

    def hdl():
        return LoadData.blood_sample(blood_sample_id="NPU01567", new_col_str="hdl")

    def ldl():
        return LoadData.aggregate_blood_samples(
            blood_sample_id=["NPU01568", "AAB00101"], new_col_str="ldl"
        )

    def fasting_ldl():
        return LoadData.aggregate_blood_samples(
            blood_sample_ids=["NPU10171", "AAB00102"], new_col_str="fasting_ldl"
        )

    def alat():
        return LoadData.blood_sample(blood_sample_id="NPU19651", new_col_str="alat")

    def asat():
        return LoadData.blood_sample(blood_sample_id="NPU19654", new_col_str="asat")

    def lymphocytes():
        return LoadData.blood_sample(
            blood_sample_id="NPU02636", new_col_str="lymphocytes"
        )

    def leukocytes():
        return LoadData.blood_sample(
            blood_sample_id="NPU02593", new_col_str="leukocytes"
        )

    def crp():
        return LoadData.blood_sample(blood_sample_id="NPU19748", new_col_str="crp")

    def creatinine():
        return LoadData.aggregate_blood_samples(
            blood_sample_ids=["NPU18016", "ASS00355", "ASS00354"],
            new_col_str="creatinine",
        )

    def egfr():
        return LoadData.aggregate_blood_samples(
            blood_sample_ids=["DNK35302", "DNK35131", "AAB00345", "AAB00343"],
            new_col_str="egfr",
        )

    def albumine_creatinine_ratio():
        return LoadData.blood_sample(
            blood_sample_id="NPU19661", new_col_str="albumine_creatinine_ratio"
        )

    def blood_sample(blood_sample_id: str, new_col_str=None) -> pd.DataFrame:
        print_str = f"blood samples matching NPU-code {blood_sample_id}"
        msg.info(f"Loading {print_str}")

        view = "[FOR_labka_alle_blodprover_inkl_2021_feb2022]"
        sql = f"SELECT dw_ek_borger, datotid_sidstesvar, numerisksvar FROM [fct].{view} WHERE NPUkode = '{blood_sample_id}'"

        df = sql_load(sql, database="USR_PS_FORSK", chunksize=None)

        if new_col_str == None:
            new_col_str = blood_sample_id

        df.rename(
            columns={
                "datotid_sidstesvar": "timestamp",
                "numerisksvar": f"{new_col_str}_val",
            },
            inplace=True,
        )

        msg.good(f"Loaded {print_str}")
        return df.reset_index(drop=True)

    def aggregate_blood_samples(blood_sample_ids: list, new_col_str=None):
        dfs = [
            LoadData.blood_sample(blood_sample_id=f"{id}", new_col_str=new_col_str)
            for id in blood_sample_ids
        ]

        return pd.concat(dfs, axis=0).reset_index(drop=True)

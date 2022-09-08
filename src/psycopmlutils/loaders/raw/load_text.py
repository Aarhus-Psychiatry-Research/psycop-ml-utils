from pathlib import Path
from typing import List, Optional, Set, Union

import dill as pkl
import numpy as np
import pandas as pd

from psycopmlutils.loaders.raw.sql_load import sql_load
from psycopmlutils.utils import data_loaders


class LoadText:
    def get_valid_note_types() -> Set[str]:
        """Returns a set of valid note types. Notice that 'Konklusion' is
        replaced by 'Vurdering/konklusion' in 2020, so make sure to use both.
        'Ordination' was replaced by 'Ordination, Psykiatry' in 2022, but
        'Ordination, Psykiatri' is not included in the table. Use with caution.

        Returns:
            Set[str]: Set of valid note types
        """
        return {
            "Observation af patient, Psykiatri",
            "Samtale med behandlingssigte",
            "Ordination",  # OBS replaced "Ordination, Psykiatri" in 01/02-22
            # but is not included in this table. Use with caution
            "Aktuelt psykisk",
            "Aktuelt socialt, Psykiatri",
            "Aftaler, Psykiatri",
            "Medicin",
            "Aktuelt somatisk, Psykiatri",
            "Objektivt psykisk",
            "Kontaktårsag",
            "Telefonkonsultation",
            "Journalnotat",
            "Telefonnotat",
            "Objektivt, somatisk",
            "Plan",
            "Semistruktureret diagnostisk interview",
            "Vurdering/konklusion",
        }

    def load_and_featurize_notes(
        note_name: Union[str, List[str]],
        featurizer: str,
        n: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:

        valid_featurizers = {"tfidf", "huggingface", None}
        if featurizer not in valid_featurizers:
            raise ValueError(
                f"featurizer must be one of {valid_featurizers}, got {featurizer}",
            )

        if isinstance(note_name, str):
            note_name = [note_name]
        # check for invalid note types
        if not set(note_name).issubset(LoadText.get_valid_note_types()):
            raise ValueError(
                "Invalid note type. Valid note types are: "
                + str(LoadText.get_valid_note_types()),
            )

        # convert note_names to sql query
        note_names = "('" + "', '".join(note_name) + "')"

        view = "[FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret"

        dfs = []
        for year in [str(y) for y in np.arange(2011, 2021)]:
            df = LoadText.load_notes(note_names, year, view, n)
            if featurizer == "tfidf":
                df = LoadText._tfidf_featurize(df, **kwargs)
            elif featurizer == "huggingface":
                df = LoadText._huggingface_featurize(df, **kwargs)
            dfs.append(df)

        dfs = pd.concat(dfs)

        dfs = dfs.rename(
            {"datotid_senest_aendret_i_sfien": "timestamp", "fritekst": "text"},
            axis=1,
        )
        return dfs

    def load_notes(
        note_names: Union[str, list[str]],
        year: str,
        view: str = "[FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret",
        n: Optional[int] = None,
    ) -> pd.DataFrame:

        sql = (
            "SELECT dw_ek_borger, datotid_senest_aendret_i_sfien, fritekst"
            + f" FROM [fct].{view}_{year}_inkl_2021_feb2022]"
            + f" WHERE overskrift IN {note_names}"
        )
        return sql_load(sql, database="USR_PS_FORSK", chunksize=None, n=n)

    def _tfidf_featurize(df: pd.DataFrame, tfidf_path: Optional[Path]) -> pd.DataFrame:
        # Load tfidf model
        with open(tfidf_path, "rb") as f:
            tfidf = pkl.load(f)

        # get tfidf vocabulary
        vocab = tfidf.get_feature_names()

        text = df["text"].values
        df = df.drop("text", axis=1)

        text = tfidf.transform(text)
        text = pd.DataFrame(text.toarray(), columns=vocab)
        return pd.concat([df, text], axis=1)

    def _huggingface_featurize(model_id: str) -> pd.DataFrame:
        # Load paraphrase-multilingual-MiniLM-L12-v2
        #  split tokens to list of list if longer than allowed sequence length
        ## which is often 128 for sentence transformers
        # encode tokens
        ## average by list of list
        # return embeddings
        pass

    @data_loaders.register("all_notes")
    def load_all_notes(featurizer: str, n: Optional[int] = None) -> pd.DataFrame:
        return LoadText.load_and_featurize_notes(
            LoadText.get_valid_note_types(),
            featurizer=featurizer,
            n=n,
        )


if __name__ == "__main__":
    p = Path("tests") / "test_data"

    tfidf_path = p / "test_tfidf" / "tfidf_100.pkl"
    df_p = p / "synth_txt_data.csv"

    df = pd.read_csv(df_p)
    df = df.dropna()

    x = LoadText._tfidf_featurize(df, tfidf_path)

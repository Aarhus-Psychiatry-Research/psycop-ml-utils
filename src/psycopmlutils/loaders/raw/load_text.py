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
        featurizer_kwargs: Optional[dict] = None,
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Loads clinical notes from all years that match the specified note
        types. Featurizes the notes using the specified featurizer (tf-idf or
        huggingface model). Kwargs passed to.

        Args:
            note_name (Union[str, List[str]]): Which note types to load. See
                `LoadText.get_valid_note_types()` for valid note types.
            featurizer (str): Which featurizer to use. Either 'tf-idf' or 'huggingface'.
            featurizer_kwargs (Optional[dict]): Kwargs passed to the featurizer. Defaults to None.
                For tf-idf, this is `tfidf_path` to the vectorizer. For huggingface,
                this is `model_id` to the model.
            n (Optional[int], optional): How many rows to load. Defaults to None.

        Raises:
            ValueError: If given invalid featurizer
            ValueError: If given invlaid note type

        Returns:
            pd.DataFrame: Featurized clinical notes
        """

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
            df = LoadText._load_notes(note_names, year, view, n)
            if featurizer == "tfidf":
                df = LoadText._tfidf_featurize(df, **featurizer_kwargs)
            elif featurizer == "huggingface":
                df = LoadText._huggingface_featurize(df, **featurizer_kwargs)
            dfs.append(df)

        dfs = pd.concat(dfs)

        dfs = dfs.rename(
            {"datotid_senest_aendret_i_sfien": "timestamp", "fritekst": "text"},
            axis=1,
        )
        return dfs

    def _load_notes(
        note_names: Union[str, List[str]],
        year: str,
        view: Optional[
            str
        ] = "[FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret",
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Loads clinical notes from sql from a specified year and matching
        specified note types.

        Args:
            note_names (Union[str, list[str]]): Which types of notes to load.
            year (str): Which year to load
            view (str, optional): Which table to load.
                Defaults to "[FOR_SFI_fritekst_resultat_udfoert_i_psykiatrien_aendret".
            n (Optional[int], optional): Number of rows to load. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe with clinical notes
        """

        sql = (
            "SELECT dw_ek_borger, datotid_senest_aendret_i_sfien, fritekst"
            + f" FROM [fct].{view}_{year}_inkl_2021_feb2022]"
            + f" WHERE overskrift IN {note_names}"
        )
        return sql_load(sql, database="USR_PS_FORSK", chunksize=None, n=n)

    @staticmethod
    def _tfidf_featurize(df: pd.DataFrame, tfidf_path: Optional[Path]) -> pd.DataFrame:
        """TF-IDF featurize text. Assumes `df` to have a column named `text`.

        Args:
            df (pd.DataFrame): Dataframe with text column
            tfidf_path (Optional[Path]): Path to a sklearn tf-idf vectorizer

        Returns:
            pd.DataFrame: Original dataframe with tf-idf features appended
        """
        with open(tfidf_path, "rb") as f:
            tfidf = pkl.load(f)

        vocab = ["tfidf-" + word for word in tfidf.get_feature_names()]

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
    def load_all_notes(
        featurizer: str,
        n: Optional[int] = None,
        featurizer_kwargs: Optional[dict] = None,
    ) -> pd.DataFrame:
        return LoadText.load_and_featurize_notes(
            LoadText.get_valid_note_types(),
            featurizer=featurizer,
            n=n,
            featurizer_kwargs=featurizer_kwargs,
        )

    @data_loaders.register("aktuelt_psykisk")
    def load_aktuel_psykisk(featurizer: str, n: Optional[int] = None) -> pd.DataFrame:
        return LoadText.load_and_featurize_notes(
            "Aktuelt psykisk",
            featurizer=featurizer,
            n=n,
        )

    @data_loaders.register("load_arbirary_notes")
    def load_arbitrary_notes(
        note_names: Union[str, list[str]],
        featurizer: str,
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        return LoadText.load_and_featurize_notes(
            note_names,
            featurizer=featurizer,
            n=n,
        )

    @data_loaders.register("synth_notes")
    def load_synth_notes(featurizer: str) -> pd.DataFrame:
        p = Path("tests") / "test_data"
        df = pd.read_csv(p / "synth_txt_data.csv")
        df = df.dropna()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if featurizer == "tfidf":
            return LoadText._tfidf_featurize(
                df,
                tfidf_path=p / "test_tfidf" / "tfidf_10.pkl",
            )
        else:
            raise ValueError("Only tfidf featurizer supported for synth notes")

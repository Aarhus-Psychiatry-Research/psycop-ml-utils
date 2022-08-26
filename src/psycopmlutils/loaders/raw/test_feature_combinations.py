from typing import Dict, List, Optional, Union

from psycopmlutils.utils import data_loaders


def test_that_feature_combinations_return_values(
    predictor_dict: List[Dict[str, Union[str, float, int]]],
    n: Optional[int] = 100,
):
    """Test that all predictor_dfs in predictor_list return a valid df.

    Args:
        predictor_dict (Dict[str, Union[str, float, int]]): List of dictionaries where the key predictor_df maps to an SQL database.
        n (int, optional): Number of rows to test. Defaults to 100.
    """

    loader_fns_dict = data_loaders.get_all()

    dfs = []

    for d in predictor_dict:
        df = loader_fns_dict[d["predictor_df"]](n=n)

        if df.shape[0] == 0:
            raise ValueError(f"{d['predictor_df']} returned an empty df.")
        else:
            dfs.append(df)

    print(f"Checked {len(dfs)} predictor_dfs, all returned non-empty dfs")

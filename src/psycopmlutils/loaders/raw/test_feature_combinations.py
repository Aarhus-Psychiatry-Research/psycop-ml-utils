from typing import Dict, Union

from psycopmlutils.utils import data_loaders


def test_that_feature_combinations_return_values(
    predictor_dict: Dict[str, Union[str, float, int]],
):
    """Test that all predictor_dfs in predictor_list return a valid df.

    Args:
        predictor_dict (Dict[str, Union[str, float, int]]): List of dictionaries where the key predictor_df maps to an SQL database.
    """

    loader_fns_dict = data_loaders.get_all()

    dfs = []

    for d in predictor_dict.items():
        dfs.append(loader_fns_dict[d["predictor_df"]](n=100))

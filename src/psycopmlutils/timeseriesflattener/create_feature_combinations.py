import itertools
from typing import Dict, List, Union


def create_feature_combinations(
    arg_sets: Union[List[Dict[str, Union[str, List]]], Dict[str, Union[str, List]]],
) -> List[Dict[str, Union[str, float, int]]]:
    """Create feature combinations from a dictionary or list of dictionaries of
    feature specifications.

    Args:
        arg_sets (Union[List[Dict[str, Union[str, List]]], Dict[str, Union[str, List]]]):
            dict/list of dicts containing arguments for .add_predictor.

    Returns:
        List[Dict[str, Union[str, float, int]]]: All possible combinations of
            arguments.
    """
    if isinstance(arg_sets, dict):
        arg_sets = [arg_sets]
    feature_combinations = []
    for arg_set in arg_sets:
        feature_combinations.extend(create_feature_combinations_from_dict(arg_set))
    return feature_combinations


def create_feature_combinations_from_dict(
    d: Dict[str, Union[str, List]],
) -> List[Dict[str, Union[str, float, int]]]:
    """Create feature combinations from a dictionary of feature specifications.
    Only unpacks the top level of lists.

    Args:
        d (Dict[str]): A dictionary of feature specifications.

    Returns:
        List[Dict[str]]: List of all possible combinations of the arguments.
    """

    # Make all elements iterable
    d = {k: v if isinstance(v, list) else [v] for k, v in d.items()}
    keys, values = zip(*d.items())
    # Create all combinations of top level elements
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permutations_dicts

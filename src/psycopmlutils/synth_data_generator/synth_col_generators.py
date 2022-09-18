"""Column generators for synthetic data"""

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate_data_columns(
    predictors: Iterable[Dict],
    n_samples: int,
    df: pd.DataFrame,
    text_prompt: Optional[str],
) -> pd.DataFrame:
    """Generate a dataframe with columns from the predictors iterable.

    Args:
        predictors (Iterable[Dict]): A dict representing each column. Key is col_name (str), values is a dict with column_type (str), min (int) and max(int).
        n_samples (int): Number of rows to generate.
        df (pd.DataFrame): Dataframe to append to
        text_prompt (str): Text prompt to use for generating text data. Defaults to "The quick brown fox jumps over the lazy dog".

    Raises:
        ValueError: If column_type isn't either uniform_int, text, or datetime_uniform.

    Returns:
        pd.DataFrame: The generated dataframe.


    Example:
        >>> column_specifications = {
        >>>   "citizen_ids": {"column_type": "uniform_int", "min": 0, "max": 1_200_000},
        >>>   "timestamp": {"column_type": "datetime_uniform", "min": 0, "max": 5 * 365},
        >>>   "text": {"column_type": "text"},
        >>> }
        >>>
        >>> df = generate_synth_data(
        >>>     predictors=column_specifications,
        >>>     n_samples=100,
        >>>     text_prompt="The patient",
        >>> )
    """
    sequence = text_prompt if text_prompt else None

    for col_name, col_props in predictors.items():
        # np.nan objects turn into "nan" strings in the real life dataframe.
        # imitate this in the synthetic data as well.
        if "nan" in col_name:
            df = df.rename({col_name: col_name.replace("np.nan", "nan")}, axis=1)
            col_name = col_name.replace("np.nan", "nan")

        column_type = col_props["column_type"]

        df[col_name] = generate_col_from_specs(
            column_type=column_type,
            n_samples=n_samples,
            sequence=sequence,
            col_specs=col_props,
        )

        # If column has min and/or max, floor and ceil appropriately
        if df[col_name].dtype not in ["datetime64[ns]"]:
            if "min" in col_props:
                df[col_name] = df[col_name].clip(lower=col_props["min"])
            if "max" in col_props:
                df[col_name] = df[col_name].clip(upper=col_props["max"])

    return df


def generate_col_from_specs(
    column_type: str,
    n_samples: int,
    col_specs: Dict,
    sequence: str,
    tokenizer: Optional[Any] = None,
    model: Optional[Any] = None,
) -> Iterable:
    """Generate a column of data.

    Args:
        column_type (str): Type of column to generate. Either uniform_int, text, or datetime_uniform.
        n_samples (int): Number of rows to generate.
        col_specs (Dict): A dict representing each column. Key is col_name (str), values is a dict with column_type (str), min (int) and max(int).
        sequence (str): Text prompt to use for generating text data. Defaults to "The quick brown fox jumps over the lazy dog".
        tokenizer (Optional[Any]): Huggingface tokenizer.
        model (Optional[Any]): Huggingface model.

    Raises:
        ValueError: If column_type isn't either uniform_int, text, or datetime_uniform.

    Returns:
        Iterable: The generated column.
    """

    if column_type == "text":
        generated_texts = generate_text_data(
            n_samples=n_samples, sequence=sequence, tokenizer=tokenizer, model=model
        )

        return generated_texts

    elif column_type == "uniform_int":
        return np.random.randint(
            low=col_specs["min"],
            high=col_specs["max"],
            size=n_samples,
        )
    elif column_type == "uniform_float":
        return np.random.uniform(
            low=col_specs["min"],
            high=col_specs["max"],
            size=n_samples,
        )
    elif column_type == "normal":
        return np.random.normal(
            loc=col_specs["mean"],
            scale=col_specs["sd"],
            size=n_samples,
        )
    elif column_type == "datetime_uniform":
        return pd.to_datetime(
            np.random.uniform(
                low=col_specs["min"],
                high=col_specs["max"],
                size=n_samples,
            ),
            unit="D",
        ).round("min")
    else:
        raise ValueError(f"Unknown distribution: {column_type}")


def generate_text_data(
    n_samples: int,
    sequence: str,
    tokenizer: Optional[Any] = None,
    model: Optional[Any] = None,
) -> List[str]:
    """
    Generate text data.

    Args:
        n_samples (int): Number of rows to generate.
        sequence (str): Text prompt to use for generating text data. Defaults to "The quick brown fox jumps over the lazy dog".
        tokenizer (Optional[Any]): Huggingface tokenizer
        model (Optional[Any]): Huggingface model

    Returns:
        List[str]: List of generated text data.
    """

    tokenizer = (
        GPT2Tokenizer.from_pretrained("gpt2") if tokenizer is None else tokenizer
    )
    model = GPT2LMHeadModel.from_pretrained("gpt2") if model is None else model

    inputs = tokenizer.encode(sequence, return_tensors="pt")

    generated_texts = []
    for _ in range(n_samples):
        max_tokens = np.random.randint(
            low=0,
            high=500,
            size=1,
        )[0]

        outputs = model.generate(
            inputs,
            min_length=0,
            max_length=max_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(text)

    return generated_texts

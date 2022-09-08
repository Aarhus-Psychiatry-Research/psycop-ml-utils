"""Script for creating synthetic text data for testing purposes.

Produces a .csv file with the following columns: citizen_id, timestamp,
text.
"""
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate_synth_data(
    predictors: Dict,
    n_samples: int,
    text_prompt: str = "The quick brown fox jumps over the lazy dog",
    na_prob: Optional[float] = 0.1,
    na_ignore_cols: List[str] = [],
) -> pd.DataFrame:
    """Takes a dict and generates synth data from it.

    Args:
        predictors (Dict): A dict representing each column. Key is col_name (str), values are column_type (str), output_type (float|int), min (int), max(int).
        n_samples (int): Number of samples (rows) to generate.
        text_prompt (str): Text prompt to use for generating text data. Defaults to "The quick brown fox jumps over the lazy dog".
        na_prob (float): Probability of changing a value in a predictor column to NA.
        na_ignore_cols (List[str]): Columns to ignore when creating NAs

    Returns:
        pd.DataFrame: The synthetic dataset
    """

    # Initialise dataframe
    df = pd.DataFrame(columns=list(predictors.keys()))

    # Generate data
    df = generate_data_columns(predictors, n_samples, df, text_prompt)

    # randomly replace predictors with NAs
    if na_prob:
        mask = np.random.choice([True, False], size=df.shape, p=[na_prob, 1 - na_prob])
        df_ = df.mask(mask)

        # For all columns in df.columns if column is not in na_ignore_cols
        for col in df.columns:
            if col not in na_ignore_cols:
                df[col] = df_[col]

    return df


def generate_data_columns(
    predictors: Iterable[Dict],
    n_samples: int,
    df: pd.DataFrame,
    text_prompt: str ,
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

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    sequence = text_prompt
    for col_name, col_props in predictors.items():
        # np.nan objects turn into "nan" strings in the real life dataframe.
        # imitate this in the synthetic data as well.
        if "nan" in col_name:
            df = df.rename({col_name: col_name.replace("np.nan", "nan")}, axis=1)
            col_name = col_name.replace("np.nan", "nan")

        column_type = col_props["column_type"]

        if column_type == "text":
            inputs = tokenizer.encode(sequence, return_tensors="pt")

            generated_texts = []
            for row in range(n_samples):
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

            df[col_name] = generated_texts

        elif column_type == "uniform_int":
            df[col_name] = np.random.randint(
                low=col_props["min"],
                high=col_props["max"],
                size=n_samples,
            )
        elif column_type == "datetime_uniform":
            df[col_name] = pd.to_datetime(
                np.random.uniform(
                    low=col_props["min"],
                    high=col_props["max"],
                    size=n_samples,
                ),
                unit="D",
            ).round("min")
        else:
            raise ValueError(f"Unknown distribution: {column_type}")

        # If column has min and/or max, floor and ceil appropriately
        if df[col_name].dtype not in ["datetime64[ns]"]:
            if "min" in col_props:
                df[col_name] = df[col_name].clip(lower=col_props["min"])
            if "max" in col_props:
                df[col_name] = df[col_name].clip(upper=col_props["max"])

    return df


if __name__ == "__main__":
    column_specifications = {
        "citizen_ids": {"column_type": "uniform_int", "min": 0, "max": 1_200_000},
        "timestamp": {"column_type": "datetime_uniform", "min": 0, "max": 5 * 365},
        "text": {"column_type": "text"},
    }

    df = generate_synth_data(
        predictors=column_specifications,
        n_samples=100,
        text_prompt="The patient",
    )

    save_path = Path(__file__).parent.parent.parent.parent
    df.to_csv(save_path / "tests" / "test_data" / "synth_txt_data.csv")

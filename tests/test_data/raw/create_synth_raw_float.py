"""Generate raw float dataframe."""

from pathlib import Path

from psycopmlutils.synth_data_generator.synth_col_generators import (
    generate_data_columns,
)

if __name__ == "__main__":
    # Get project root directory
    project_root = Path(__file__).resolve().parents[3]

    column_specs = {
        "dw_ek_borger": {
            "column_type": "id",
        },
        "raw_predictor": {"column_type": "uniform_float", "min": 0, "max": 10},
    }

    df = generate_data_columns(
        predictors=column_specs,
        n_samples=10_000,
    )

    df.to_csv(project_root / "tests" / "test_data" / "synth_raw.csv", index=False)

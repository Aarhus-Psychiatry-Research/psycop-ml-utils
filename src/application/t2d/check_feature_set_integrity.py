from pathlib import Path

from psycopmlutils.timeseriesflattener.data_integrity import (
    check_feature_set_integrity_from_dir,
)

if __name__ == "__main__":
    subdir = Path(
        "E:/shared_resources/feature_sets/t2d/adminmanber_260_features_2022_08_26_14_10/",
    )

    check_feature_set_integrity_from_dir(path=subdir, splits=["train", "val", "test"])

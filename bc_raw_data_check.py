from psycopmlutils.data_checks.validate_raw_data import validate_raw_data
from psycopmlutils.utils import data_loaders

if __name__ == "__main__":
    all_loaders = data_loaders.get_all()
    print(all_loaders)
    for feature_set_name, loader in all_loaders.items():
        print(f"Validating {feature_set_name}")
        df = loader()
        validate_raw_data(df, feature_set_name)

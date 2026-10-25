
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

FEATURE_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
]

TARGET_COLUMN = "income"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the official Adult Income training and test files."""

    train_path = RAW_DATA_DIR / "adult.data"
    test_path = RAW_DATA_DIR / "adult.test"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Dataset files not found in {RAW_DATA_DIR}. "
            "Check that adult.data and adult.test are in data/raw/."
        )

    train_df = pd.read_csv(
        train_path,
        header=None,
        names=ALL_COLUMNS,
        skipinitialspace=True,
        na_values="?",
    )

    test_df = pd.read_csv(
        test_path,
        header=None,
        names=ALL_COLUMNS,
        skipinitialspace=True,
        na_values="?",
        comment="|",
    )

    test_df[TARGET_COLUMN] = (
        test_df[TARGET_COLUMN].str.strip().str.rstrip(".")
    )

    return train_df, test_df

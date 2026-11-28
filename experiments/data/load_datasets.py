from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv_dataset(
    file_path,
    target_column,
    test_size=0.25,
    random_state=42,
):
    """
    Φορτώνει ένα CSV dataset και δημιουργεί
    training και test sets.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    if df[target_column].isna().any():
        raise ValueError(
            "The target column contains missing values."
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

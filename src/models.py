import os
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor


def load_feature_data(path: str = "data/processed/features.csv") -> pd.DataFrame:
    """Load feature dataset."""
    df = pd.read_csv(path, parse_dates=["week"])
    return df


def train_test_split_time(
    df: pd.DataFrame,
    test_size_weeks: int = 16,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simple time-based train-test split on the 'week' column."""
    df = df.sort_values("week")
    unique_weeks: List[pd.Timestamp] = sorted(df["week"].unique())
    split_point = unique_weeks[-test_size_weeks]

    train_df = df[df["week"] < split_point].copy()
    test_df = df[df["week"] >= split_point].copy()
    return train_df, test_df


def build_pipeline(
    categorical_cols: list,
    numeric_cols: list,
) -> Pipeline:
    """Build a preprocessing + LightGBM pipeline."""
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipe


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute simple regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)  # returns MSE
    rmse = np.sqrt(mse)                       # manually take sqrt
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}



def main():
    df = load_feature_data()

    # Define target and feature columns
    target_col = "sales_units"

    # Columns not to use as features
    drop_cols = [
        target_col,
        "week",
    ]

    categorical_cols = ["partner_id", "sku_id", "category"]
    numeric_cols = [
        col
        for col in df.columns
        if col not in drop_cols + categorical_cols
    ]

    train_df, test_df = train_test_split_time(df, test_size_weeks=16)

    X_train = train_df[categorical_cols + numeric_cols]
    y_train = train_df[target_col]

    X_test = test_df[categorical_cols + numeric_cols]
    y_test = test_df[target_col]

    pipeline = build_pipeline(
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = evaluate_predictions(y_test.values, y_pred)

    print("Test metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    # Save test predictions for later inspection
    test_df = test_df.copy()
    test_df["pred_sales_units"] = y_pred

    os.makedirs("data/processed", exist_ok=True)
    test_df.to_csv("data/processed/test_predictions.csv", index=False)
    print("Saved test predictions to data/processed/test_predictions.csv")


if __name__ == "__main__":
    main()

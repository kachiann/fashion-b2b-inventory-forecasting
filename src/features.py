import os
import pandas as pd


def load_sales_data(path: str = "data/raw/sales.csv") -> pd.DataFrame:
    """Load raw weekly sales data."""
    df = pd.read_csv(path, parse_dates=["week"])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based time features."""
    df = df.copy()
    df["weekofyear"] = df["week"].dt.isocalendar().week.astype(int)
    df["year"] = df["week"].dt.year.astype(int)
    return df


def add_lag_features(
    df: pd.DataFrame,
    group_cols=("partner_id", "sku_id"),
    target_col: str = "sales_units",
    lags=(1, 2, 3, 4),
) -> pd.DataFrame:
    """Add lag features for the target column."""
    df = df.copy()
    df = df.sort_values(list(group_cols) + ["week"])

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = (
            df.groupby(list(group_cols))[target_col]
            .shift(lag)
        )

    return df


def add_rolling_features(
    df: pd.DataFrame,
    group_cols=("partner_id", "sku_id"),
    target_col: str = "sales_units",
    windows=(4, 8),
) -> pd.DataFrame:
    """Add simple rolling mean features for the target column."""
    df = df.copy()
    df = df.sort_values(list(group_cols) + ["week"])

    for window in windows:
        df[f"{target_col}_rollmean_{window}"] = (
            df.groupby(list(group_cols))[target_col]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=list(range(len(group_cols))), drop=True)
        )

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df_feat = add_time_features(df)
    df_feat = add_lag_features(df_feat)
    df_feat = add_rolling_features(df_feat)
    # Drop rows where initial lags are NaN to keep training clean
    df_feat = df_feat.dropna().reset_index(drop=True)
    return df_feat


def save_features(df: pd.DataFrame, path: str = "data/processed/features.csv") -> None:
    """Save feature dataset to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def main():
    sales_df = load_sales_data()
    feat_df = build_features(sales_df)
    save_features(feat_df)
    print(f"Saved features with shape {feat_df.shape} to data/processed/features.csv")


if __name__ == "__main__":
    main()

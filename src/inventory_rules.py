import os
import numpy as np
import pandas as pd


def load_sales_data(path: str = "data/raw/sales.csv") -> pd.DataFrame:
    """Load raw weekly sales data."""
    return pd.read_csv(path, parse_dates=["week"])


def compute_demand_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average demand and std per partner-SKU."""
    grouped = (
        df.groupby(["partner_id", "sku_id"], as_index=False)
        .agg({"sales_units": ["mean", "std"]})
    )

    # Flatten multi-index columns and rename
    grouped.columns = ["partner_id", "sku_id", "avg_weekly_demand", "std_weekly_demand"]

    # Replace NaN std (constant demand) with 0
    grouped["std_weekly_demand"] = grouped["std_weekly_demand"].fillna(0.0)
    return grouped


def get_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    """Extract unique lead_time_weeks per partner-SKU."""
    lt = (
        df.groupby(["partner_id", "sku_id"], as_index=False)["lead_time_weeks"]
        .first()
    )
    return lt


def compute_inventory_policy(
    stats_df: pd.DataFrame,
    lead_time_df: pd.DataFrame,
    z: float = 1.65,
) -> pd.DataFrame:
    """
    Compute safety stock, reorder point, and a simple current stock & order qty.

    Formulas (classic ROP logic):
    - safety_stock = z * std_weekly_demand * sqrt(lead_time_weeks)
    - reorder_point = avg_weekly_demand * lead_time_weeks + safety_stock
    - current_stock = 2 * avg_weekly_demand   (simple assumption for demo)
    - suggested_order_qty = max(reorder_point - current_stock, 0)
    """
    df = stats_df.merge(lead_time_df, on=["partner_id", "sku_id"], how="left")

    # Safety stock
    df["safety_stock"] = (
        z * df["std_weekly_demand"] * np.sqrt(df["lead_time_weeks"])
    )

    # Reorder point
    df["reorder_point"] = (
        df["avg_weekly_demand"] * df["lead_time_weeks"] + df["safety_stock"]
    )

    # Simple current stock assumption
    df["current_stock"] = 2.0 * df["avg_weekly_demand"]

    # Suggested order quantity
    df["suggested_order_qty"] = np.maximum(
        df["reorder_point"] - df["current_stock"], 0.0
    )

    return df


def save_policies(
    df: pd.DataFrame,
    path: str = "data/processed/recommendations.csv",
) -> None:
    """Save inventory recommendations."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def main():
    sales_df = load_sales_data()
    stats_df = compute_demand_stats(sales_df)
    lt_df = get_lead_time(sales_df)

    policy_df = compute_inventory_policy(stats_df, lt_df)
    save_policies(policy_df)

    print(
        f"Saved inventory recommendations for {len(policy_df)} partner-SKU pairs "
        f"to data/processed/recommendations.csv"
    )


if __name__ == "__main__":
    main()

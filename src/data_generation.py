import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def create_date_range(start_date: str, n_weeks: int) -> list:
    """Create a list of weekly dates starting from start_date."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    return [start + timedelta(weeks=i) for i in range(n_weeks)]


def generate_sku_table(n_skus: int, random_state: int = 42) -> pd.DataFrame:
    """Generate a table of SKUs with category and price."""
    rng = np.random.default_rng(random_state)
    categories = ["tops", "shoes", "dresses", "outerwear"]
    sku_ids = [f"S{str(i).zfill(3)}" for i in range(1, n_skus + 1)]
    sku_categories = rng.choice(categories, size=n_skus)
    base_prices = rng.integers(20, 150, size=n_skus)

    return pd.DataFrame(
        {
            "sku_id": sku_ids,
            "category": sku_categories,
            "price": base_prices,
        }
    )


def generate_partner_table(n_partners: int) -> pd.DataFrame:
    """Generate a table of partner IDs."""
    partner_ids = [f"P{str(i)}" for i in range(1, n_partners + 1)]
    return pd.DataFrame({"partner_id": partner_ids})


def generate_sales_data(
    n_weeks: int = 104,
    n_partners: int = 5,
    n_skus: int = 50,
    start_date: str = "2022-01-03",
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic weekly sales data for partner-SKU combinations."""
    rng = np.random.default_rng(random_state)

    dates = create_date_range(start_date, n_weeks)
    sku_df = generate_sku_table(n_skus, random_state=random_state)
    partner_df = generate_partner_table(n_partners)

    # Cartesian product of partners and SKUs
    cross = partner_df.merge(sku_df, how="cross")

    # Assign a lead time per partner-SKU (1–4 weeks)
    cross["lead_time_weeks"] = rng.integers(1, 5, size=len(cross))

    records = []

    for week_idx, week_date in enumerate(dates):
        # Seasonality factor: simple yearly seasonality using sine
        seasonal = 1.0 + 0.3 * np.sin(2 * np.pi * week_idx / 52)

        # Base demand per SKU (1–40 units), scaled by category
        base_demand = rng.integers(1, 40, size=len(cross))
        category_factor = cross["category"].map(
            {"tops": 1.0, "shoes": 1.2, "dresses": 1.1, "outerwear": 0.9}
        ).values

        mean_demand = base_demand * seasonal * category_factor

        # Add noise and ensure non-negative integers
        noise = rng.normal(loc=0.0, scale=5.0, size=len(cross))
        week_demand = np.maximum(mean_demand + noise, 0).round().astype(int)

        tmp = cross.copy()
        tmp["week"] = week_date
        tmp["sales_units"] = week_demand

        records.append(tmp)

    df = pd.concat(records, ignore_index=True)
    return df


def save_sales_data(df: pd.DataFrame, path: str = "data/raw/sales.csv") -> None:
    """Save the sales DataFrame to CSV, creating directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def main():
    df = generate_sales_data()
    save_sales_data(df)
    print(f"Generated dataset with {len(df)} rows and saved to data/raw/sales.csv")


if __name__ == "__main__":
    main()

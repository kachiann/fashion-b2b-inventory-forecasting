import pandas as pd
import streamlit as st


@st.cache_data
def load_data():
    sales = pd.read_csv("data/raw/sales.csv", parse_dates=["week"])
    recs = pd.read_csv("data/processed/recommendations.csv")
    preds = pd.read_csv("data/processed/test_predictions.csv", parse_dates=["week"])
    return sales, recs, preds


def main():
    st.title("Fashion B2B Inventory Forecasting Demo")

    sales, recs, preds = load_data()

    partners = sorted(sales["partner_id"].unique())
    partner_id = st.selectbox("Select partner", partners)

    skus = sorted(sales.loc[sales["partner_id"] == partner_id, "sku_id"].unique())
    sku_id = st.selectbox("Select SKU", skus)

    # Time series: actual vs predicted (test period)
    mask_pred = (preds["partner_id"] == partner_id) & (preds["sku_id"] == sku_id)
    df_pred = preds.loc[mask_pred].sort_values("week")

    st.subheader("Actual vs predicted weekly demand (test period)")
    if df_pred.empty:
        st.info("No predictions for this partner–SKU in the test window.")
    else:
        st.line_chart(
            df_pred.set_index("week")[["sales_units", "pred_sales_units"]]
        )

    # Inventory recommendation
    mask_rec = (recs["partner_id"] == partner_id) & (recs["sku_id"] == sku_id)
    df_rec = recs.loc[mask_rec]

    st.subheader("Inventory policy recommendation")
    if df_rec.empty:
        st.info("No recommendation for this partner–SKU.")
    else:
        cols = [
            "avg_weekly_demand",
            "std_weekly_demand",
            "lead_time_weeks",
            "safety_stock",
            "reorder_point",
            "current_stock",
            "suggested_order_qty",
        ]
        st.dataframe(df_rec[cols].round(2))


if __name__ == "__main__":
    main()

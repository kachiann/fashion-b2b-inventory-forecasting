# Fashion b2b Inventory forecasting

## Overview
This is a small end-to-end project built to play with demand forecasting and inventory planning.

- I generate a synthetic weekly sales dataset for multiple partners and SKUs (fashion items like tops, shoes, and dresses).
- I train a basic ML model to predict short-term demand for each partner–SKU combination.
- I use simple safety-stock and reorder-point logic to turn those forecasts into replenishment suggestions.
- I include a lightweight Streamlit app to explore historical sales, view forecasts, and see recommended reorder quantities.

## Environment
- Python 3.9+ is recommended

Next, install dependencies with:
`pip install -r requirements.txt`



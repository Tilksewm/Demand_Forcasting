import pandas as pd
import numpy as np
import joblib

from config import (FEATURES_BASE, TARGET, MIN_HISTORY_DAYS)
from feature_engineering import (
    add_sku_rolling_stats,
    add_time_aware_category_te
)
from config import (DATA_PROCESSED, MODELS_DIR)


model = joblib.load(MODELS_DIR / "global_demand_model.joblib")

train_df = pd.read_csv(DATA_PROCESSED / "train_data.csv", parse_dates=["date"])
test_df  = pd.read_csv(DATA_PROCESSED / "test_data.csv",  parse_dates=["date"])

# Filter SKUs consistently
valid_skus = (
    train_df.groupby("sku_id").size()
    .loc[lambda x: x >= MIN_HISTORY_DAYS]
    .index
)

train_df = train_df[train_df["sku_id"].isin(valid_skus)]
test_df  = test_df[test_df["sku_id"].isin(valid_skus)]

# Feature engineering (same logic as training)
train_df = add_sku_rolling_stats(train_df)
test_df  = add_sku_rolling_stats(test_df)

train_df = add_time_aware_category_te(train_df, TARGET)
test_df  = add_time_aware_category_te(test_df, TARGET)

FEATURES = FEATURES_BASE + [
    "sku_avg_28d",
    "sku_std_28d",
    "category_te",
]

test_df = test_df.dropna(subset=FEATURES + [TARGET])

test_df["daily_pred"] = model.predict(test_df[FEATURES]).clip(0)
test_df["year_month"] = test_df["date"].dt.to_period("M")

monthly_preds = (
    test_df
    .groupby(["sku_id", "year_month"])["daily_pred"]
    .sum()
    .reset_index()
    .rename(columns={"daily_pred": "predicted_monthly_qty"})
)

# Comparing with actual pridiction
actual_monthly = (
    test_df
    .groupby(["sku_id", "year_month"])["target_qty"]
    .sum()
    .reset_index()
    .rename(columns={"target_qty": "actual_monthly_qty"})
)

monthly_results = monthly_preds.merge(
    actual_monthly,
    on=["sku_id", "year_month"],
    how="left"
)

monthly_results["abs_error"] = (
    monthly_results["predicted_monthly_qty"]
    - monthly_results["actual_monthly_qty"]
).abs()

monthly_results["pct_error"] = np.where(
    monthly_results["actual_monthly_qty"] > 0,
    monthly_results["abs_error"] / monthly_results["actual_monthly_qty"] * 100,
    0
)
monthly_results.to_csv(
    DATA_PROCESSED / "global_monthly_predictions_per_sku.csv",
    index=False
)

print("Monthly per-SKU predictions generated.")
print(monthly_results.head())
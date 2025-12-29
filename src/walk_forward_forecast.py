import pandas as pd
import numpy as np
import joblib

from src.config import (FEATURES_BASE, TARGET, MIN_HISTORY_DAYS, DATA_PROCESSED, MODELS_DIR)
from src.feature_engineering import (
    add_sku_rolling_stats,
    add_time_aware_category_te
)

def walk_forward_forecast():
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
        DATA_PROCESSED / "walk_forward_monthly_vs_actual.csv",
        index=False
    )

    print("Monthly per-SKU predictions generated.")
    # print(monthly_results.head())
    # Daily metrics
    daily_actual = test_df[TARGET]
    daily_pred = test_df["daily_pred"]
    daily_abs = (daily_pred - daily_actual).abs()

    daily_mae = daily_abs.mean()
    daily_mape = float(np.nanmean(np.where(daily_actual > 0, daily_abs / daily_actual * 100, np.nan))) if (daily_actual > 0).any() else 0.0
    daily_wape = (daily_abs.sum() / daily_actual.abs().sum() * 100) if daily_actual.abs().sum() > 0 else 0.0

    print("===== DAILY WALK_FORWARD vs ACTUAL =====")
    print(f"Daily MAE: {daily_mae:.4f}")
    print(f"Daily MAPE: {daily_mape:.2f}%")
    print(f"Daily WAPE: {daily_wape:.2f}%")

    # Monthly metrics
    monthly_abs = monthly_results["abs_error"]
    monthly_actual = monthly_results["actual_monthly_qty"]

    print("===== MONTHLY WALK_FORWARD vs ACTUAL =====")
    monthly_mae = monthly_abs.mean()
    monthly_mape = float(np.nanmean(np.where(monthly_actual > 0, monthly_abs / monthly_actual * 100, np.nan))) if (monthly_actual > 0).any() else 0.0
    monthly_wape = (monthly_abs.sum() / monthly_actual.abs().sum() * 100) if monthly_actual.abs().sum() > 0 else 0.0

    print(f"Monthly MAE: {monthly_mae:.4f}")
    print(f"Monthly MAPE: {monthly_mape:.2f}%")
    print(f"Monthly WAPE: {monthly_wape:.2f}%")

if __name__ == "__main__":
    walk_forward_forecast()
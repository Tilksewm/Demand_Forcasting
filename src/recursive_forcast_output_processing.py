import pandas as pd
import numpy as np
from src.config import DATA_PROCESSED
def process_recursive_forecast_output(model_type="hist_gbr"):
    # ----------------------------
    # PATHS
    # ----------------------------
    TEST_PATH = DATA_PROCESSED / "test_data.csv"
    if model_type == "linear":
        RECURSIVE_DAILY_PATH = DATA_PROCESSED / "linear_recursive_daily_forecast.csv"
    else:
        RECURSIVE_DAILY_PATH = DATA_PROCESSED / "recursive_daily_forecast.csv"

    # ----------------------------
    # LOAD
    # ----------------------------
    test_df = pd.read_csv(TEST_PATH, parse_dates=["date"])
    recursive_daily = pd.read_csv(RECURSIVE_DAILY_PATH, parse_dates=["date"])

    # Keep only what we need
    test_daily = test_df[["sku_id", "date", "target_qty"]].copy()

    # ----------------------------
    # MERGE
    # ----------------------------
    daily_compare = (
        recursive_daily
        .merge(
            test_daily,
            on=["sku_id", "date"],
            how="inner"
        )
    )

    # ----------------------------
    # ERRORS
    # ----------------------------
    daily_compare["abs_error"] = np.abs(
        daily_compare["target_qty"] - daily_compare["predicted_qty"]
    )

    daily_compare["pct_error"] = (
        daily_compare["abs_error"] /
        np.maximum(daily_compare["target_qty"], 1)
    ) * 100

    # ----------------------------
    # METRICS
    # ----------------------------
    daily_mae = daily_compare["abs_error"].mean()
    daily_mape = daily_compare["pct_error"].mean()
    daily_wape = (
        daily_compare["abs_error"].sum() /
        daily_compare["target_qty"].sum()
    ) * 100

    print("===== DAILY RECURSIVE vs ACTUAL =====")
    print(f"MAE  : {daily_mae:.2f}")
    print(f"MAPE : {daily_mape:.2f}%")
    print(f"WAPE : {daily_wape:.2f}%")


    # ----------------------------
    # MONTHLY ACTUAL
    # ----------------------------
    test_df["year_month"] = test_df["date"].dt.to_period("M")

    actual_monthly = (
        test_df
        .groupby(["sku_id", "year_month"])
        .agg(actual_qty=("target_qty", "sum"))
        .reset_index()
    )

    # ----------------------------
    # LOAD RECURSIVE MONTHLY
    # ----------------------------
    if model_type == "linear":
        recursive_monthly = pd.read_csv(
            DATA_PROCESSED / "linear_recursive_monthly_forecast.csv"
        )
    else:
        recursive_monthly = pd.read_csv(
            DATA_PROCESSED / "recursive_monthly_forecast.csv"
        )

    recursive_monthly["year_month"] = recursive_monthly["year_month"].astype("period[M]")
    # ----------------------------
    # MERGE
    # ----------------------------
    monthly_compare = (
        recursive_monthly
        .merge(
            actual_monthly,
            on=["sku_id", "year_month"],
            how="inner"
        )
    )

    # ----------------------------
    # ERRORS
    # ----------------------------
    monthly_compare["abs_error"] = np.abs(
        monthly_compare["actual_qty"] - monthly_compare["predicted_qty"]
    )

    monthly_compare["pct_error"] = (
        monthly_compare["abs_error"] /
        np.maximum(monthly_compare["actual_qty"], 1)
    ) * 100

    # ----------------------------
    # METRICS
    # ----------------------------
    monthly_mae = monthly_compare["abs_error"].mean()
    monthly_mape = monthly_compare["pct_error"].mean()
    monthly_wape = (
        monthly_compare["abs_error"].sum() /
        monthly_compare["actual_qty"].sum()
    ) * 100

    print("\n===== MONTHLY RECURSIVE vs ACTUAL =====")
    print(f"MAE  : {monthly_mae:.2f}")
    print(f"MAPE : {monthly_mape:.2f}%")
    print(f"WAPE : {monthly_wape:.2f}%")

    # ----------------------------
    # SAVE
    # ----------------------------
    monthly_compare = monthly_compare.rename(columns={
        "predicted_qty": "predicted_monthly_qty",
        "actual_qty": "actual_monthly_qty"
    })
    if model_type == "linear":
        daily_compare.to_csv(
            DATA_PROCESSED / "linear_recursive_daily_vs_actual.csv",
            index=False
        )
        monthly_compare.to_csv(
            DATA_PROCESSED / "linear_recursive_monthly_vs_actual.csv",
            index=False
        )
        print("Saved: linear_recursive_daily_vs_actual.csv and linear_recursive_monthly_vs_actual.csv")
    else:
        daily_compare.to_csv(
            DATA_PROCESSED / "recursive_daily_vs_actual.csv",
            index=False
        )
        monthly_compare.to_csv(
            DATA_PROCESSED / "recursive_monthly_vs_actual.csv",
            index=False
        )
        print("Saved: recursive_daily_vs_actual.csv and recursive_monthly_vs_actual.csv")
if __name__ == "__main__":
    process_recursive_forecast_output()
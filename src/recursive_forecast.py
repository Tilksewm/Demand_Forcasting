import pandas as pd
import numpy as np
import joblib
from tqdm.auto import tqdm
import os
from src.config import (DATA_PROCESSED, MODELS_DIR, DATA_RAW)
from src.feature_engineering import (
    add_sku_rolling_stats,
    add_time_aware_category_te
)

# ----------------------------
# Re-load train_df for consistency with global model feature engineering
# ----------------------------
def recursive_forecast(FORECAST_DAYS=30, model_type="hist_gbr"):
    train_df_for_features = pd.read_csv(DATA_PROCESSED / "train_data.csv",
        parse_dates=["date"]
    )
    train_df_for_features = train_df_for_features.sort_values(["sku_id", "date"])

    # ----------------------------
    # CONFIG
    # ----------------------------
    TARGET = "target_qty"
    MIN_HISTORY_DAYS = 90 # Minimum history days to consider SKU for forecasting

    # ----------------------------
    # Model and Data Paths
    # ----------------------------
    if model_type == "linear":
        model = joblib.load(
            MODELS_DIR / "linear_regression.joblib"
        )
        print("Using Linear Regression model for recursive forecasting.")
    else:
        model = joblib.load(
            MODELS_DIR / "global_demand_model.joblib"
        )
        print("Using Histogram-based GBR model for recursive forecasting.")

    # Re-load daily to ensure clean slate for feature engineering
    daily = pd.read_csv(
        DATA_PROCESSED / "train_data.csv", 
        parse_dates=["date"]
    )
    daily = daily.sort_values(["sku_id", "date"])
    daily = add_sku_rolling_stats(daily, 28)
    daily = add_time_aware_category_te(daily, TARGET)

    # ----------------------------
    # Helper functions for holiday/fasting
    # Fasting
    fasting = pd.read_csv(DATA_RAW / "fasting_periods.csv")
    fasting["start_date"] = pd.to_datetime(fasting["start_date"])
    fasting["end_date"] = pd.to_datetime(fasting["end_date"])

    # Holidays
    events = pd.read_csv(DATA_RAW / "ethiopian_events.csv")
    events["date"] = pd.to_datetime(events["date"])

    holiday_dates = events[events["is_holiday"] == 1]["date"].sort_values()

    def is_fasting_day(d):
        return int(((fasting["start_date"] <= d) & (fasting["end_date"] >= d)).any())

    def is_holiday_day(d):
        return int(d in set(holiday_dates))

    def days_to_next_holiday(d):
        future = holiday_dates[holiday_dates >= d]
        return (future.iloc[0] - d).days if len(future) else np.nan

    def days_since_last_holiday(d):
        past = holiday_dates[holiday_dates <= d]
        return (d - past.iloc[-1]).days if len(past) else np.nan


    # FINAL FEATURES LIST 

    FEATURES_BASE = [
        "day_of_week",
        "is_weekend",
        "day_of_year",
        "qty_lag_1",
        "qty_lag_7",
        "qty_lag_14",
        "qty_roll_7",
        "qty_roll_14",
        "is_holiday",
        "days_to_holiday",
        "days_since_holiday",
        "is_fasting"
    ]

    FEATURES = FEATURES_BASE + [
        "sku_avg_28d",
        "sku_std_28d",
        "category_te",
    ]

    # ----------------------------
    # DROP NA (Only after features ready) - Essential for features involving rolling means/stds
    # ----------------------------
    daily = daily.dropna(subset=FEATURES + [TARGET])

    # Start predictions

    all_daily_preds = []

    for sku in tqdm(daily["sku_id"].unique(), desc="Recursive Forcast"):
        sku_hist = daily[daily["sku_id"] == sku].copy()
        sku_hist = sku_hist.sort_values("date")

        if len(sku_hist) < 30: # Keeping original 30 from the cell
            continue

        history_qty = sku_hist[TARGET].tolist()
        last_date = sku_hist["date"].max()
        sku_category_te = sku_hist["category_te"].iloc[-1] # category_te is constant for a SKU

        for step in range(FORECAST_DAYS):
            next_date = last_date + pd.Timedelta(days=1)

            row = {
                "day_of_week": next_date.weekday(),
                "is_weekend": int(next_date.weekday() >= 5),
                "day_of_year": next_date.dayofyear,
                "is_holiday": is_holiday_day(next_date),
                "days_to_holiday": days_to_next_holiday(next_date),
                "days_since_holiday": days_since_last_holiday(next_date),
                "is_fasting": is_fasting_day(next_date),
                "category_te": sku_category_te # Add category_te to the row
            }

            # Lag & rolling features - based on history_qty (actuals + predictions)
            row["qty_lag_1"] = history_qty[-1]
            row["qty_lag_7"] = history_qty[-7] if len(history_qty) >= 7 else history_qty[-1]
            row["qty_lag_14"] = history_qty[-14] if len(history_qty) >= 14 else history_qty[-1]

            row["qty_roll_7"] = np.mean(history_qty[-7:])
            row["qty_roll_14"] = np.mean(history_qty[-14:])

            rolling_window_28 = 28

            if len(history_qty) >= rolling_window_28:
                window = history_qty[-rolling_window_28:]
            else:
                window = history_qty  # shorter history, same logic

            row["sku_avg_28d"] = np.mean(window)
            row["sku_std_28d"] = np.std(window, ddof=1)

            # Handle edge case for std = 0 or NaN
            if not np.isfinite(row["sku_std_28d"]) or row["sku_std_28d"] == 0:
                row["sku_std_28d"] = 1e-6


            X = pd.DataFrame([row])[FEATURES]

            y_pred = max(0, round(model.predict(X)[0]))

            all_daily_preds.append({
                "sku_id": sku,
                "date": next_date,
                "predicted_qty": y_pred
            })

            history_qty.append(y_pred)
            last_date = next_date

    # Aggregate to monthly
    daily_pred_df = pd.DataFrame(all_daily_preds)
    daily_pred_df["year_month"] = daily_pred_df["date"].dt.to_period("M")

    monthly_pred_df = (
        daily_pred_df
        .groupby(["sku_id", "year_month"])["predicted_qty"]
        .sum()
        .reset_index()
    )

    # Saving outputs
    if model_type == "linear":
        daily_pred_df.to_csv(
            DATA_PROCESSED / "linear_recursive_daily_forecast.csv",
            index=False
        )

        monthly_pred_df.to_csv(
            DATA_PROCESSED / "linear_recursive_monthly_forecast.csv",
            index=False
        )
        print("Linear Regression recursive forecasting completed.")
    else:
        daily_pred_df.to_csv(
            DATA_PROCESSED / "recursive_daily_forecast.csv",
            index=False
        )
        monthly_pred_df.to_csv(
            DATA_PROCESSED / "recursive_monthly_forecast.csv",
            index=False
        )
        print("Recursive forecasting completed.")
        
if __name__ == "__main__":
    recursive_forecast()
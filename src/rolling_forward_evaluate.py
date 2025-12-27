import pandas as pd
import numpy as np
import joblib
from tqdm.auto import tqdm
import os
from config import (DATA_PROCESSED, MODELS_DIR, DATA_RAW)
from feature_engineering import (
    add_sku_rolling_stats,
    add_time_aware_category_te
)

# ----------------------------
# Re-load train_df for consistency with global model feature engineering
# ----------------------------
train_df_for_features = pd.read_csv(DATA_PROCESSED / "train_data.csv",
    parse_dates=["date"]
)
train_df_for_features = train_df_for_features.sort_values(["sku_id", "date"])

# ----------------------------
# CONFIG
# ----------------------------
TARGET = "target_qty"
MIN_HISTORY_DAYS = 90 # Not directly used for filtering in this cell, but good to keep consistent

# ----------------------------
# Model and Data Paths
# ----------------------------

model = joblib.load(
    MODELS_DIR / "global_demand_model.joblib" # Ensure correct global model name
)

# Re-load daily to ensure clean slate for feature engineering
daily = pd.read_csv(
    DATA_PROCESSED / "train_data.csv", # Assuming daily is the training data to forecast from
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

# ----------------------------
# Feature Engineering for `daily` (consistent with Mv_jgUlwXkQp)
# ----------------------------
# # CATEGORY TARGET ENCODING
# category_mean = (
#     train_df_for_features # Use the re-loaded train_df for feature calculation
#     .groupby("category")[TARGET]
#     .mean()
# )
# daily["category_te"] = daily["category"].map(category_mean)

# global_mean = train_df_for_features[TARGET].mean()
# daily["category_te"] = daily["category_te"].fillna(global_mean)

# # SKU BEHAVIOR FEATURES (ROLLING)
# def add_sku_behavior_for_daily(df):
#     df = df.copy()
#     df["sku_avg_28d"] = (
#         df.groupby("sku_id")["daily_qty"]
#         .rolling(28)
#         .mean()
#         .reset_index(level=0, drop=True)
#     )
#     df["sku_std_28d"] = (
#         df.groupby("sku_id")["daily_qty"]
#         .rolling(28)
#         .std()
#         .reset_index(level=0, drop=True)
#     )
#     return df

# daily = add_sku_behavior_for_daily(daily)

# ----------------------------
# FINAL FEATURES LIST (Updated to include new features)
# ----------------------------
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

# Start prediction
FORECAST_DAYS = 90  # ~3 months

all_daily_preds = []

for sku in tqdm(daily["sku_id"].unique(), desc="Rolling forward"):
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

        # Calculate sku_avg_28d and sku_std_28d dynamically for forecast steps
        # rolling_window_28 = 28
        # if len(history_qty) >= rolling_window_28:
        #     row["sku_avg_28d"] = np.mean(history_qty[-rolling_window_28:])
        #     row["sku_std_28d"] = np.std(history_qty[-rolling_window_28:], ddof=1)
        # elif len(history_qty) > 0: # If history is present but less than rolling_window_28
        #     row["sku_avg_28d"] = np.mean(history_qty)
        #     row["sku_std_28d"] = np.std(history_qty, ddof=1)
        # else: # Fallback if history is completely empty (should not be reached if len(sku_hist) < 30 check is robust)
        #     row["sku_avg_28d"] = 0.0
        #     row["sku_std_28d"] = 0.0

        # # If std is 0 (e.g., all values are same), set to small epsilon to avoid potential issues if std is used as a denominator
        # if row["sku_std_28d"] == 0:
        #     row["sku_std_28d"] = 1e-6

        rolling_window_28 = 28

        if len(history_qty) >= rolling_window_28:
            window = history_qty[-rolling_window_28:]
        else:
            window = history_qty  # shorter history, same logic

        row["sku_avg_28d"] = np.mean(window)
        row["sku_std_28d"] = np.std(window, ddof=1)

        # Guardrail for zero or nan std
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
daily_pred_df.to_csv(
    DATA_PROCESSED / "rolling_daily_forecast.csv",
    index=False
)

monthly_pred_df.to_csv(
    DATA_PROCESSED / "rolling_monthly_forecast.csv",
    index=False
)

print("Rolling-forward forecasting completed.")
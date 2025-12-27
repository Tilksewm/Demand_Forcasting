import pandas as pd
import numpy as np
import joblib
from tqdm.auto import tqdm
from pathlib import Path
from config import DATA_PROCESSED, DATA_RAW, MODELS_DIR

# --- Constants ---
TARGET = "target_qty"
FORECAST_DAYS = 90
ROLLING_WINDOW = 28

# --- Load static external data ---
def load_static_data():
    fasting = pd.read_csv(DATA_RAW / "fasting_periods.csv")
    fasting["start_date"] = pd.to_datetime(fasting["start_date"])
    fasting["end_date"] = pd.to_datetime(fasting["end_date"])

    events = pd.read_csv(DATA_RAW / "ethiopian_events.csv")
    events["date"] = pd.to_datetime(events["date"])
    holiday_dates = events[events["is_holiday"] == 1]["date"].sort_values()

    return fasting, holiday_dates

# --- Holiday feature functions ---
def is_fasting_day(d, fasting_df):
    return int(((fasting_df["start_date"] <= d) & (fasting_df["end_date"] >= d)).any())

def is_holiday_day(d, holiday_dates):
    return int(d in set(holiday_dates))

def days_to_next_holiday(d, holiday_dates):
    future = holiday_dates[holiday_dates >= d]
    return (future.iloc[0] - d).days if len(future) else np.nan

def days_since_last_holiday(d, holiday_dates):
    past = holiday_dates[holiday_dates <= d]
    return (d - past.iloc[-1]).days if len(past) else np.nan

# --- Feature engineering helpers ---
def compute_category_te(train_df, target_col):
    cat_mean = train_df.groupby("category")[target_col].mean()
    global_mean = train_df[target_col].mean()
    return cat_mean, global_mean

def add_sku_behavior_features(df):
    df = df.copy()
    df["sku_avg_28d"] = df.groupby("sku_id")["daily_qty"].rolling(ROLLING_WINDOW).mean().reset_index(level=0, drop=True)
    df["sku_std_28d"] = df.groupby("sku_id")["daily_qty"].rolling(ROLLING_WINDOW).std().reset_index(level=0, drop=True)
    return df

# --- Main function ---
def rolling_forecast():
    print("Starting rolling forward forecast...")
    
    model = joblib.load(MODELS_DIR / "global_demand_model.joblib")
    train_df = pd.read_csv(DATA_PROCESSED / "train_data.csv", parse_dates=["date"]).sort_values(["sku_id", "date"])
    daily = train_df.copy()

    fasting_df, holiday_dates = load_static_data()
    cat_te, global_te = compute_category_te(train_df, TARGET)
    daily["category_te"] = daily["category"].map(cat_te).fillna(global_te)
    daily = add_sku_behavior_features(daily)
    
    FEATURES_BASE = [
        "day_of_week", "is_weekend", "day_of_year",
        "qty_lag_1", "qty_lag_7", "qty_lag_14",
        "qty_roll_7", "qty_roll_14",
        "is_holiday", "days_to_holiday", "days_since_holiday", "is_fasting"
    ]
    FEATURES = FEATURES_BASE + ["sku_avg_28d", "sku_std_28d", "category_te"]
    daily = daily.dropna(subset=FEATURES + [TARGET])

    all_preds = []

    for sku in tqdm(daily["sku_id"].unique(), desc="Rolling forward"):
        sku_hist = daily[daily["sku_id"] == sku].copy().sort_values("date")

        if len(sku_hist) < 30:
            continue

        history_qty = sku_hist[TARGET].tolist()
        last_date = sku_hist["date"].max()
        sku_category_te = sku_hist["category_te"].iloc[-1]

        for step in range(FORECAST_DAYS):
            next_date = last_date + pd.Timedelta(days=1)
            row = {
                "day_of_week": next_date.weekday(),
                "is_weekend": int(next_date.weekday() >= 5),
                "day_of_year": next_date.dayofyear,
                "is_holiday": is_holiday_day(next_date, holiday_dates),
                "days_to_holiday": days_to_next_holiday(next_date, holiday_dates),
                "days_since_holiday": days_since_last_holiday(next_date, holiday_dates),
                "is_fasting": is_fasting_day(next_date, fasting_df),
                "category_te": sku_category_te,
                "qty_lag_1": history_qty[-1],
                "qty_lag_7": history_qty[-7] if len(history_qty) >= 7 else history_qty[-1],
                "qty_lag_14": history_qty[-14] if len(history_qty) >= 14 else history_qty[-1],
                "qty_roll_7": np.mean(history_qty[-7:]),
                "qty_roll_14": np.mean(history_qty[-14:]),
            }

            if len(history_qty) >= ROLLING_WINDOW:
                row["sku_avg_28d"] = np.mean(history_qty[-ROLLING_WINDOW:])
                row["sku_std_28d"] = np.std(history_qty[-ROLLING_WINDOW:], ddof=1)
            else:
                row["sku_avg_28d"] = np.mean(history_qty)
                row["sku_std_28d"] = np.std(history_qty, ddof=1)

            if row["sku_std_28d"] == 0:
                row["sku_std_28d"] = 1e-6

            X = pd.DataFrame([row])[FEATURES]
            y_pred = max(0, round(model.predict(X)[0]))

            all_preds.append({
                "sku_id": sku,
                "date": next_date,
                "predicted_qty": y_pred
            })

            history_qty.append(y_pred)
            last_date = next_date

    # Save outputs
    daily_pred_df = pd.DataFrame(all_preds)
    daily_pred_df["year_month"] = daily_pred_df["date"].dt.to_period("M")

    daily_pred_df.to_csv(DATA_PROCESSED / "rolling_daily_forecast.csv", index=False)
    monthly_pred_df = (
        daily_pred_df
        .groupby(["sku_id", "year_month"])["predicted_qty"]
        .sum()
        .reset_index()
    )
    monthly_pred_df.to_csv(DATA_PROCESSED / "rolling_monthly_forecast.csv", index=False)

    print("Rolling forward forecast complete.")
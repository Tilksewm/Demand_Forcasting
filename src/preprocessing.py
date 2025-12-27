import pandas as pd
import numpy as np
from src.config import SALES_FILE, FASTING_FILE, EVENTS_FILE


def load_sales_data():
    df = pd.read_csv(SALES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


def aggregate_daily_sales(df):
    daily = (
        df.groupby(["date", "sku_id", "category"], as_index=False)
          .agg(daily_qty=("quantity", "sum"))
    )

    all_dates = pd.date_range(
        start=daily["date"].min(),
        end=daily["date"].max(),
        freq="D"
    )

    all_skus = daily[["sku_id", "category"]].drop_duplicates()

    grid = (
        all_skus.assign(key=1)
        .merge(pd.DataFrame({"date": all_dates, "key": 1}), on="key")
        .drop("key", axis=1)
    )

    daily = grid.merge(
        daily,
        on=["date", "sku_id", "category"],
        how="left"
    )

    daily["daily_qty"] = daily["daily_qty"].fillna(0)
    return daily


def add_calendar_features(df):
    df = df.copy()
    df["day_of_week"] = df["date"].dt.weekday
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] == 6).astype(int)
    return df


def add_lag_features(df):
    df = df.sort_values(["sku_id", "date"])

    for lag in [1, 7, 14]:
        df[f"qty_lag_{lag}"] = df.groupby("sku_id")["daily_qty"].shift(lag)

    for window in [7, 14]:
        df[f"qty_roll_{window}"] = (
            df.groupby("sku_id")["daily_qty"]
              .shift(1)
              .rolling(window)
              .mean()
        )

    return df


def add_fasting_feature(df):
    fasting = pd.read_csv(FASTING_FILE)
    fasting["start_date"] = pd.to_datetime(fasting["start_date"])
    fasting["end_date"] = pd.to_datetime(fasting["end_date"])

    df["is_fasting"] = 0

    for _, row in fasting.iterrows():
        mask = (df["date"] >= row["start_date"]) & (df["date"] <= row["end_date"])
        df.loc[mask, "is_fasting"] = 1

    return df


def add_holiday_features(df):
    events = pd.read_csv(EVENTS_FILE)
    events["date"] = pd.to_datetime(events["date"])

    df = df.merge(
        events[["date", "is_holiday"]],
        on="date",
        how="left"
    )

    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)

    holiday_dates = events.loc[events["is_holiday"] == 1, "date"].sort_values()

    def days_to_next(d):
        future = holiday_dates[holiday_dates >= d]
        return (future.iloc[0] - d).days if len(future) else np.nan

    def days_since_last(d):
        past = holiday_dates[holiday_dates <= d]
        return (d - past.iloc[-1]).days if len(past) else np.nan

    df["days_to_holiday"] = df["date"].apply(days_to_next)
    df["days_since_holiday"] = df["date"].apply(days_since_last)

    return df


def create_target(df):
    df["target_qty"] = df.groupby("sku_id")["daily_qty"].shift(-1)
    return df


def train_test_split(df, split_date):
    df = df.dropna().reset_index(drop=True)
    train = df[df["date"] < split_date]
    test = df[df["date"] >= split_date]
    return train, test

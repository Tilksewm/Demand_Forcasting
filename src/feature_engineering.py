import pandas as pd
import numpy as np

def add_sku_rolling_stats(df, window=28):
    df = df.sort_values(["sku_id", "date"]).copy()

    rolled = (
        df.groupby("sku_id")["daily_qty"]
          .shift(1)
          .rolling(window)
    )

    df[f"sku_avg_{window}d"] = rolled.mean()
    df[f"sku_std_{window}d"] = rolled.std()

    return df
def add_time_aware_category_te(df, target_col):
    df = df.sort_values("date").copy()

    df["category_te"] = np.nan
    global_mean = df[target_col].mean()

    for cat, grp in df.groupby("category"):
        expanding_mean = (
            grp[target_col]
            .shift(1)
            .expanding()
            .mean()
        )
        df.loc[grp.index, "category_te"] = expanding_mean

    df["category_te"] = df["category_te"].fillna(global_mean)
    return df

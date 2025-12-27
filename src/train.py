import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from config import (DATA_PROCESSED, MODELS_DIR)
from feature_engineering import (
    add_sku_rolling_stats,
    add_time_aware_category_te
)

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
    "is_fasting",
]

TARGET = "target_qty"


train_df = pd.read_csv(DATA_PROCESSED / "train_data.csv", parse_dates=["date"])
test_df  = pd.read_csv(DATA_PROCESSED / "test_data.csv",  parse_dates=["date"])

train_df = add_sku_rolling_stats(train_df, 28)
test_df  = add_sku_rolling_stats(test_df, 28)

train_df = add_time_aware_category_te(train_df, TARGET)
test_df  = add_time_aware_category_te(test_df, TARGET)

FEATURES = FEATURES_BASE + [
    "sku_avg_28d",
    "sku_std_28d",
    "category_te",
]

train_df = train_df.dropna(subset=FEATURES + [TARGET])
test_df  = test_df.dropna(subset=FEATURES + [TARGET])

model = HistGradientBoostingRegressor(
    max_depth=8,
    learning_rate=0.05,
    max_iter=300,
    random_state=42
)

model.fit(train_df[FEATURES], train_df[TARGET])

joblib.dump(model, MODELS_DIR / "global_demand_model.joblib")

test_df["pred"] = model.predict(test_df[FEATURES]).clip(0)

mae = mean_absolute_error(test_df[TARGET], test_df["pred"])
print(f"Global MAE: {mae:.2f}")

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

SALES_FILE = DATA_RAW / "sales.csv"
FASTING_FILE = DATA_RAW / "fasting_periods.csv"
EVENTS_FILE = DATA_RAW / "ethiopian_events.csv"
MODELS_DIR = PROJECT_ROOT / "models"

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
MIN_HISTORY_DAYS = 90
ROLLING_WINDOW = 28

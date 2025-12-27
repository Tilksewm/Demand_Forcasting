from src.preprocessing import (
    load_sales_data,
    aggregate_daily_sales,
    add_calendar_features,
    add_lag_features,
    add_fasting_feature,
    add_holiday_features,
    create_target,
    train_test_split
)
from src.config import DATA_PROCESSED

def run_pipeline():
    # Load and preprocess data
    df = load_sales_data()
    df = aggregate_daily_sales(df)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_fasting_feature(df)
    df = add_holiday_features(df)
    df = create_target(df)

    train_df, test_df = train_test_split(df, "2023-09-11")

    train_df.to_csv(DATA_PROCESSED / "train_data.csv", index=False)
    test_df.to_csv(DATA_PROCESSED / "test_data.csv", index=False)
    print("Data preprocessing completed and train/test files saved.")

if __name__ == "__main__":
    run_pipeline()
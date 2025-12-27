import matplotlib.pyplot as plt
import seaborn as sns


def plot_monthly_actual_vs_pred(
    monthly_results,
    sku_id,
    save_path=None
):
    df = (
        monthly_results[monthly_results["sku_id"] == sku_id]
        .sort_values("year_month")
    )

    plt.figure(figsize=(12, 6))
    plt.plot(df["year_month"].astype(str), df["actual_monthly_qty"], marker="o", label="Actual")
    plt.plot(df["year_month"].astype(str), df["predicted_monthly_qty"], marker="x", label="Predicted")

    plt.title(f"Monthly Actual vs Predicted — SKU {sku_id}")
    plt.xlabel("Month")
    plt.ylabel("Quantity")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_scatter_and_top_skus(
    monthly_results,
    top_n=5,
    save_path=None
):
    overall_mape = monthly_results["pct_error"].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter
    sns.scatterplot(
        data=monthly_results,
        x="actual_monthly_qty",
        y="predicted_monthly_qty",
        alpha=0.5,
        ax=ax1
    )

    max_val = max(
        monthly_results["actual_monthly_qty"].max(),
        monthly_results["predicted_monthly_qty"].max()
    )
    ax1.plot([0, max_val], [0, max_val], "r--")
    ax1.set_title(f"Actual vs Predicted (MAPE={overall_mape:.2f}%)")

    # Top SKUs
    top_skus = (
        monthly_results.groupby("sku_id")["actual_monthly_qty"]
        .sum()
        .nlargest(top_n)
        .index
    )

    plot_df = monthly_results[monthly_results["sku_id"].isin(top_skus)]
    melted = plot_df.melt(
        id_vars=["sku_id", "year_month"],
        value_vars=["actual_monthly_qty", "predicted_monthly_qty"],
        var_name="Type",
        value_name="Quantity"
    )

    sns.barplot(data=melted, x="sku_id", y="Quantity", hue="Type", ax=ax2)
    ax2.set_title(f"Top {top_n} High-Volume SKUs")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()

def plot_error_trends(results_df, sku_category_df, save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))

    # Error over time
    monthly_avg_error = (
        results_df.groupby("year_month")["pct_error"]
        .mean()
        .reset_index()
    )

    sns.lineplot(
        ax=axes[0],
        data=monthly_avg_error,
        x=monthly_avg_error["year_month"].astype(str),
        y="pct_error",
        marker="o"
    )
    axes[0].set_title("Average % Error Over Time")
    axes[0].tick_params(axis="x", rotation=45)

    # Error by category
    error_by_sku = (
        results_df.groupby("sku_id")["pct_error"]
        .mean()
        .reset_index()
    )

    error_by_cat = (
        error_by_sku
        .merge(sku_category_df, on="sku_id", how="left")
        .groupby("category")["pct_error"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    sns.barplot(
        ax=axes[1],
        data=error_by_cat,
        x="category",
        y="pct_error"
    )
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].set_title("Average % Error by Category")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()

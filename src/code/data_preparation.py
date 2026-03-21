import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")


# ─────────────────────────────────────────────────────────────────────────────
# DATA UNDERSTANDING (to use in 1)Data_Understanding.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def data_understanding_summary(df, dataset_name="Dataset"):
    rows, cols = df.shape

    numeric_cols     = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    datetime_cols    = df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns

    total_missing = df.isnull().sum().sum()
    total_cells   = rows * cols if rows * cols > 0 else 1
    missing_pct   = (total_missing / total_cells) * 100
    duplicates    = df.duplicated().sum()

    summary_text = f"""
============================================================
DATA UNDERSTANDING SUMMARY - {dataset_name}
============================================================

Structure
---------
Rows:                     {rows:,}
Columns:                  {cols:,}

Variable Types
--------------
Numeric variables:        {len(numeric_cols)}
  {list(numeric_cols)}

Categorical variables:    {len(categorical_cols)}
  {list(categorical_cols)}

Datetime variables:       {len(datetime_cols)}
  {list(datetime_cols)}

Data Quality
------------
Total missing values:     {total_missing:,}
Missing percentage:       {missing_pct:.2f}%
Duplicate rows:           {duplicates:,}

============================================================
"""
    return summary_text


def plot_missing(df):
    """Horizontal barplot showing missing % per column (only columns with missing values)."""
    missing = df.isnull().mean().sort_values(ascending=False) * 100
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values found.")
        return

    fig, ax = plt.subplots(figsize=(8, max(3, len(missing) * 0.4)))
    sns.barplot(x=missing.values, y=missing.index, ax=ax)
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Values by Column")
    ax.set_xlim(0, 100)
    plt.tight_layout()
    plt.show()


def visualize_by_variable(df, max_cat=10, dataset_name="Dataset"):
    """
    Visualizations per variable: each variable gets a row with 3 plots.
    - Numeric:     histogram | boxplot | missing %
    - Categorical: countplot | top N categories | missing %
    - Datetime:    count per year | count per month | count per weekday
    """
    numeric_cols     = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    datetime_cols    = df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns
    all_cols         = df.columns

    if len(all_cols) > 20:
        print(f"⚠️  {len(all_cols)} variables detected — this may take a moment.")

    n_rows = len(all_cols)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 3))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, col in enumerate(all_cols):
        col_data = df[col]

        # ── Numeric ──────────────────────────────────────────────────────────
        if col in numeric_cols:
            non_null = col_data.dropna()

            sns.histplot(non_null, bins=30, kde=True, ax=axes[i, 0])
            axes[i, 0].set_title(f"{col} — histogram")

            if not non_null.empty:
                sns.boxplot(x=non_null, ax=axes[i, 1])
                axes[i, 1].set_title(f"{col} — boxplot")

            missing_pct = col_data.isnull().mean() * 100
            axes[i, 2].bar(0, missing_pct)
            axes[i, 2].set_title(f"{col} — missing %")
            axes[i, 2].set_xticks([])
            axes[i, 2].set_ylim(0, 100)

        # ── Categorical ───────────────────────────────────────────────────────
        elif col in categorical_cols:
            if col_data.nunique() <= max_cat:
                sns.countplot(y=col, data=df, order=col_data.value_counts().index, ax=axes[i, 0])
                axes[i, 0].set_title(f"{col} — countplot")

                top_counts = col_data.value_counts().head(max_cat)
                axes[i, 1].barh(top_counts.index, top_counts.values)
                axes[i, 1].set_title(f"{col} — top {max_cat}")
            else:
                axes[i, 0].text(0.5, 0.5, "Too many categories", ha="center")
                axes[i, 0].set_title(f"{col} — skipped")
                axes[i, 1].axis("off")

            missing_pct = col_data.isnull().mean() * 100
            axes[i, 2].bar(0, missing_pct)
            axes[i, 2].set_title(f"{col} — missing %")
            axes[i, 2].set_xticks([])
            axes[i, 2].set_ylim(0, 100)

        # ── Datetime ──────────────────────────────────────────────────────────
        elif col in datetime_cols:
            non_null = col_data.dropna()
            if not non_null.empty:
                df_temp            = non_null.to_frame()
                df_temp["year"]    = df_temp[col].dt.year
                df_temp["month"]   = df_temp[col].dt.month
                df_temp["weekday"] = df_temp[col].dt.day_name()

                sns.countplot(x="year", data=df_temp, ax=axes[i, 0])
                axes[i, 0].set_title(f"{col} — by year")

                sns.countplot(x="month", data=df_temp, ax=axes[i, 1])
                axes[i, 1].set_title(f"{col} — by month")

                sns.countplot(x="weekday", data=df_temp, ax=axes[i, 2],
                              order=["Monday", "Tuesday", "Wednesday", "Thursday",
                                     "Friday", "Saturday", "Sunday"])
                axes[i, 2].set_title(f"{col} — by weekday")
            else:
                for j in range(3):
                    axes[i, j].axis("off")
                axes[i, 0].text(0.5, 0.5, "No data", ha="center")
                axes[i, 0].set_title(f"{col} — empty")

        # ── Other ─────────────────────────────────────────────────────────────
        else:
            axes[i, 0].text(0.5, 0.5, "Plot skipped", ha="center")
            axes[i, 0].set_title(f"{col}")
            for j in range(1, 3):
                axes[i, j].axis("off")

    plt.suptitle(f"{dataset_name} — Variable Overview", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()
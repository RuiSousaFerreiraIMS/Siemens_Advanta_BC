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

    # ─────────────────────────────────────────────────────────────────────────────
# BUs — parse hierarchical sheet into flat DataFrame
# ─────────────────────────────────────────────────────────────────────────────
 
def parse_bus(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Parses the BUs sheet (wide hierarchical format) into a flat DataFrame:
        BU | Segment | Subsegment
    """
    cols = df_raw.columns.tolist()

    # BU names are in the header — forward-fill only from cols starting with 'SSI'
    bu_map = {}
    current_bu = None
    for i, col in enumerate(cols):
        if col.startswith("SSI"):   # <-- fix: era 'not Unnamed' antes
            current_bu = col
        bu_map[i] = current_bu

    segment_row = df_raw.iloc[0]
    records = []

    for col_i, col in enumerate(cols[1:], start=1):
        bu      = bu_map[col_i]
        segment = segment_row[col]
        if pd.isna(segment):
            continue
        for val in df_raw.iloc[1:][col]:
            if pd.notna(val):
                records.append({"BU": bu, "Segment": segment, "Subsegment": val})

    return pd.DataFrame(records)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL DATA PREPARATION HELPERS (to use in 2)Data_Preparation.ipynb)

# This section contains general-purpose functions for data cleaning and 
# preparation that can be applied to any dataset without the risk of data leakage. 
# They are designed to be flexible and configurable via parameters, including 
# detailed printouts of the actions taken for transparency.
# ─────────────────────────────────────────────────────────────────────────────

def drop_high_missing_rows(df, threshold=0.5):
    """Drop rows where more than `threshold` fraction of values are missing."""
    missing_pct = df.isnull().mean(axis=1)
    n_dropped = (missing_pct > threshold).sum()
    print(f"Dropped {n_dropped} rows with >{threshold*100:.0f}% missing values")
    return df[missing_pct <= threshold].reset_index(drop=True)

def validate_categorical_codes(df: pd.DataFrame, bus_df: pd.DataFrame, dataset_name: str = "Dataset") -> pd.DataFrame:
    """
    Checks that TGL Business Unit, Segment, and Subsegment values in df
    all exist in the reference BUs hierarchy.

    Returns a summary DataFrame with any invalid codes found.
    """
    checks = {
        "TGL Business Unit":       bus_df["BU"].unique(),
        "TGL Business Segment":    bus_df["Segment"].unique(),
        "TGL Business Subsegment": bus_df["Subsegment"].unique(),
    }

    results = []
    for col, valid_values in checks.items():
        if col not in df.columns:
            continue
        actual  = set(df[col].dropna().unique())
        valid   = set(valid_values)
        invalid = actual - valid
        results.append({
            "Column":         col,
            "Unique in data": len(actual),
            "Valid in BUs":   len(actual & valid),
            "Invalid codes":  len(invalid),
            "Invalid values": sorted(invalid) if invalid else "✓ All valid",
        })

    summary = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print(f"CATEGORICAL VALIDATION — {dataset_name}")
    print(f"{'='*60}")
    print(summary.to_string(index=False))
    return summary

def impute_gdp(macro_data: pd.DataFrame, period_data: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes GDP missing values in macro_data using a deterministic business rule:
    - Annual GDP columns: value is reported in December only → bfill within year (limit=11)
    - Quarterly GDP columns: value is reported in Mar/Jun/Sep/Dec → bfill within quarter (limit=2)
 
    Japan_GDP_from_Construction and Japan_GDP_from_Manufacturing are treated as
    annual (December only) despite being GDP_from columns.
 
    Esporadic missings (~1-2%) in other columns are left for the pipeline imputer
    to avoid data leakage.
 
    Args:
        macro_data:  raw macro_data DataFrame (Sheet1)
        period_data: period mapping DataFrame (Sheet2) with columns [DATE, Period]
 
    Returns:
        macro_data with GDP columns imputed, without any date helper columns.
    """
    # Temporary merge to get month/year/quarter info
    df = macro_data.merge(period_data, on="Period", how="left")
    df["DATE"]    = pd.to_datetime(df["DATE"])
    df["year"]    = df["DATE"].dt.year
    df["quarter"] = df["DATE"].dt.quarter
    df = df.sort_values("Period").reset_index(drop=True)
 
    # Identify GDP column groups
    gdp_annual = (
        [c for c in df.columns if "GDP" in c and "from" not in c]
        + ["Japan_GDP_from_Construction", "Japan_GDP_from_Manufacturing"]
    )
    gdp_quarterly = [
        c for c in df.columns
        if "GDP_from" in c and "Japan" not in c
    ]
 
    # bfill annual GDPs within each year (December → 11 preceding months)
    df[gdp_annual] = (
        df.groupby("year")[gdp_annual]
        .transform(lambda x: x.bfill(limit=11))
    )
 
    # bfill quarterly GDPs within each year-quarter (last month → 2 preceding months)
    df[gdp_quarterly] = (
        df.groupby(["year", "quarter"])[gdp_quarterly]
        .transform(lambda x: x.bfill(limit=2))
    )
 
    # Drop temporary date helper columns
    df = df.drop(columns=["DATE", "year", "quarter"])
 
    return df

def impute_macro_sporadic(macro_data: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes sporadic missing values (~1-2%) in macro_data using forward fill.
    These are isolated missing values in monthly indicators (e.g. Inflation, 
    Exports, Interest Rate) where the previous period is a valid estimate.

    GDP columns are excluded as they are handled separately by impute_gdp().
    Applied before merging with training/validation to avoid data leakage.
    """
    gdp_cols = [c for c in macro_data.columns if "GDP" in c]
    non_gdp_cols = [c for c in macro_data.columns 
                    if c not in gdp_cols and c != "Period"]

    macro_data = macro_data.sort_values("Period").reset_index(drop=True)
    macro_data[non_gdp_cols] = macro_data[non_gdp_cols].ffill()

    return macro_data

def merge_with_macro(df: pd.DataFrame, macro_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merges training or validation with macro_data on Period.
    Uses inner join to keep only periods present in both datasets (1-48).
    The macro historical buffer (periods -131 to 0) is preserved in macro_data_clean
    for future lag feature engineering.
    """
    return (
        df.merge(macro_data, left_on="Anon Period", right_on="Period", how="inner")
        .drop(columns=["Period"])
    )

# ─────────────────────────────────────────────────────────────────────────────
# HIERARCHICAL AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────
 
def create_hierarchy(df: pd.DataFrame) -> dict:
    """
    Creates 4 hierarchical aggregation levels from a merged dataset
    (training or validation).
 
    Aggregation keys:
        - subsegment : Anon Period + BU + Segment + Subsegment  (no aggregation)
        - segment    : Anon Period + BU + Segment
        - bu         : Anon Period + BU
        - total      : Anon Period
 
    Orders and Revenue are summed at each level.
    Macro columns are the same per period so they are kept via first().
 
    Returns:
        dict with keys "subsegment", "segment", "bu", "total"
    """
    target_cols = ["Orders cons. (anon)", "Revenue cons. (anon)"]
    macro_cols  = [c for c in df.columns if c not in [
        "Anon Period", "TGL Business Unit", "TGL Business Segment",
        "TGL Business Subsegment"] + target_cols
    ]
 
    def agg(group_cols):
        return (
            df.groupby(group_cols, as_index=False)
            .agg({**{c: "sum" for c in target_cols},
                  **{c: "first" for c in macro_cols}})
        )
 
    return {
        "subsegment": df.copy(),
        "segment": agg(["Anon Period", "TGL Business Unit", "TGL Business Segment"]),
        "bu":      agg(["Anon Period", "TGL Business Unit"]),
        "total":   agg(["Anon Period"]),
    }
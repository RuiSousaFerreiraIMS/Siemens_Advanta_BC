"""
Reusable EDA functions for the Siemens Business Case project.
Organized into 4 sections:
  1. Univariate Analysis
  2. Missing Values
  3. Multivariate Analysis (Macro Data)
  4. Target Correlation Analysis
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# Country prefixes used in macro data columns
COUNTRIES = [
    "China", "France", "Germany", "Italy",
    "Japan", "Switzerland", "United_Kingdom", "United_States",
]


def _get_country(col: str) -> str | None:
    """Extract country prefix from a macro column name."""
    for c in COUNTRIES:
        if col.startswith(c):
            return c
    return None


def _get_indicator(col: str) -> str | None:
    """Extract indicator name (part after country prefix) from a macro column name."""
    for c in COUNTRIES:
        if col.startswith(c + "_"):
            return col[len(c) + 1:]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. UNIVARIATE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extended descriptive statistics for numeric columns.
    Includes count, mean, std, min, max, skewness, and kurtosis.
    """
    stats = df.describe().T
    stats["skew"] = df.skew(numeric_only=True)
    stats["kurtosis"] = df.kurtosis(numeric_only=True)
    stats["missing_%"] = (df.isnull().mean() * 100).round(2)
    return stats


def plot_distributions(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    ncols: int = 4,
    figsize_per_plot: tuple = (4, 3),
):
    """Histogram + KDE for numeric columns."""
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in cols if c != "Period"]

    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_plot[0] * ncols,
                                      figsize_per_plot[1] * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        sns.histplot(df[col].dropna(), bins=30, kde=True, ax=axes[i])
        axes[i].set_title(col, fontsize=8)
        axes[i].tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Distributions of Numeric Variables", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_boxplots(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    ncols: int = 4,
    figsize_per_plot: tuple = (4, 3),
):
    """Boxplots grid for numeric columns."""
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in cols if c != "Period"]

    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_plot[0] * ncols,
                                      figsize_per_plot[1] * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        sns.boxplot(x=df[col].dropna(), ax=axes[i])
        axes[i].set_title(col, fontsize=8)
        axes[i].tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Boxplots of Numeric Variables", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 2. MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────

def plot_missing_heatmap(df: pd.DataFrame, figsize: tuple = (20, 8)):
    """
    Heatmap showing missing value patterns across all columns and rows.
    Rows = observations (periods), Columns = features.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df.isnull().T, cbar=False, cmap="YlOrRd",
                yticklabels=True, ax=ax)
    ax.set_title("Missing Value Pattern (yellow = missing)", fontsize=14)
    ax.set_xlabel("Row index (Period)")
    ax.set_ylabel("Feature")
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    plt.show()


def plot_missing_by_country(df: pd.DataFrame):
    """
    Bar chart showing average missing % grouped by country prefix.
    Useful to identify if certain countries have systematically more gaps.
    """
    records = []
    for col in df.columns:
        country = _get_country(col)
        if country:
            pct = df[col].isnull().mean() * 100
            records.append({"Country": country, "Column": col, "Missing_%": pct})

    if not records:
        print("No country-prefixed columns found.")
        return

    miss_df = pd.DataFrame(records)
    avg = miss_df.groupby("Country")["Missing_%"].mean().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Average by country
    sns.barplot(x=avg.values, y=avg.index, ax=axes[0], palette="OrRd_r")
    axes[0].set_title("Average Missing % by Country")
    axes[0].set_xlabel("Missing (%)")
    axes[0].set_xlim(0, 100)

    # Per-column breakdown
    sns.barplot(data=miss_df.sort_values("Missing_%", ascending=False).head(20),
                x="Missing_%", y="Column", hue="Country", dodge=False, ax=axes[1])
    axes[1].set_title("Top 20 Columns with Most Missing Values")
    axes[1].set_xlabel("Missing (%)")
    axes[1].set_xlim(0, 100)
    axes[1].tick_params(axis="y", labelsize=7)
    axes[1].legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 3. MULTIVARIATE ANALYSIS — MACRO DATA
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_matrix(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    title: str = "Correlation Matrix",
    figsize: tuple = (18, 15),
    annot: bool = False,
):
    """Heatmap of Pearson correlation for numeric columns."""
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in cols if c != "Period"]

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0,
                annot=annot, fmt=".2f" if annot else "",
                linewidths=0.5, ax=ax,
                xticklabels=True, yticklabels=True)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="both", labelsize=7)
    plt.tight_layout()
    plt.show()


def plot_correlation_by_country(df: pd.DataFrame, annot: bool = True):
    """
    Separate correlation heatmap per country — shows how that country's
    indicators relate to each other.
    """
    for country in COUNTRIES:
        country_cols = [c for c in df.columns if c.startswith(country + "_")]
        if len(country_cols) < 2:
            continue

        # Rename for readability: remove country prefix
        sub = df[country_cols].rename(
            columns={c: _get_indicator(c) for c in country_cols}
        )
        corr = sub.corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0,
                    annot=annot, fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title(f"Correlation Matrix — {country.replace('_', ' ')}",
                     fontsize=12)
        plt.tight_layout()
        plt.show()


def plot_time_series(
    df: pd.DataFrame,
    cols: list[str],
    period_col: str = "Period",
    title: str = "Time Series",
    figsize: tuple = (14, 5),
):
    """Line plots of selected columns over time."""
    fig, ax = plt.subplots(figsize=figsize)
    for col in cols:
        ax.plot(df[period_col], df[col], label=col, linewidth=1)
    ax.set_xlabel("Period")
    ax.set_ylabel("Value")
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.show()


def plot_time_series_by_country(
    df: pd.DataFrame,
    indicator: str,
    period_col: str = "Period",
    figsize: tuple = (14, 5),
):
    """
    Compare the same indicator across all countries over time.
    `indicator` should be the part after the country prefix, e.g. 'GDP', 'Exports'.
    """
    cols = [f"{c}_{indicator}" for c in COUNTRIES
            if f"{c}_{indicator}" in df.columns]
    if not cols:
        print(f"No columns found for indicator '{indicator}'.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    for col in cols:
        country = _get_country(col)
        ax.plot(df[period_col], df[col],
                label=country.replace("_", " "), linewidth=1)
    ax.set_xlabel("Period")
    ax.set_ylabel(indicator.replace("_", " "))
    ax.set_title(f"{indicator.replace('_', ' ')} — Cross-Country Comparison",
                 fontsize=14)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 4. TARGET CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def plot_target_correlations(
    df: pd.DataFrame,
    target_cols: list[str],
    top_n: int = 15,
    figsize: tuple = (10, 6),
):
    """
    Horizontal bar chart of the top N features most correlated
    (positive or negative) with each target variable.
    """
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()

    for target in target_cols:
        if target not in corr.columns:
            print(f"Target '{target}' not found in DataFrame.")
            continue

        target_corrs = (
            corr[target]
            .drop(labels=target_cols, errors="ignore")
            .dropna()
            .abs()
            .sort_values(ascending=False)
            .head(top_n)
        )

        # Get actual signed values for the top N
        signed = corr[target].loc[target_corrs.index]

        fig, ax = plt.subplots(figsize=figsize)
        colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in signed.values]
        ax.barh(signed.index[::-1], signed.values[::-1], color=colors[::-1])
        ax.set_xlabel("Pearson Correlation")
        ax.set_title(f"Top {top_n} Features Correlated with {target}",
                     fontsize=13)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()
        plt.show()


def plot_scatter_vs_target(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    ncols: int = 3,
    figsize_per_plot: tuple = (4, 3),
):
    """Scatter plot grid of features vs a target variable."""
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_plot[0] * ncols,
                                      figsize_per_plot[1] * nrows))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(features):
        axes[i].scatter(df[feat], df[target], alpha=0.4, s=10)
        axes[i].set_xlabel(feat, fontsize=7)
        axes[i].set_ylabel(target, fontsize=7)
        axes[i].set_title(f"{feat} vs {target}", fontsize=8)
        axes[i].tick_params(labelsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Feature Scatter Plots vs {target}", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_pairplot(df: pd.DataFrame, cols: list[str], hue: str | None = None):
    """Seaborn pairplot for selected columns."""
    sns.pairplot(df[cols].dropna(), diag_kind="kde", hue=hue,
                 plot_kws={"alpha": 0.4, "s": 15})
    plt.suptitle("Pairplot", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5. BUSINESS DATA ANALYSIS — Revenue & Orders by BU / Segment
# ─────────────────────────────────────────────────────────────────────────────

def plot_revenue_by_bu(
    df: pd.DataFrame,
    target: str = "Revenue cons. (anon)",
    bu_col: str = "TGL Business Unit",
    period_col: str = "Anon Period",
    figsize: tuple = (16, 5),
):
    """
    Two-panel view:
    - Left:  Total revenue per BU (horizontal bar)
    - Right: Revenue over time per BU (line chart)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: aggregate bar
    bu_total = (
        df.groupby(bu_col)[target]
        .sum()
        .sort_values(ascending=True)
    )
    palette = sns.color_palette("viridis", n_colors=len(bu_total))
    axes[0].barh(bu_total.index, bu_total.values, color=palette)
    axes[0].set_xlabel("Total Revenue (sum over all periods)")
    axes[0].set_title("Total Revenue by Business Unit", fontsize=13)
    axes[0].tick_params(axis="y", labelsize=9)
    # Format x-axis with M suffix
    axes[0].xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M")
    )

    # Right: time series by BU
    bu_ts = (
        df.groupby([period_col, bu_col])[target]
        .sum()
        .reset_index()
    )
    for bu in bu_total.index:
        sub = bu_ts[bu_ts[bu_col] == bu]
        axes[1].plot(sub[period_col], sub[target], label=bu, linewidth=1.5)
    axes[1].set_xlabel("Period")
    axes[1].set_ylabel("Revenue")
    axes[1].set_title("Revenue Over Time by Business Unit", fontsize=13)
    axes[1].legend(fontsize=8, loc="best")
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M")
    )

    plt.tight_layout()
    plt.show()


def plot_revenue_by_segment(
    df: pd.DataFrame,
    target: str = "Revenue cons. (anon)",
    bu_col: str = "TGL Business Unit",
    seg_col: str = "TGL Business Segment",
    period_col: str = "Anon Period",
    top_n: int = 8,
):
    """
    For each BU, show:
    - Left:  Top N Segments by total revenue (bar chart)
    - Right: Revenue over time per Segment (line chart)
    """
    bus = df[bu_col].unique()

    for bu in sorted(bus):
        bu_data = df[df[bu_col] == bu]
        seg_total = (
            bu_data.groupby(seg_col)[target]
            .sum()
            .sort_values(ascending=False)
        )
        top_segs = seg_total.head(top_n)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Left: bar chart
        palette = sns.color_palette("magma", n_colors=len(top_segs))
        axes[0].barh(
            top_segs.index[::-1],
            top_segs.values[::-1],
            color=palette[::-1],
        )
        axes[0].set_xlabel("Total Revenue")
        axes[0].set_title(f"Top {min(top_n, len(seg_total))} Segments — {bu}", fontsize=12)
        axes[0].tick_params(axis="y", labelsize=8)
        axes[0].xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M")
        )

        # Right: time series
        seg_ts = (
            bu_data[bu_data[seg_col].isin(top_segs.index)]
            .groupby([period_col, seg_col])[target]
            .sum()
            .reset_index()
        )
        for seg in top_segs.index:
            sub = seg_ts[seg_ts[seg_col] == seg]
            axes[1].plot(sub[period_col], sub[target], label=seg, linewidth=1.2)
        axes[1].set_xlabel("Period")
        axes[1].set_ylabel("Revenue")
        axes[1].set_title(f"Revenue Over Time — {bu}", fontsize=12)
        axes[1].legend(fontsize=7, loc="best")
        axes[1].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M")
        )

        plt.tight_layout()
        plt.show()


def plot_revenue_share(
    df: pd.DataFrame,
    target: str = "Revenue cons. (anon)",
    bu_col: str = "TGL Business Unit",
    seg_col: str = "TGL Business Segment",
    figsize: tuple = (14, 7),
):
    """
    Stacked bar chart showing each BU's revenue composition by Segment.
    Reveals which segments dominate within each BU.
    """
    pivot = (
        df.groupby([bu_col, seg_col])[target]
        .sum()
        .reset_index()
        .pivot_table(index=bu_col, columns=seg_col, values=target, fill_value=0)
    )

    # Sort BUs by total revenue
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    # Keep only top segments for readability, group rest as "Other"
    seg_totals = pivot.sum(axis=0).sort_values(ascending=False)
    top_segs = seg_totals.head(12).index.tolist()
    other_segs = [s for s in pivot.columns if s not in top_segs]
    if other_segs:
        pivot["Other"] = pivot[other_segs].sum(axis=1)
        pivot = pivot.drop(columns=other_segs)

    fig, ax = plt.subplots(figsize=figsize)
    pivot.plot(kind="barh", stacked=True, ax=ax,
               colormap="tab20", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Total Revenue")
    ax.set_title("Revenue Composition — Segments within each Business Unit", fontsize=13)
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", title="Segment")
    ax.tick_params(axis="y", labelsize=9)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M")
    )
    plt.tight_layout()
    plt.show()


def plot_orders_vs_revenue(
    df: pd.DataFrame,
    target: str = "Revenue cons. (anon)",
    orders_col: str = "Orders cons. (anon)",
    bu_col: str = "TGL Business Unit",
    period_col: str = "Anon Period",
    figsize: tuple = (16, 5),
):
    """
    Two-panel view:
    - Left:  Scatter plot Orders vs Revenue (colored by BU)
    - Right: Orders and Revenue time series (aggregated)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: scatter
    bus = sorted(df[bu_col].unique())
    palette = sns.color_palette("viridis", n_colors=len(bus))
    for j, bu in enumerate(bus):
        sub = df[df[bu_col] == bu]
        axes[0].scatter(
            sub[orders_col], sub[target],
            alpha=0.4, s=15, label=bu, color=palette[j],
        )
    axes[0].set_xlabel("Orders")
    axes[0].set_ylabel("Revenue")
    axes[0].set_title("Orders vs Revenue (by BU)", fontsize=13)
    axes[0].legend(fontsize=7, loc="best")
    axes[0].xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M")
    )
    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M")
    )

    # Right: aggregate time series
    agg = df.groupby(period_col)[[target, orders_col]].sum().reset_index()
    axes[1].plot(agg[period_col], agg[target], label="Revenue", linewidth=1.5, color="#2ecc71")
    axes[1].plot(agg[period_col], agg[orders_col], label="Orders", linewidth=1.5,
                 color="#3498db", linestyle="--")
    axes[1].set_xlabel("Period")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Total Revenue vs Orders Over Time", fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M")
    )

    plt.tight_layout()
    plt.show()

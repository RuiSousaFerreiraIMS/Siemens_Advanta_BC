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

# agent/nodes/node2_eda.py
#
# WHY EDA first?
# Data ko bina dekhe model banana = andheron mein teer chalana.
# EDA batata hai: missing data kitna hai, distributions kaisi hain,
# koi outlier toh nahi, columns correlated hain ya nahi.

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # WHY Agg? Server par display nahi hota, file mein save karna hai
import matplotlib.pyplot as plt
import seaborn as sns

from agent.state import AgentState

# ── Chart output directory ────────────────────────────────────────────────────
CHARTS_DIR = "outputs/charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# ── Seaborn theme (better default looks) ─────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")


def run_eda(state: AgentState) -> dict:
    """
    Node 2: Full EDA — statistics collect karo aur charts save karo.
    """
    print("\n[Node 2] Running EDA...")

    df: pd.DataFrame = state["df"]
    chart_paths = []

    # ════════════════════════════════════════════════════════════════════════
    # PART A — Statistics (numbers, no charts)
    # ════════════════════════════════════════════════════════════════════════

    # ── Separate numeric and categorical columns ──────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ── Missing value analysis ────────────────────────────────────────────────
    # For each column: count nulls + percentage
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    missing_info = {
        col: {"count": int(null_counts[col]), "percent": float(null_pct[col])}
        for col in df.columns
        if null_counts[col] > 0
    }

    # ── Descriptive statistics ────────────────────────────────────────────────
    # .describe() gives count, mean, std, min, 25%, 50%, 75%, max
    describe_dict = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().round(4)
        describe_dict = desc.to_dict()

    # ── Correlation matrix (numeric columns only) ─────────────────────────────
    # WHY? High correlation (>0.9) between features = redundant info
    correlation_dict = {}
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().round(4)
        correlation_dict = corr_matrix.to_dict()

    # ── Categorical value counts ──────────────────────────────────────────────
    # For each categorical column, top 10 most frequent values
    cat_value_counts = {}
    for col in categorical_cols:
        counts = df[col].value_counts().head(10)
        cat_value_counts[col] = counts.to_dict()

    eda_summary = {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "column_names": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_values": missing_info,
        "descriptive_stats": describe_dict,
        "correlations": correlation_dict,
        "categorical_counts": cat_value_counts,
    }

    print(f"  ✓ Stats collected: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical cols")
    print(f"  ✓ Missing values in: {list(missing_info.keys()) if missing_info else 'none'}")

    # ════════════════════════════════════════════════════════════════════════
    # PART B — Charts
    # ════════════════════════════════════════════════════════════════════════

    # ── Chart 1: Missing Values Heatmap ──────────────────────────────────────
    if missing_info:
        path = _plot_missing_values(df)
        chart_paths.append(path)

    # ── Chart 2: Numeric Distributions (histogram per column) ────────────────
    if numeric_cols:
        path = _plot_distributions(df, numeric_cols)
        chart_paths.append(path)

    # ── Chart 3: Correlation Heatmap ─────────────────────────────────────────
    if len(numeric_cols) >= 2:
        path = _plot_correlation_heatmap(df, numeric_cols)
        chart_paths.append(path)

    # ── Chart 4: Categorical Bar Charts ──────────────────────────────────────
    if categorical_cols:
        path = _plot_categorical_counts(df, categorical_cols)
        chart_paths.append(path)

    print(f"  ✓ Charts saved: {chart_paths}")

    return {
        "eda_summary": eda_summary,
        "chart_paths": chart_paths,
    }


# ════════════════════════════════════════════════════════════════════════════
# Chart helper functions (private — underscore prefix convention)
# ════════════════════════════════════════════════════════════════════════════

def _plot_missing_values(df: pd.DataFrame) -> str:
    """
    Missing values ko heatmap mein dikhao.
    Yellow = missing, Purple = present (seaborn default).
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    # Create boolean mask: True = missing
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap="viridis", ax=ax)
    ax.set_title("Missing Values Heatmap\n(Yellow = Missing)", fontsize=13)
    ax.set_xlabel("Columns")

    path = os.path.join(CHARTS_DIR, "missing_values.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)   # WHY close? Memory leak rokne ke liye — server par bohot CSVs process ho sakti hain
    return path


def _plot_distributions(df: pd.DataFrame, numeric_cols: list) -> str:
    """
    Har numeric column ka histogram + KDE (Kernel Density Estimate) plot.
    KDE = smoothed version of histogram, distribution ka shape dikhata hai.
    """
    # ── Grid layout calculate karo ────────────────────────────────────────────
    n = len(numeric_cols)
    ncols = min(3, n)                        # Max 3 charts per row
    nrows = (n + ncols - 1) // ncols         # Ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))

    # axes ko always 1D array bana do (easy iteration)
    if n == 1:
        axes = [axes]
    elif nrows == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        # dropna() because histogram NaN values handle nahi karta
        data = df[col].dropna()
        if len(data) == 0 or data.nunique() < 2:
          ax.set_title(f"{col}\n(insufficient data)", fontsize=11)
          ax.text(0.5, 0.5, "No data to plot", transform=ax.transAxes,
            ha="center", va="center", color="gray")
        continue

        

        # KDE overlay on a twin axis
        ax2 = ax.twinx()
        data.plot.kde(ax=ax2, color="#E05C3A", linewidth=1.5)
        ax2.set_yticks([])     # KDE y-axis numbers confusing hote hain, hide karo

        ax.set_title(col, fontsize=11)
        ax.set_xlabel("")
        ax2.set_ylabel("")

    # Extra axes hide karo (agar columns < grid size)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric Column Distributions", fontsize=14, y=1.01)
    path = os.path.join(CHARTS_DIR, "distributions.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list) -> str:
    """
    Correlation matrix heatmap.
    +1 = perfect positive correlation, -1 = perfect negative, 0 = no relation.
    """
    corr = df[numeric_cols].corr()

    # Figure size: zyada columns = bada chart
    size = max(6, len(numeric_cols) * 0.8)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))

    # annot=True = numbers bhi dikhao andar
    # fmt=".2f" = 2 decimal places
    # mask upper triangle (duplicate info hai)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",    # Blue = negative, Red = positive correlation
        mask=mask,
        ax=ax,
        linewidths=0.5,
        vmin=-1, vmax=1,
        square=True,
    )
    ax.set_title("Feature Correlation Matrix\n(|value| > 0.8 = high correlation)", fontsize=12)

    path = os.path.join(CHARTS_DIR, "correlation_heatmap.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_categorical_counts(df: pd.DataFrame, categorical_cols: list) -> str:
    """
    Categorical columns ke liye bar charts — value counts.
    """
    # Max 6 categorical columns plot karo (zyada hote hain toh cluttered)
    cols_to_plot = categorical_cols[:6]
    n = len(cols_to_plot)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))

    if n == 1:
        axes = [axes]
    elif nrows == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        counts = df[col].value_counts().head(10)   # Top 10 only

        counts.plot.bar(ax=ax, color="#5B8DB8", edgecolor="white", alpha=0.8)
        ax.set_title(f"{col}\n(top 10 values)", fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=40, labelsize=8)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Categorical Column Distributions", fontsize=13)
    path = os.path.join(CHARTS_DIR, "categorical_counts.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path

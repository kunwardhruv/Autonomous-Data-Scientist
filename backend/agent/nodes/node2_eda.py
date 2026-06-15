# agent/nodes/node2_eda.py
#
# WHY EDA first?
# Data ko bina dekhe model banana = andheron mein teer chalana.
# EDA batata hai: missing data kitna hai, distributions kaisi hain,
# koi outlier toh nahi, columns correlated hain ya nahi.
#
# WHY base64 instead of saving files?
# Deployment pe (Railway/Render) filesystem ephemeral hota hai —
# server restart hone par saari saved files delete ho jaati hain.
# Base64 = chart ko PNG bytes mein convert karo, phir string bana do.
# Yeh string directly JSON response mein jaati hai — koi file system dependency nahi.
# Frontend <img src="data:image/png;base64,..." /> se seedha render karta hai.

import base64
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # WHY Agg? Server par display nahi hota, memory mein render karna hai
import matplotlib.pyplot as plt
import seaborn as sns

from agent.state import AgentState

sns.set_theme(style="whitegrid", palette="muted")


def run_eda(state: AgentState) -> dict:
    """
    Node 2: Full EDA — statistics collect karo aur charts base64 mein generate karo.
    """
    print("\n[Node 2] Running EDA...")

    df: pd.DataFrame = state["df"]
    chart_base64s = []   # List of {"name": "...", "data": "base64string"}

    # ════════════════════════════════════════════════════════════════════════
    # PART A — Statistics (numbers, no charts)
    # ════════════════════════════════════════════════════════════════════════

    # ── Numeric aur categorical columns alag karo ─────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ── Missing value analysis ────────────────────────────────────────────────
    # Har column ke liye: count nulls + percentage
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

    # ── Correlation matrix ────────────────────────────────────────────────────
    # WHY? High correlation (>0.9) between features = redundant info
    # Ek feature doosre ka kaam kar raha hai — redundancy ML mein problem hai
    correlation_dict = {}
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().round(4)
        correlation_dict = corr_matrix.to_dict()

    # ── Categorical value counts ──────────────────────────────────────────────
    # Har categorical column mein top 10 most frequent values
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
    # PART B — Charts (base64, no file saving)
    # ════════════════════════════════════════════════════════════════════════

    # ── Chart 1: Missing Values Heatmap ──────────────────────────────────────
    if missing_info:
        b64 = _plot_missing_values(df)
        if b64:
            chart_base64s.append({"name": "Missing Values", "data": b64})

    # ── Chart 2: Numeric Distributions ───────────────────────────────────────
    if numeric_cols:
        b64 = _plot_distributions(df, numeric_cols)
        if b64:
            chart_base64s.append({"name": "Distributions", "data": b64})

    # ── Chart 3: Correlation Heatmap ─────────────────────────────────────────
    if len(numeric_cols) >= 2:
        b64 = _plot_correlation_heatmap(df, numeric_cols)
        if b64:
            chart_base64s.append({"name": "Correlation Heatmap", "data": b64})

    # ── Chart 4: Categorical Bar Charts ──────────────────────────────────────
    if categorical_cols:
        b64 = _plot_categorical_counts(df, categorical_cols)
        if b64:
            chart_base64s.append({"name": "Categorical Counts", "data": b64})

    print(f"  ✓ Charts generated: {len(chart_base64s)}")

    return {
        "eda_summary": eda_summary,
        "chart_base64s": chart_base64s,
    }


# ════════════════════════════════════════════════════════════════════════════
# Helper: matplotlib figure → base64 PNG string
# ════════════════════════════════════════════════════════════════════════════

def _fig_to_base64(fig) -> str:
    """
    WHY BytesIO?
    File mein save karne ki jagah memory buffer mein save karo.
    BytesIO = RAM mein ek "virtual file" — disk touch nahi hota.
    Phir us buffer ko base64 string mein convert karo.
    Result: "iVBORw0KGgo..." jaisi string jo JSON mein fit ho jaaye.
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)   # Memory leak rokne ke liye — figure RAM se free karo
    buf.seek(0)      # Buffer ka pointer start pe le aao
    return base64.b64encode(buf.read()).decode("utf-8")


def _plot_missing_values(df: pd.DataFrame) -> str:
    """
    Missing values ko heatmap mein dikhao.
    Yellow = missing, Purple = present (viridis colormap).
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        # df.isnull() = True/False matrix, heatmap isko color mein dikhata hai
        sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap="viridis", ax=ax)
        ax.set_title("Missing Values Heatmap\n(Yellow = Missing)", fontsize=13)
        ax.set_xlabel("Columns")
        fig.tight_layout()
        return _fig_to_base64(fig)
    except Exception as e:
        print(f"  ✗ Missing values chart failed: {e}")
        return None


def _plot_distributions(df: pd.DataFrame, numeric_cols: list) -> str:
    """
    Har numeric column ka histogram + KDE (Kernel Density Estimate) plot.
    KDE = smoothed version of histogram — distribution ka actual shape dikhata hai.
    """
    try:
        n = len(numeric_cols)
        ncols = min(3, n)                     # Max 3 charts per row
        nrows = (n + ncols - 1) // ncols      # Ceiling division for rows

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))

        # axes ko always 1D list bana do — easy iteration ke liye
        if n == 1:
            axes = [axes]
        elif nrows == 1:
            axes = list(axes)
        else:
            axes = [ax for row in axes for ax in row]

        for i, col in enumerate(numeric_cols):
            ax = axes[i]
            data = df[col].dropna()   # WHY dropna? Histogram NaN handle nahi karta

            # Empty ya single-value column → chart banana possible nahi
            if len(data) == 0 or data.nunique() < 2:
                ax.set_title(f"{col}\n(insufficient data)", fontsize=11)
                ax.text(0.5, 0.5, "No data to plot", transform=ax.transAxes,
                        ha="center", va="center", color="gray")
                continue

            ax.hist(data, bins=30, alpha=0.6, color="#5B8DB8", edgecolor="white")

            # KDE on twin axis — WHY twinx? KDE aur histogram ka y-scale alag hota hai
            ax2 = ax.twinx()
            try:
                data.plot.kde(ax=ax2, color="#E05C3A", linewidth=1.5)
            except Exception:
                pass
            ax2.set_yticks([])   # KDE ke y numbers confusing hote hain — hide karo

            ax.set_title(col, fontsize=11)
            ax.set_xlabel("")
            ax2.set_ylabel("")

        # Extra axes hide karo (agar columns < grid size)
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Numeric Column Distributions", fontsize=14, y=1.01)
        fig.tight_layout()
        return _fig_to_base64(fig)
    except Exception as e:
        print(f"  ✗ Distributions chart failed: {e}")
        return None


def _plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list) -> str:
    """
    Correlation matrix heatmap.
    +1 = perfect positive correlation, -1 = perfect negative, 0 = no relation.
    Sirf lower triangle dikhao — upper triangle duplicate info hai.
    """
    try:
        corr = df[numeric_cols].corr()

        # Agar saari values NaN hain (e.g. single-row dataset) → skip
        if corr.isnull().all().all():
            return None

        size = max(6, len(numeric_cols) * 0.8)   # Columns ke hisaab se size adjust
        fig, ax = plt.subplots(figsize=(size, size * 0.85))

        # mask = upper triangle hide karo (duplicate info)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, annot=True, fmt=".2f",
            cmap="coolwarm",   # Blue = negative, Red = positive correlation
            mask=mask, ax=ax,
            linewidths=0.5, vmin=-1, vmax=1, square=True,
        )
        ax.set_title("Feature Correlation Matrix\n(|value| > 0.8 = high correlation)", fontsize=12)
        fig.tight_layout()
        return _fig_to_base64(fig)
    except Exception as e:
        print(f"  ✗ Correlation chart failed: {e}")
        return None


def _plot_categorical_counts(df: pd.DataFrame, categorical_cols: list) -> str:
    """
    Categorical columns ke value counts bar chart.
    Max 6 columns plot karo — zyada hote hain toh cluttered dikhta hai.
    """
    try:
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
            counts = df[col].value_counts().head(10)   # Top 10 only — baaki clutter
            counts.plot.bar(ax=ax, color="#5B8DB8", edgecolor="white", alpha=0.8)
            ax.set_title(f"{col}\n(top 10 values)", fontsize=10)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=40, labelsize=8)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Categorical Column Distributions", fontsize=13)
        fig.tight_layout()
        return _fig_to_base64(fig)
    except Exception as e:
        print(f"  ✗ Categorical chart failed: {e}")
        return None
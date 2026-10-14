"""
generate_eda_images.py
Generates EDA images for the project report's Chapter 3 (Methodology).
Run from the project root: python generate_eda_images.py
Saves all images to results/ directory.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

ROOT    = os.path.dirname(os.path.abspath(__file__))
PROC    = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi":   150,
    "savefig.dpi":  150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# ──────────────────────────────────────────────────────────────────────────────
# 1. Dataset Summary Bar Chart
# ──────────────────────────────────────────────────────────────────────────────
def plot_dataset_summary():
    categories = ["Total\nInteractions", "Unique\nStudents", "Unique\nSkills", "Unique\nProblems", "Valid\nProblem-Skill\nMappings", "Sequences\n(T=50)"]
    values     = [6_123_270, 4_918, 265, 179_999, 53_091, 149_047]
    colors     = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(categories, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("ASSISTments 2012-2013 Dataset Overview")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "eda_dataset_summary.png"))
    plt.close()
    print("  [OK] eda_dataset_summary.png")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Affect Annotation Distribution
# ──────────────────────────────────────────────────────────────────────────────
def plot_affect_distribution():
    aff_path = os.path.join(PROC, "affect_interactions.csv")
    if not os.path.exists(aff_path):
        print("  [SKIP] affect_interactions.csv not found")
        return

    print("  Loading affect_interactions.csv ...")
    df = pd.read_csv(aff_path, usecols=lambda c: c in [
        "ORIG_frustration", "ORIG_confusion",
        "ORIG_concentration", "ORIG_boredom",
        "frustration", "confusion", "concentration", "boredom"
    ], nrows=200_000)

    # Use whatever columns exist
    affect_cols = [c for c in ["frustration", "confusion", "concentration", "boredom"] if c in df.columns]
    if not affect_cols:
        affect_cols = [c for c in ["ORIG_frustration", "ORIG_confusion", "ORIG_concentration", "ORIG_boredom"] if c in df.columns]
    if not affect_cols:
        print("  [SKIP] No affect columns found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    palette = ["#C44E52", "#8172B2", "#55A868", "#CCB974"]
    labels  = ["Frustration", "Confusion", "Concentration", "Boredom"]

    for i, (col, label, color) in enumerate(zip(affect_cols[:4], labels, palette)):
        data = df[col].dropna()
        axes[i].hist(data, bins=40, color=color, edgecolor="white", alpha=0.85)
        axes[i].set_title(f"{label} Distribution")
        axes[i].set_xlabel("Annotation Value")
        axes[i].set_ylabel("Frequency")
        mean_val = data.mean()
        axes[i].axvline(mean_val, color="black", linestyle="--", linewidth=1.2, label=f"Mean={mean_val:.2f}")
        axes[i].legend(fontsize=9)
        axes[i].yaxis.grid(True, linestyle="--", alpha=0.4)
        axes[i].set_axisbelow(True)

    plt.suptitle("ASSISTments Affect Annotation Distributions", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "eda_affect_distributions.png"))
    plt.close()
    print("  [OK] eda_affect_distributions.png")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Student Interaction Sequence Length Distribution
# ──────────────────────────────────────────────────────────────────────────────
def plot_sequence_lengths():
    clean_path = os.path.join(PROC, "clean_interactions.csv")
    if not os.path.exists(clean_path):
        print("  [SKIP] clean_interactions.csv not found")
        return

    print("  Loading clean_interactions.csv for sequence lengths ...")
    df = pd.read_csv(clean_path, usecols=["user_id"], nrows=500_000)
    seq_lens = df.groupby("user_id").size()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(seq_lens, bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Number of Interactions per Student")
    axes[0].set_ylabel("Number of Students")
    axes[0].set_title("Student Sequence Length Distribution")
    axes[0].axvline(seq_lens.mean(), color="red", linestyle="--", linewidth=1.5,
                    label=f"Mean = {seq_lens.mean():.0f}")
    axes[0].axvline(seq_lens.median(), color="orange", linestyle="--", linewidth=1.5,
                    label=f"Median = {seq_lens.median():.0f}")
    axes[0].legend()
    axes[0].yaxis.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_axisbelow(True)

    # CDF
    sorted_lens = np.sort(seq_lens.values)
    cdf = np.arange(1, len(sorted_lens) + 1) / len(sorted_lens)
    axes[1].plot(sorted_lens, cdf, color="#4C72B0", linewidth=1.8)
    axes[1].axvline(50, color="red", linestyle="--", linewidth=1.5,
                    label="Window T=50")
    axes[1].set_xlabel("Sequence Length")
    axes[1].set_ylabel("Cumulative Fraction of Students")
    axes[1].set_title("CDF of Student Sequence Lengths")
    axes[1].legend()
    axes[1].yaxis.grid(True, linestyle="--", alpha=0.4)
    axes[1].set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "eda_sequence_lengths.png"))
    plt.close()
    print("  [OK] eda_sequence_lengths.png")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Skill Coverage (Top 20 Skills by Interaction Count)
# ──────────────────────────────────────────────────────────────────────────────
def plot_skill_coverage():
    feat_path = os.path.join(PROC, "featured_interactions.csv")
    if not os.path.exists(feat_path):
        feat_path = os.path.join(PROC, "clean_interactions.csv")
    if not os.path.exists(feat_path):
        print("  [SKIP] No feature/clean csv found")
        return

    print("  Loading data for skill coverage ...")
    df = pd.read_csv(feat_path, usecols=["skill_id", "skill"], nrows=300_000)
    top = df.groupby("skill").size().nlargest(20).sort_values()

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(top.index, top.values, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Number of Interactions")
    ax.set_title("Top 20 Skills by Interaction Count")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, top.values):
        ax.text(bar.get_width() + top.values.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "eda_skill_coverage.png"))
    plt.close()
    print("  [OK] eda_skill_coverage.png")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Correctness Rate Distribution across Students
# ──────────────────────────────────────────────────────────────────────────────
def plot_correctness_distribution():
    clean_path = os.path.join(PROC, "clean_interactions.csv")
    if not os.path.exists(clean_path):
        print("  [SKIP] clean_interactions.csv not found")
        return

    print("  Computing correctness distribution ...")
    df = pd.read_csv(clean_path, usecols=["user_id", "correct"], nrows=500_000)
    student_acc = df.groupby("user_id")["correct"].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(student_acc, bins=50, color="#55A868", edgecolor="white", alpha=0.85)
    ax.axvline(student_acc.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {student_acc.mean():.2f}")
    ax.axvline(student_acc.median(), color="orange", linestyle="--", linewidth=1.5,
               label=f"Median = {student_acc.median():.2f}")
    ax.set_xlabel("Per-Student Correctness Rate")
    ax.set_ylabel("Number of Students")
    ax.set_title("Distribution of Student Correctness Rates")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "eda_correctness_distribution.png"))
    plt.close()
    print("  [OK] eda_correctness_distribution.png")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Hint Usage Distribution
# ──────────────────────────────────────────────────────────────────────────────
def plot_hint_usage():
    clean_path = os.path.join(PROC, "clean_interactions.csv")
    if not os.path.exists(clean_path):
        print("  [SKIP] clean_interactions.csv not found")
        return

    print("  Computing hint usage ...")
    df = pd.read_csv(clean_path, usecols=["hint_count"], nrows=500_000)
    df["hint_count"] = df["hint_count"].fillna(0)

    counts = df["hint_count"].value_counts().sort_index()
    top = counts.iloc[:8]  # Show 0-7 hints

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(top.index.astype(str), top.values, color="#C44E52", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Number of Hints Requested")
    ax.set_ylabel("Number of Interactions")
    ax.set_title("Hint Request Frequency per Interaction")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    for i, (x, y) in enumerate(zip(top.index, top.values)):
        ax.text(i, y + top.values.max() * 0.01, f"{y:,}", ha="center", fontsize=9)
    pct_no_hint = top.get(0, 0) / df.shape[0] * 100
    ax.text(0.98, 0.95, f"{pct_no_hint:.1f}% use\nno hints",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="gray"))
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "eda_hint_usage.png"))
    plt.close()
    print("  [OK] eda_hint_usage.png")


# ──────────────────────────────────────────────────────────────────────────────
# 7. ACDT Pipeline Architecture Diagram
# ──────────────────────────────────────────────────────────────────────────────
def plot_pipeline_architecture():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    phases = [
        ("Raw Interaction\nTelemetry\n(6.1M rows)", 0.8, "#D9E8F5"),
        ("Feature\nEngineering\n& Tensors", 2.5, "#FDEBD0"),
        ("Dual-State\nStudent Encoder\n(Transformer+LSTM)", 4.4, "#D5E8D4"),
        ("Probabilistic\nWorld Model\n(MDN)", 6.4, "#E1D5E7"),
        ("MCTS\nPlanner\n(UCB1, K=3)", 8.3, "#FFE6CC"),
        ("LLM Tutor\n(OpenRouter\nFallback Chain)", 10.2, "#F8CECC"),
        ("MERN\nDashboard\n(SSE Stream)", 12.1, "#DAE8FC"),
    ]

    box_w, box_h = 1.5, 1.6
    for label, cx, color in phases:
        rect = mpatches.FancyBboxPatch(
            (cx - box_w / 2, 1.7), box_w, box_h,
            boxstyle="round,pad=0.12", linewidth=1.2,
            edgecolor="#444", facecolor=color
        )
        ax.add_patch(rect)
        ax.text(cx, 2.5, label, ha="center", va="center",
                fontsize=8, fontweight="bold", wrap=True,
                multialignment="center")

    # Arrows
    arrow_xs = [(p[1] + box_w / 2 + 0.02) for p in phases[:-1]]
    for x in arrow_xs:
        ax.annotate("", xy=(x + 0.3, 2.5), xytext=(x, 2.5),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="#444"))

    # Labels underneath
    sublabels = [
        "ASSISTments 2012-13\nwith affect annotations",
        "hint_ratio, z-score\nlatency, attempts",
        "z_t = [z_cog ⊕ z_aff]\n128-dim",
        "P(z_{t+1}|z_t,a_t)\n+uncertainty σ",
        "53,091 candidate\nproblems",
        "Gemini→LLaMA\n→Claude→Mistral",
        "Student View\nTeacher View\nLive Demo",
    ]
    for label, (_, cx, __) in zip(sublabels, phases):
        ax.text(cx, 1.5, label, ha="center", va="top", fontsize=7, color="#555",
                multialignment="center")

    ax.set_title("ACDT End-to-End System Architecture", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "acdt_architecture_diagram.png"))
    plt.close()
    print("  [OK] acdt_architecture_diagram.png")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating EDA and architecture images for report Chapter 3 & 4...")
    plot_dataset_summary()
    plot_affect_distribution()
    plot_sequence_lengths()
    plot_skill_coverage()
    plot_correctness_distribution()
    plot_hint_usage()
    plot_pipeline_architecture()
    print(f"\nAll images saved to: {RESULTS}")

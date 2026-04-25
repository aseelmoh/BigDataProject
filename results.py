"""
results.py
─────────────────────────────────────────
  6. Model Comparison Bar Chart
  7. Training Time Bar Chart
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def visualize_results(results, output_dir="reports/figures"):
    os.makedirs(output_dir, exist_ok=True)

    model_names = list(results.keys())
    metrics     = ["AUC-ROC", "Accuracy", "F1 Score", "Precision", "Recall"]
    colors      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    # ──────────────────────────
    x     = np.arange(len(model_names))
    width = 0.15

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [results[m][metric] for m in model_names]
        ax.bar(x + i * width, vals, width, label=metric, color=color, alpha=0.85)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Model Comparison — CS4074 Diabetes Readmission",
        fontsize=13, fontweight="bold"
    )
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/06_model_comparison.png", dpi=150)
    plt.close()
    print("   📊 Saved: 06_model_comparison.png")

    # ───────────────────────────────
    times = [results[m]["Train Time (s)"] for m in model_names]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(
        model_names, times,
        color=["#4C72B0", "#DD8452", "#55A868"],
        edgecolor="white", width=0.5
    )
    for bar, t in zip(bars, times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{t}s", ha="center", fontsize=10
        )
    plt.title("Training Time per Model", fontsize=13, fontweight="bold")
    plt.ylabel("Seconds")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/07_training_time.png", dpi=150)
    plt.close()
    print("   📊 Saved: 07_training_time.png")

    print(f"\n   ✅ Result figures saved to → {output_dir}/")


def print_summary(results):

    print("\n" + "═" * 72)
    print("   FINAL RESULTS SUMMARY")
    print("═" * 72)
    print(f"{'Model':<22} {'AUC':>7} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Time':>8}")
    print("─" * 72)

    for name, m in results.items():
        print(
            f"{name:<22} "
            f"{m['AUC-ROC']:>7.4f} "
            f"{m['Accuracy']:>8.4f} "
            f"{m['F1 Score']:>8.4f} "
            f"{m['Precision']:>8.4f} "
            f"{m['Recall']:>8.4f} "
            f"{m['Train Time (s)']:>7.1f}s"
        )

    print("═" * 72)
    best = max(results, key=lambda k: results[k]["F1 Score"])
    print(f"\n🏆 Best Model (by F1 Score): {best}")
    print(f"   AUC-ROC  : {results[best]['AUC-ROC']}")
    print(f"   Accuracy : {results[best]['Accuracy']}")
    print(f"   F1 Score : {results[best]['F1 Score']}")
    print("═" * 72)

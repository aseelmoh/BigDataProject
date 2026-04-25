"""
eda.py

Responsible for: Exploratory Data Analysis (EDA)

Generated visualizations:

1. Target variable distribution (`readmitted`)
2. Age distribution
3. Missing values percentage for each column
4. Distribution of numerical features (histograms)
5. Correlation heatmap

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(df, output_dir="reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    print("\n🔍 Running EDA...")

    # ──  1: Target ──────────────────────────
    readmit_pd = (
        df.groupBy("readmitted")
          .count()
          .orderBy("count", ascending=False)
          .toPandas()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Readmission Class Distribution", fontsize=14, fontweight="bold")

    axes[0].pie(
        readmit_pd["count"],
        labels=readmit_pd["readmitted"],
        autopct="%1.1f%%",
        colors=["#4C72B0", "#DD8452", "#55A868"],
        startangle=140
    )
    axes[0].set_title("Pie Chart")

    axes[1].bar(
        readmit_pd["readmitted"],
        readmit_pd["count"],
        color=["#4C72B0", "#DD8452", "#55A868"]
    )
    axes[1].set_title("Bar Chart")
    axes[1].set_xlabel("Readmission Class")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_target_distribution.png", dpi=150)
    plt.close()
    print("   📊 Saved: 01_target_distribution.png")

    # ── 2. ─────────────────────────────
    age_pd = (
        df.groupBy("age")
          .count()
          .orderBy("age")
          .toPandas()
    )

    plt.figure(figsize=(11, 4))
    plt.bar(age_pd["age"], age_pd["count"], color="#4C72B0", edgecolor="white")
    plt.title("Patient Age Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Age Group")
    plt.ylabel("Number of Patients")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_age_distribution.png", dpi=150)
    plt.close()
    print("   📊 Saved: 02_age_distribution.png")

    # ── 3. ─────────
    pdf = df.toPandas().replace("?", np.nan)
    missing_pct = (
        pdf.isnull()
           .mean()
           .sort_values(ascending=False)
    )
    missing_pct = missing_pct[missing_pct > 0]

    plt.figure(figsize=(12, 5))
    missing_pct.plot(kind="bar", color="#DD8452", edgecolor="white")
    plt.axhline(y=0.4, color="red", linestyle="--", linewidth=1.5, label="40% threshold")
    plt.title("Missing Value Rate per Column", fontsize=13, fontweight="bold")
    plt.ylabel("Missing Fraction")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_missing_values.png", dpi=150)
    plt.close()
    print("   📊 Saved: 03_missing_values.png")

    # ──4. Histograms ────────────────
    num_cols = [
        "time_in_hospital", "num_lab_procedures",
        "num_medications", "number_inpatient"
    ]
    existing = [c for c in num_cols if c in pdf.columns]

    pdf[existing].hist(
        figsize=(12, 5),
        bins=30,
        color="#4C72B0",
        edgecolor="white",
        layout=(1, len(existing))
    )
    plt.suptitle("Numeric Feature Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_numeric_distributions.png", dpi=150)
    plt.close()
    print("   📊 Saved: 04_numeric_distributions.png")

    # ── 5: Correlation Heatmap ───────────────────────
    corr_cols = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient",
        "number_emergency", "number_inpatient", "number_diagnoses"
    ]
    existing_corr = [c for c in corr_cols if c in pdf.columns]
    corr_matrix = pdf[existing_corr].apply(
        lambda col: col.astype(float, errors="ignore")
    ).corr()

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        corr_matrix,
        annot=True, fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        square=True
    )
    plt.title("Correlation Heatmap – Numeric Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_correlation_heatmap.png", dpi=150)
    plt.close()
    print("   📊 Saved: 05_correlation_heatmap.png")

    print(f"\n   ✅ All 5 EDA figures saved to → {output_dir}/")

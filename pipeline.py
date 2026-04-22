"""
CS4074 – Big Data Analytics
Clinical Outcome Prediction from Noisy Medical Records
Hospital Readmission of Diabetic Patients

Authors: [Your Team Names]
Instructor: Dr. Naila Marir | Effat University | Spring 2026
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler,
    StandardScaler, Imputer
)
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time

# ─────────────────────────────────────────────
# 1. SPARK SESSION INITIALIZATION
# ─────────────────────────────────────────────

def create_spark_session():
    """Create and configure the Spark session."""
    spark = (
        SparkSession.builder
        .appName("CS4074_DiabetesReadmission")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"✅ Spark session created — version: {spark.version}")
    return spark


# ─────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────

def load_data(spark, path="diabetic_data.csv"):
    """Load the dataset with Spark."""
    print("\n📂 Loading dataset...")
    df = spark.read.csv(path, header=True, inferSchema=True)
    print(f"   Rows: {df.count():,}  |  Columns: {len(df.columns)}")
    return df


# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────

def run_eda(df, output_dir="reports/figures"):
    """Perform EDA and save plots."""
    os.makedirs(output_dir, exist_ok=True)
    print("\n🔍 Running EDA...")

    # --- Class distribution
    readmit_counts = (
        df.groupBy("readmitted")
          .count()
          .orderBy("count", ascending=False)
          .toPandas()
    )
    print("\n  Readmission distribution:")
    print(readmit_counts.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("EDA – Readmission of Diabetic Patients", fontsize=14, fontweight="bold")

    # Pie chart
    axes[0].pie(
        readmit_counts["count"],
        labels=readmit_counts["readmitted"],
        autopct="%1.1f%%",
        startangle=140,
        colors=["#4C72B0", "#DD8452", "#55A868"]
    )
    axes[0].set_title("Readmission Classes")

    # Bar chart — top diagnoses
    diag_counts = (
        df.groupBy("diag_1")
          .count()
          .orderBy("count", ascending=False)
          .limit(10)
          .toPandas()
    )
    axes[1].barh(
        diag_counts["diag_1"].astype(str),
        diag_counts["count"],
        color="#4C72B0"
    )
    axes[1].set_xlabel("Patient Count")
    axes[1].set_title("Top 10 Primary Diagnoses")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/eda_overview.png", dpi=150)
    plt.close()

    # --- Age distribution
    age_counts = (
        df.groupBy("age")
          .count()
          .orderBy("age")
          .toPandas()
    )
    plt.figure(figsize=(10, 4))
    sns.barplot(data=age_counts, x="age", y="count", palette="Blues_d")
    plt.title("Patient Age Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/age_distribution.png", dpi=150)
    plt.close()

    # --- Missing value heatmap
    missing_pdf = df.toPandas()
    missing_pct = (
        missing_pdf.replace("?", np.nan)
                   .isnull()
                   .mean()
                   .sort_values(ascending=False)
                   .head(20)
    )
    plt.figure(figsize=(10, 5))
    missing_pct.plot(kind="bar", color="#DD8452")
    plt.title("Top 20 Columns by Missing / '?' Rate")
    plt.ylabel("Missing Fraction")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/missing_values.png", dpi=150)
    plt.close()

    print(f"   Figures saved to {output_dir}/")
    return readmit_counts


# ─────────────────────────────────────────────
# 4. DATA CLEANING & PREPROCESSING
# ─────────────────────────────────────────────
def clean_and_preprocess(df):

    print("\n🧹 Cleaning & preprocessing...")

    df = df.replace("?", None)

    """
    Data cleaning pipeline:
      - Replace '?' placeholders with null
      - Drop columns with >40% missing valuz      - Drop irrelevant ID columns
      - Encode the target variable
      - Cast numerics
    """

    # Replace '?' sentinel values with null
    string_cols = [field.name for field in df.schema.fields if field.dataType.simpleString() == "string"]

    for col_name in string_cols:
        df = df.withColumn(
            col_name,
            F.when(F.col(col_name) == "?", None).otherwise(F.col(col_name))
    )


    # Drop high-missing columns (>40% null)
    total = df.count()
    cols_to_drop = []
    for col_name in df.columns:
        try:
            null_count = df.filter(F.col(col_name).isNull()).count()

            if null_count / total > 0.40:
                cols_to_drop.append(col_name)

        except:
            continue
        if null_count / total > 0.40:
            cols_to_drop.append(col_name)
    print(f"   Dropping high-missing columns: {cols_to_drop}")
    df = df.drop(*cols_to_drop)

    # Drop identifiers that add no signal
    id_cols = ["encounter_id", "patient_nbr"]
    df = df.drop(*[c for c in id_cols if c in df.columns])

    # Remove rows with null in critical columns
    critical_cols = ["gender", "age", "admission_type_id", "readmitted"]
    df = df.dropna(subset=critical_cols)

    # Drop invalid gender entries
    df = df.filter(F.col("gender") != "Unknown/Invalid")

    # Remove duplicate encounters – keep first per patient
    df = df.dropDuplicates(["encounter_id"]) if "encounter_id" in df.columns else df

    # ── Target variable: binary readmission (1 = readmitted within 30 days)
    df = df.withColumn(
        "label",
        F.when(F.col("readmitted") == "<30", 1).otherwise(0).cast(IntegerType())
    )
    df = df.drop("readmitted")

    # ── Cast numeric columns
    numeric_cols = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast(DoubleType()))

    print(f"   Cleaned rows: {df.count():,}  |  Columns remaining: {len(df.columns)}")
    return df


# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df):
    """
    Feature engineering:
      - Encode age bracket as ordinal integer
      - Create composite features
      - Aggregate medication change counts
    """
    print("\n⚙️  Engineering features...")

    # ── Age → ordinal encoding
    age_map = {
        "[0-10)": 1, "[10-20)": 2, "[20-30)": 3, "[30-40)": 4,
        "[40-50)": 5, "[50-60)": 6, "[60-70)": 7, "[70-80)": 8,
        "[80-90)": 9, "[90-100)": 10
    }
    mapping_expr = F.create_map([F.lit(x) for pair in age_map.items() for x in pair])
    df = df.withColumn("age_ordinal", mapping_expr[F.col("age")].cast(DoubleType()))
    df = df.drop("age")

    # ── Total service utilisation (engineered composite)
    df = df.withColumn(
        "total_visits",
        F.col("number_outpatient") + F.col("number_emergency") + F.col("number_inpatient")
    )

    # ── Procedure burden
    df = df.withColumn(
        "procedure_burden",
        F.col("num_procedures") + F.col("num_lab_procedures")
    )

    # ── Medication change flag columns → count of changes
    med_change_cols = [
        c for c in df.columns
        if c not in ["label", "age_ordinal"] and df.schema[c].dataType.typeName() == "string"
        and c not in ["gender", "race", "diag_1", "diag_2", "diag_3",
                      "admission_type_id", "discharge_disposition_id",
                      "admission_source_id", "payer_code", "medical_specialty",
                      "max_glu_serum", "A1Cresult", "change", "diabetesMed"]
    ]
    # Count medication changes (value == 'Ch' means changed)
    change_expr = sum(
        F.when(F.col(c) == "Ch", 1).otherwise(0)
        for c in med_change_cols
    ) if med_change_cols else F.lit(0)
    df = df.withColumn("num_med_changes", change_expr.cast(DoubleType()))

    # ── Binary: any medication changed
    df = df.withColumn(
        "any_med_change",
        F.when(F.col("change") == "Ch", 1.0).otherwise(0.0)
    )

    # ── Binary: on diabetes medication
    df = df.withColumn(
        "on_diabetes_med",
        F.when(F.col("diabetesMed") == "Yes", 1.0).otherwise(0.0)
    )

    print(f"   Features after engineering: {len(df.columns)}")
    return df


# ─────────────────────────────────────────────
# 6. SPARK ML PIPELINE
# ─────────────────────────────────────────────

def build_ml_pipeline(df):
    """
    Build and return train/test splits + feature column list.
    Handles:
      - StringIndexer for categoricals
      - OneHotEncoder
      - VectorAssembler
      - StandardScaler
    """
    print("\n🔧 Building ML feature pipeline...")

    categorical_cols = ["gender", "race", "max_glu_serum", "A1Cresult"]
    # Keep only columns that exist in df
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    numeric_cols = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses",
        "age_ordinal", "total_visits", "procedure_burden",
        "num_med_changes", "any_med_change", "on_diabetes_med"
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    # Null-fill numerics
    imputer = Imputer(
        inputCols=numeric_cols,
        outputCols=[f"{c}_imp" for c in numeric_cols],
        strategy="median"
    )
    imputed_numeric = [f"{c}_imp" for c in numeric_cols]

    # Encode categoricals
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in categorical_cols
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
        for c in categorical_cols
    ]
    ohe_cols = [f"{c}_ohe" for c in categorical_cols]

    assembler = VectorAssembler(
        inputCols=imputed_numeric + ohe_cols,
        outputCol="features_raw",
        handleInvalid="keep"
    )
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=False
    )

    prep_stages = indexers + encoders + [imputer, assembler, scaler]

    # Train / Test split (80/20, stratified by label)
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print(f"   Train: {train_df.count():,}  |  Test: {test_df.count():,}")

    return prep_stages, train_df, test_df


# ─────────────────────────────────────────────
# 7. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────

def train_and_evaluate(prep_stages, train_df, test_df):
    """Train LR, DT, RF models and return metrics."""

    models_config = {
        "Logistic Regression": LogisticRegression(
            featuresCol="features", labelCol="label",
            maxIter=20, regParam=0.01
        ),
        "Decision Tree": DecisionTreeClassifier(
            featuresCol="features", labelCol="label",
            maxDepth=6, seed=42
        ),
        "Random Forest": RandomForestClassifier(
            featuresCol="features", labelCol="label",
            numTrees=50, maxDepth=8, seed=42
        ),
    }

    binary_eval = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    mc_eval_acc  = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    mc_eval_f1   = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")
    mc_eval_prec = MulticlassClassificationEvaluator(labelCol="label", metricName="weightedPrecision")
    mc_eval_rec  = MulticlassClassificationEvaluator(labelCol="label", metricName="weightedRecall")

    results = {}
    trained_models = {}

    for name, classifier in models_config.items():
        print(f"\n🚀 Training: {name}...")
        start = time.time()

        pipeline = Pipeline(stages=prep_stages + [classifier])
        model = pipeline.fit(train_df)
        elapsed = time.time() - start

        predictions = model.transform(test_df)

        auc      = binary_eval.evaluate(predictions)
        accuracy = mc_eval_acc.evaluate(predictions)
        f1       = mc_eval_f1.evaluate(predictions)
        precision= mc_eval_prec.evaluate(predictions)
        recall   = mc_eval_rec.evaluate(predictions)

        results[name] = {
            "AUC-ROC":   round(auc, 4),
            "Accuracy":  round(accuracy, 4),
            "F1 Score":  round(f1, 4),
            "Precision": round(precision, 4),
            "Recall":    round(recall, 4),
            "Train Time (s)": round(elapsed, 2)
        }
        trained_models[name] = model

        print(f"   AUC: {auc:.4f} | Accuracy: {accuracy:.4f} | F1: {f1:.4f} | Time: {elapsed:.1f}s")

    return results, trained_models


# ─────────────────────────────────────────────
# 8. RESULTS VISUALIZATION
# ─────────────────────────────────────────────

def visualize_results(results, output_dir="reports/figures"):
    """Plot model comparison charts."""
    os.makedirs(output_dir, exist_ok=True)

    metrics = ["AUC-ROC", "Accuracy", "F1 Score", "Precision", "Recall"]
    model_names = list(results.keys())

    data = {
        m: [results[model][m] for model in model_names]
        for m in metrics
    }

    x = np.arange(len(model_names))
    width = 0.15
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (metric, vals) in enumerate(data.items()):
        ax.bar(x + i * width, vals, width, label=metric, color=colors[i], alpha=0.85)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison – CS4074 Diabetes Readmission Project", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=150)
    plt.close()

    # Training time
    times = [results[m]["Train Time (s)"] for m in model_names]
    plt.figure(figsize=(8, 4))
    plt.bar(model_names, times, color=["#4C72B0", "#DD8452", "#55A868"])
    plt.title("Training Time by Model")
    plt.ylabel("Seconds")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_time.png", dpi=150)
    plt.close()

    print(f"\n📊 Result plots saved to {output_dir}/")


# ─────────────────────────────────────────────
# 9. PRINT SUMMARY TABLE
# ─────────────────────────────────────────────

def print_summary(results):
    print("\n" + "═" * 70)
    print("  FINAL RESULTS SUMMARY")
    print("═" * 70)
    header = f"{'Model':<22} {'AUC':>7} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Time':>7}"
    print(header)
    print("─" * 70)
    for name, m in results.items():
        print(
            f"{name:<22} {m['AUC-ROC']:>7.4f} {m['Accuracy']:>8.4f} "
            f"{m['F1 Score']:>8.4f} {m['Precision']:>8.4f} "
            f"{m['Recall']:>8.4f} {m['Train Time (s)']:>6.1f}s"
        )
    print("═" * 70)
    best = max(results, key=lambda k: results[k]["F1 Score"])
    print(f"\n🏆 Best Model (by F1): {best}  —  F1={results[best]['F1 Score']:.4f}")


# ─────────────────────────────────────────────
# 10. MAIN ENTRY POINT
# ─────────────────────────────────────────────

def main():
    spark = create_spark_session()

    # Load
    df = load_data(spark, path="diabetic_data.csv")

    # EDA
    run_eda(df)

    # Clean
    df = clean_and_preprocess(df)

    # Feature engineering
    df = engineer_features(df)

    # Persist in memory for reuse
    df.cache()
    df.count()  # trigger caching

    # Build ML prep stages + splits
    prep_stages, train_df, test_df = build_ml_pipeline(df)

    # Train & evaluate
    results, trained_models = train_and_evaluate(prep_stages, train_df, test_df)

    # Visualize
    visualize_results(results)

    # Summary
    print_summary(results)

    spark.stop()
    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()

"""
main.py
────────────────────────────────────────
────────────────────────────────────────
CS4074 – Big Data Analytics
Clinical Outcome Prediction from Noisy Medical Records
Effat University | Spring 2026 | Dr. Naila Marir
"""

from src.spark_session      import create_spark_session
from src.data_loader        import load_data
from src.eda                import run_eda
from src.data_cleaning      import clean_data
from src.feature_engineering import engineer_features
from src.model_training     import build_preprocessing_stages, split_data, train_all_models
from src.results            import visualize_results, print_summary


def main():
    print("=" * 55)
    print("  CS4074 — Diabetes Readmission Prediction Pipeline")
    print("=" * 55)

    # ── Step 1: Spark ──────────────────────────────────────
    spark = create_spark_session()

    # ── Step 2: Load ───────────────────────────────────────
    df = load_data(spark, path="data/diabetic_data.csv")

    # ── Step 3: EDA ────────────────────────────────────────
    run_eda(df)

    # ── Step 4: Clean ──────────────────────────────────────
    df = clean_data(df)

    # ── Step 5: Feature Engineering ────────────────────────
    df = engineer_features(df)

    # Cache 
    df.cache()
    df.count()

    # ── Step 6: Train & Evaluate ───────────────────────────
    prep_stages      = build_preprocessing_stages(df)
    train_df, test_df = split_data(df)
    results          = train_all_models(prep_stages, train_df, test_df)

    # ── Step 7: Results ────────────────────────────────────
    print("\n📊 Generating result visualizations...")
    visualize_results(results)
    print_summary(results)

    spark.stop()
    print("\n✅ Pipeline complete. Check reports/figures/ for all plots.")


if __name__ == "__main__":
    main()

"""
data_loader.py
─────────────────────────────────────────
 CSV
"""


def load_data(spark, path="data/diabetic_data.csv"):
    print("\n📂 Loading dataset...")

    df = spark.read.csv(path, header=True, inferSchema=True)

    print(f"   ✅ Rows    : {df.count():,}")
    print(f"   ✅ Columns : {len(df.columns)}")

    return df

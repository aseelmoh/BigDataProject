"""
data_cleaning.py
─────────────────────────────────────────
"""

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

def clean_data(df):
    print("\n🧹 Cleaning data...")

    total_before = df.count()

    # ── 1.'?'ـ null ──────────────────────────────
    for col_name in df.columns:
        df = df.withColumn(
            col_name,
            F.when(F.col(col_name) == "?", None).otherwise(F.col(col_name))
        )
    print("   ✅ Step 1: Replaced '?' with null")

    # ── 2 ──
    total = df.count()
    cols_to_drop = []
    for col_name in df.columns:
        null_count = df.filter(F.col(col_name).isNull()).count()
        if null_count / total > 0.40:
            cols_to_drop.append(col_name)

    df = df.drop(*cols_to_drop)
    print(f"   ✅ Step 2: Dropped high-missing columns: {cols_to_drop}")

    # ── 3) ───────────
    id_cols = [c for c in ["encounter_id", "patient_nbr"] if c in df.columns]
    df = df.drop(*id_cols)
    print(f"   ✅ Step 3: Dropped ID columns: {id_cols}")

    # ── 4) ───────────
    critical = [c for c in ["gender", "age", "readmitted"] if c in df.columns]
    df = df.dropna(subset=critical)
    print(f"   ✅ Step 4: Dropped rows with null in critical columns")

    # ── 5.────────────────────
    df = df.filter(F.col("gender") != "Unknown/Invalid")
    print("   ✅ Step 5: Removed invalid gender entries")

    # ── 6. ────────────────────────────
    df = df.dropDuplicates()
    print("   ✅ Step 6: Removed duplicate rows")

    # ── 7.  target to binary label ───────────────
    df = df.withColumn(
        "label",
        F.when(F.col("readmitted") == "<30", 1)
         .otherwise(0)
         .cast(IntegerType())
    )
    df = df.drop("readmitted")
    print("   ✅ Step 7: Encoded target → label (1 = readmitted <30 days)")

    total_after = df.count()
    removed = total_before - total_after
    print(f"\n   📋 Before : {total_before:,} rows")
    print(f"   📋 After  : {total_after:,} rows")
    print(f"   📋 Removed: {removed:,} rows")

    return df

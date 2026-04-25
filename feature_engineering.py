"""
feature_engineering.py
─────────────────────────────────────────
features
"""

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType


def engineer_features(df):
    print("\n⚙️  Engineering features...")

    # ── 1. Age → bracket ───────────────
    age_map = {
        "[0-10)": 1,  "[10-20)": 2, "[20-30)": 3,
        "[30-40)": 4, "[40-50)": 5, "[50-60)": 6,
        "[60-70)": 7, "[70-80)": 8, "[80-90)": 9, "[90-100)": 10
    }
    mapping_expr = F.create_map(
        [F.lit(x) for pair in age_map.items() for x in pair]
    )
    df = df.withColumn("age_ordinal", mapping_expr[F.col("age")].cast(DoubleType()))
    df = df.drop("age")
    print("   ✅ Feature 1: age_ordinal  (age bracket → 1 to 10)")

    # ── 2. ───────────────────────────
    df = df.withColumn(
        "total_visits",
        (F.col("number_outpatient") +
         F.col("number_emergency") +
         F.col("number_inpatient")).cast(DoubleType())
    )
    print("   ✅ Feature 2: total_visits  (outpatient + emergency + inpatient)")

    # ── 3. ───────────────────────────
    df = df.withColumn(
        "procedure_burden",
        (F.col("num_procedures") +
         F.col("num_lab_procedures")).cast(DoubleType())
    )
    print("   ✅ Feature 3: procedure_burden  (procedures + lab procedures)")

    # ── 4. ───────────────────────────────
    df = df.withColumn(
        "any_med_change",
        F.when(F.col("change") == "Ch", 1.0).otherwise(0.0)
    )
    print("   ✅ Feature 4: any_med_change  (1 = medication changed)")

    # ── 5. ────────────────────
    df = df.withColumn(
        "on_diabetes_med",
        F.when(F.col("diabetesMed") == "Yes", 1.0).otherwise(0.0)
    )
    print("   ✅ Feature 5: on_diabetes_med  (1 = on diabetes medication)")

    print(f"\n   📋 Total columns after engineering: {len(df.columns)}")
    return df

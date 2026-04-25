"""
spark_session.py
─────────────────────────────────────────
Spark Session
"""

from pyspark.sql import SparkSession


def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("CS4074_DiabetesReadmission")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"✅ Spark Session created — version: {spark.version}")
    return spark

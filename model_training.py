"""
model_training.py
─────────────────────────────────────────
"""

import time
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder,
    VectorAssembler, StandardScaler, Imputer
)
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)


def build_preprocessing_stages(df):


    categorical_cols = [
        c for c in ["gender", "race", "max_glu_serum", "A1Cresult"]
        if c in df.columns
    ]

    numeric_cols = [
        c for c in [
            "time_in_hospital", "num_lab_procedures", "num_procedures",
            "num_medications", "number_outpatient", "number_emergency",
            "number_inpatient", "number_diagnoses",
            "age_ordinal", "total_visits", "procedure_burden",
            "any_med_change", "on_diabetes_med"
        ]
        if c in df.columns
    ]

    # Imputer
    imputer = Imputer(
        inputCols=numeric_cols,
        outputCols=[f"{c}_imp" for c in numeric_cols],
        strategy="median"
    )

    # StringIndexer
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in categorical_cols
    ]

    # OneHotEncoder
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
        for c in categorical_cols
    ]

    # VectorAssembler
    assembler = VectorAssembler(
        inputCols=[f"{c}_imp" for c in numeric_cols] +
                  [f"{c}_ohe" for c in categorical_cols],
        outputCol="features_raw",
        handleInvalid="keep"
    )

    # StandardScaler →
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=False
    )

    stages = indexers + encoders + [imputer, assembler, scaler]
    return stages


def split_data(df):
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print(f"\n✂️  Data Split:")
    print(f"   Train : {train_df.count():,} rows")
    print(f"   Test  : {test_df.count():,} rows")
    return train_df, test_df


def train_all_models(prep_stages, train_df, test_df):
    
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

    binary_eval = BinaryClassificationEvaluator(
        labelCol="label", metricName="areaUnderROC"
    )
    acc_eval  = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    f1_eval   = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")
    prec_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="weightedPrecision")
    rec_eval  = MulticlassClassificationEvaluator(labelCol="label", metricName="weightedRecall")

    results = {}

    for name, classifier in models_config.items():
        print(f"\n🚀 Training: {name}...")
        start = time.time()

        pipeline = Pipeline(stages=prep_stages + [classifier])
        model    = pipeline.fit(train_df)
        preds    = model.transform(test_df)
        elapsed  = time.time() - start

        results[name] = {
            "AUC-ROC":        round(binary_eval.evaluate(preds), 4),
            "Accuracy":       round(acc_eval.evaluate(preds), 4),
            "F1 Score":       round(f1_eval.evaluate(preds), 4),
            "Precision":      round(prec_eval.evaluate(preds), 4),
            "Recall":         round(rec_eval.evaluate(preds), 4),
            "Train Time (s)": round(elapsed, 2),
        }

        m = results[name]
        print(f"   AUC : {m['AUC-ROC']}  |  Acc : {m['Accuracy']}  "
              f"|  F1 : {m['F1 Score']}  |  Time : {m['Train Time (s)']}s")

    return results

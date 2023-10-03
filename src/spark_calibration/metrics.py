from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pyspark.sql.functions as F

from pyspark.sql.dataframe import DataFrame


def display_classification_calib_metrics(df: DataFrame):
    """Print pre and post calibration metrics for comparison

    Args:
        df: dataframe with score, label and prediction(calibratied score) columns
    """

    assert (
        "score" in df.columns and "label" in df.columns and "prediction" in df.columns
    ), "score and label columns should be present in the dataframe"

    model_bs = df.select(F.avg(F.pow(df["label"] - df["score"], 2))).collect()[0][0]
    model_ll = df.select(
        F.avg(
            -F.col("label") * F.log(F.col("score"))
            - (1 - F.col("label")) * F.log(1 - F.col("score"))
        )
    ).collect()[0][0]

    model_aucpr = BinaryClassificationEvaluator(
        rawPredictionCol="score", metricName="areaUnderPR"
    ).evaluate(df)
    model_roc_auc = BinaryClassificationEvaluator(
        rawPredictionCol="score", metricName="areaUnderROC"
    ).evaluate(df)
    iso_bs = df.select(F.avg(F.pow(df["label"] - df["prediction"], 2))).collect()[0][0]

    print(f"model brier score loss: {model_bs}")
    print(f"calibrated model brier score loss: {iso_bs}")

    print(f"delta: {round((iso_bs/model_bs - 1) * 100, 2)}%")
    iso_ll = df.select(
        F.avg(
            -F.col("label") * F.log(F.col("prediction"))
            - (1 - F.col("label")) * F.log(1 - F.col("prediction"))
        )
    ).collect()[0][0]

    print("")

    print(f"model log loss: {model_ll}")
    print(f"calibrated model log loss: {iso_ll}")
    print(f"delta: {round((iso_ll/model_ll - 1) * 100, 2)}%")
    iso_aucpr = BinaryClassificationEvaluator(
        rawPredictionCol="prediction", metricName="areaUnderPR"
    ).evaluate(df)

    print("")

    print(f"model aucpr: {model_aucpr}")
    print(f"calibrated model aucpr: {iso_aucpr}")
    print(f"delta: {round((iso_aucpr/model_aucpr - 1) * 100, 2)}%")
    iso_roc_auc = BinaryClassificationEvaluator(
        rawPredictionCol="prediction", metricName="areaUnderROC"
    ).evaluate(df)

    print("")

    print(f"model roc_auc: {model_roc_auc}")
    print(f"calibrated model roc_auc: {iso_roc_auc}")
    print(f"delta: {round((iso_roc_auc/model_roc_auc - 1) * 100, 2)}%")

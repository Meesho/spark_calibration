import pyspark.sql.functions as F

from pyspark.sql.types import DoubleType

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType

from pyspark.sql.dataframe import DataFrame


class Betacal:
    def __init__(self, parameters):
        assert parameters == "abm"

        self.parameters = parameters
        self.a = None
        self.b = None
        self.lr_model = None

    def fit(self, df: DataFrame):
        assert (
            "score" in df.columns and "label" in df.columns
        ), "score and label columns should be present in the dataframe"

        if self.parameters == "abm":
            df = df.withColumn("score2", 1 - F.col("score"))
            df = df.withColumn("score", F.log("score")).withColumn(
                "score2", -1 * F.log("score2")
            )
            lr = LogisticRegression()
            featurizer = VectorAssembler(
                inputCols=["score", "score2"], outputCol="features"
            )

            train_data = featurizer.transform(df)["label", "features"]

            lr_fitted = lr.fit(train_data)

            lr_coef = lr_fitted.coefficients

            if lr_coef[0] < 0:
                featurizer = VectorAssembler(inputCols=["score2"], outputCol="features")
                train_data = featurizer.transform(df)["label", "features"]
                lr_fitted = lr.fit(train_data)
                a = 0
                b = lr_fitted.coefficients[0]

            elif lr_coef[1] < 0:
                featurizer = VectorAssembler(inputCols=["score"], outputCol="features")
                train_data = featurizer.transform(df)["label", "features"]
                lr_fitted = lr.fit(train_data)
                b = 0
                a = lr_fitted.coefficients[0]
            else:
                a = lr_coef[0]
                b = lr_coef[1]

            self.a = a
            self.b = b
            self.lr_model = lr_fitted

    def predict(self, df: DataFrame):
        cols = df.columns

        assert "score" in cols, "score column should be present in the dataframe"

        if self.parameters == "abm":

            def pick_value(v):
                return float(v[1])

            pick_value = F.udf(pick_value, DoubleType())

            df = df.withColumn("orig_score", F.col("score"))

            df = df.withColumn("score2", 1 - F.col("score"))

            df = df.withColumn("score", F.log("score")).withColumn(
                "score2", -1 * F.log("score2")
            )

            if self.a == 0:
                featurizer = VectorAssembler(inputCols=["score2"], outputCol="features")

            elif self.b == 0:
                featurizer = VectorAssembler(inputCols=["score"], outputCol="features")

            else:
                featurizer = VectorAssembler(
                    inputCols=["score", "score2"], outputCol="features"
                )

            test_data = featurizer.transform(df)

            df = (
                self.lr_model.transform(test_data)
                .withColumn("prediction", pick_value("probability"))
                .drop("score")
                .withColumnRenamed("orig_score", "score")
            )

            df = df.select(cols + ["prediction"])

            return df

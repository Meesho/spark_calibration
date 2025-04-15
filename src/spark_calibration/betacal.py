import json
import os
import tempfile
from typing import Optional

import pyspark.sql.functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame


class Betacal:
    """
    Beta calibration using a logistic transformation of raw model scores.

    Formula:
        logit = a * log(score) + b * log(1 - score) + c
        prediction = 1 / (1 + exp(-logit))

    Attributes:
        a (float): Coefficient for log(score)
        b (float): Coefficient for log(1 - score)
        c (float): Intercept
    """

    EPSILON = 1e-12

    def __init__(self, parameters: str = "abm"):
        assert parameters == "abm", "Only 'abm' parameterization is supported."
        self.parameters = parameters
        self.a: Optional[float] = None
        self.b: Optional[float] = None
        self.c: Optional[float] = None

    def _log_expr(self, col: F.Column) -> F.Column:
        """Numerically stable log transformation."""
        return F.log(F.when(col < self.EPSILON, self.EPSILON).otherwise(col))

    def fit(
        self, df: DataFrame, score_col: str = "score", label_col: str = "label"
    ) -> None:
        """
        Fit a beta calibration model using logistic regression.

        Args:
            df (DataFrame): Input dataframe.
            score_col (str): Column containing raw model scores.
            label_col (str): Column containing binary labels.
        """
        assert (
            score_col in df.columns and label_col in df.columns
        ), f"Columns {score_col} and {label_col} must be present."

        log_score = self._log_expr(F.col(score_col))
        log_one_minus_score = self._log_expr(1 - F.col(score_col))

        df_transformed = df.select(
            F.col(label_col).alias("label"),
            log_score.alias("log_score"),
            (-1 * log_one_minus_score).alias("log_score_complement"),
        )

        assembler = VectorAssembler(
            inputCols=["log_score", "log_score_complement"], outputCol="features"
        )
        train_data = assembler.transform(df_transformed).select("label", "features")

        lr = LogisticRegression()
        model = lr.fit(train_data)
        coef = model.coefficients

        # Check if both coefficients are valid
        if coef[0] < 0:
            assembler = VectorAssembler(
                inputCols=["log_score_complement"], outputCol="features"
            )
            train_data = assembler.transform(df_transformed).select("label", "features")
            model = lr.fit(train_data)
            self.a = 0.0
            self.b = float(model.coefficients[0])
        elif coef[1] < 0:
            assembler = VectorAssembler(inputCols=["log_score"], outputCol="features")
            train_data = assembler.transform(df_transformed).select("label", "features")
            model = lr.fit(train_data)
            self.a = float(model.coefficients[0])
            self.b = 0.0
        else:
            self.a = float(coef[0])
            self.b = float(coef[1])

        self.c = float(model.intercept)

    def predict(self, df: DataFrame, score_col: str = "score") -> DataFrame:
        """
        Apply the learned beta calibration model to predict calibrated scores.

        Args:
            df (DataFrame): Input dataframe with raw scores.
            score_col (str): Column name for raw score.

        Returns:
            DataFrame: Original dataframe with an added 'prediction' column.

        Raises:
            ValueError: If calibration coefficients are not set.
        """
        if self.a is None or self.b is None or self.c is None:
            raise ValueError(
                "Model coefficients a, b, and c must be set. Call `.fit()` or `.load()` before prediction."
            )

        assert score_col in df.columns, f"{score_col} must be present."

        log_score = self._log_expr(F.col(score_col))
        log_one_minus_score = self._log_expr(1 - F.col(score_col))

        logit = (
            F.lit(self.a) * log_score
            + F.lit(self.b) * (-1 * log_one_minus_score)
            + F.lit(self.c)
        )
        prediction = 1 / (1 + F.exp(-logit))
        return df.withColumn("prediction", prediction)

    def save(self, path: Optional[str] = None, prefix: str = "betacal_") -> str:
        """
        Save the model coefficients to disk.

        Args:
            path (str, optional): Directory to save into. Creates temp dir if None.
            prefix (str): Prefix for temp folder name if path is None.

        Returns:
            str: The final save path.
        """
        if path is None:
            path = tempfile.mkdtemp(prefix=prefix)

        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "coeffs.json"), "w") as f:
            json.dump(
                {"a": self.a, "b": self.b, "c": self.c, "parameters": self.parameters},
                f,
            )

        return path

    @classmethod
    def load(cls, path: str) -> "Betacal":
        """
        Load model coefficients from disk.

        Args:
            path (str): Directory containing 'coeffs.json'.

        Returns:
            Betacal: The loaded model.
        """
        with open(os.path.join(path, "coeffs.json"), "r") as f:
            coeffs = json.load(f)

        model = cls(parameters=coeffs["parameters"])
        model.a = coeffs["a"]
        model.b = coeffs["b"]
        model.c = coeffs["c"]
        return model

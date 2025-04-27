import json
import logging
import os
import tempfile
from typing import Optional

import pyspark.sql.functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame


def get_logger():
    """Configure logger for Spark environment."""
    logger = logging.getLogger(__name__)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


logger = get_logger()


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

    def get_params(self) -> dict:
        """
        Get the model parameters.

        Returns:
            dict: Dictionary containing model parameters a, b, c.
        """
        return {"a": self.a, "b": self.b, "c": self.c, "parameters": self.parameters}

    def _log_expr(self, col: F.Column) -> F.Column:
        """Numerically stable log transformation."""
        return F.log(F.when(col < self.EPSILON, self.EPSILON).otherwise(col))

    def _validate_input_df(self, df: DataFrame, score_col: str, label_col: str) -> None:
        """
        Validate input DataFrame and required columns.

        Args:
            df (DataFrame): Input dataframe.
            score_col (str): Column containing raw model scores.
            label_col (str): Column containing binary labels.

        Raises:
            ValueError: If DataFrame is empty or required columns are missing.
        """
        if df.count() == 0:
            raise ValueError("Cannot fit model on empty DataFrame")

        assert (
            score_col in df.columns and label_col in df.columns
        ), f"Columns {score_col} and {label_col} must be present."

    def _handle_null_values(self, df: DataFrame, score_col: str) -> DataFrame:
        """
        Handle null values in the score column.

        Args:
            df (DataFrame): Input dataframe.
            score_col (str): Column containing raw model scores.

        Returns:
            DataFrame: Cleaned DataFrame with null values removed.

        Raises:
            ValueError: If all rows contain null values.
        """
        total_rows = df.count()
        df_clean = df.dropna(subset=[score_col])
        rows_after_drop = df_clean.count()

        if rows_after_drop == 0:
            raise ValueError(f"All rows contained null values in {score_col} column")

        dropped_rows = total_rows - rows_after_drop
        if dropped_rows > 0:
            logger.info(
                f"Dropped {dropped_rows}/{total_rows} rows ({(dropped_rows/total_rows)*100:.2f}%) "
                f"with null values in column '{score_col}'"
            )

        return df_clean

    def _prepare_features(
        self, df: DataFrame, score_col: str, label_col: str
    ) -> DataFrame:
        """
        Prepare features for logistic regression with all possible combinations.

        Args:
            df (DataFrame): Input dataframe.
            score_col (str): Column containing raw model scores.
            label_col (str): Column containing binary labels.

        Returns:
            DataFrame: Transformed DataFrame with features ready for training.
            Contains three feature vectors:
            - features_both: Both log(score) and -log(1-score)
            - features_score: Only log(score)
            - features_complement: Only -log(1-score)
        """
        log_score = self._log_expr(F.col(score_col))
        log_one_minus_score = self._log_expr(1 - F.col(score_col))

        df_transformed = df.select(
            F.col(label_col).alias("label"),
            log_score.alias("log_score"),
            (-1 * log_one_minus_score).alias("log_score_complement"),
        )

        # Prepare all possible feature combinations
        assembler_both = VectorAssembler(
            inputCols=["log_score", "log_score_complement"], outputCol="features_both"
        )
        assembler_score = VectorAssembler(
            inputCols=["log_score"], outputCol="features_score"
        )
        assembler_complement = VectorAssembler(
            inputCols=["log_score_complement"], outputCol="features_complement"
        )

        df_with_both = assembler_both.transform(df_transformed)
        df_with_score = assembler_score.transform(df_with_both)
        return assembler_complement.transform(df_with_score)

    def _fit_logistic_regression(self, train_data: DataFrame) -> None:
        """
        Fit logistic regression model and set coefficients.

        Args:
            train_data (DataFrame): Prepared training data with features.
        """
        lr = LogisticRegression()

        # First try with both features
        model = lr.fit(
            train_data.select("label", F.col("features_both").alias("features"))
        )
        coef = model.coefficients

        if coef[0] < 0:
            # Use only complement feature if first coefficient is negative
            model = lr.fit(
                train_data.select(
                    "label", F.col("features_complement").alias("features")
                )
            )
            self.a = 0.0
            self.b = float(model.coefficients[0])
        elif coef[1] < 0:
            # Use only score feature if second coefficient is negative
            model = lr.fit(
                train_data.select("label", F.col("features_score").alias("features"))
            )
            self.a = float(model.coefficients[0])
            self.b = 0.0
        else:
            self.a = float(coef[0])
            self.b = float(coef[1])

        self.c = float(model.intercept)

    def _validate_score_range(self, df: DataFrame, score_col: str) -> None:
        """
        Validate that scores are within valid range (0,1).

        Args:
            df (DataFrame): Input dataframe.
            score_col (str): Column containing raw model scores.

        Raises:
            ValueError: If scores are outside valid range.
        """
        stats = df.select(
            F.min(score_col).alias("min"), F.max(score_col).alias("max")
        ).collect()[0]

        if stats.min < 0 or stats.max > 1:
            raise ValueError(
                f"Scores must be in range [0,1], got range [{stats.min:.3f}, {stats.max:.3f}]"
            )

    def fit(
        self, df: DataFrame, score_col: str = "score", label_col: str = "label"
    ) -> "Betacal":
        """
        Fit a beta calibration model using logistic regression.

        Args:
            df (DataFrame): Input dataframe.
            score_col (str): Column containing raw model scores.
            label_col (str): Column containing binary labels.

        Returns:
            Betacal: The fitted model instance (self).

        Raises:
            ValueError: If input DataFrame is empty or contains all null values.
        """
        self._validate_input_df(df, score_col, label_col)
        self._validate_score_range(df, score_col)
        df_clean = self._handle_null_values(df, score_col)
        train_data = self._prepare_features(df_clean, score_col, label_col)
        self._fit_logistic_regression(train_data)
        return self

    def predict(
        self,
        df: DataFrame,
        score_col: str = "score",
        prediction_col: str = "prediction",
    ) -> DataFrame:
        """
        Apply the learned beta calibration model to predict calibrated scores.

        Args:
            df (DataFrame): Input dataframe with raw scores.
            score_col (str): Column name for raw score.
            prediction_col (str): Name for the output prediction column.

        Returns:
            DataFrame: Original dataframe with an added prediction column.
            Null values in score_col will result in null predictions.

        Raises:
            ValueError: If calibration coefficients are not set or scores are outside valid range.
        """
        if self.a is None or self.b is None or self.c is None:
            raise ValueError(
                "Model coefficients a, b, and c must be set. Call `.fit()` or `.load()` before prediction."
            )

        assert score_col in df.columns, f"{score_col} must be present."

        self._validate_score_range(df.filter(F.col(score_col).isNotNull()), score_col)

        log_score = self._log_expr(F.col(score_col))
        log_one_minus_score = self._log_expr(1 - F.col(score_col))

        logit = (
            F.lit(self.a) * log_score
            + F.lit(self.b) * (-1 * log_one_minus_score)
            + F.lit(self.c)
        )

        prediction = F.when(F.col(score_col).isNull(), None).otherwise(
            1 / (1 + F.exp(-logit))
        )

        return df.withColumn(prediction_col, prediction)

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

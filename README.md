# Model calibration with pyspark

This package provides a Betacal class which allows the user to fit the default beta calibration model and predict calibrated scores


## Setup

spark-calibration is [uploaded to PyPi](https://pypi.org/project/spark-calibration/) and can be installed with this command:

```
pip install spark-calibration
```

## Usage

### Training

train_df should be a pyspark dataframe with `score` and `label` columns

```
from spark_calibration import Betacal
from spark_calibration import display_classification_calib_metrics
from spark_calibration import plot_calibration_curve


bc = Betacal(parameters="abm")

# training
train_df = spark.read.parquet("s3://train/")
bc.fit(train_df)


# Get the logistic regression model and individual coefficients
print(bc.lr_model, a, b)

# a,b -> coefficients of lr model
# lr_model -> pyspark ml logistic regression model
```


### Prediction

test_df is a pyspark dataframe with `score` as one of the columns. The `predict` function adds a new column `prediction` which has the calibrated score

```
test_df = spark.read.parquet("s3://test/")
test_df = bc.predict(test_df)
```


### Pre post calibration metrics comparison

The test_df should have `score`, `prediction` & `label` columns. 
The `display_classification_calib_metrics` functions displays `brier_score_loss`, `log_loss`, `area_under_PR_curve` and `area_under_ROC_curve`
```
display_classification_calib_metrics(test_df)
```


### plot the calibration curve

Computes true, predicted probabilites (pre & post calibration) using quantile binning strategy with 50 bins and plots the calibration curve

```
plot_calibration_curve(test_df)
```
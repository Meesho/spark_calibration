# Model calibration with pyspark

<img width="1069" alt="Screenshot 2023-10-09 at 4 20 38 PM" src="https://github.com/Meesho/spark_calibration/assets/102668625/4d45d9d9-601b-406b-8b8a-55be944d4019">

This package provides a Betacal class which allows the user to fit/train the default beta calibration model on pyspark dataframes and predict calibrated scores


## Setup

spark-calibration package is [uploaded to PyPi](https://pypi.org/project/spark-calibration/) and can be installed with this command:

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


train_df = spark.read.parquet("s3://train/")

bc.fit(train_df) # training

print(bc.lr_model, a, b)
```

a,b -> coefficients of logistic regression model

lr_model -> pysparkML logistic regression model

### Prediction

test_df is a pyspark dataframe with `score` as one of the columns. The `predict` function adds a new column `prediction` which has the calibrated score

```
test_df = spark.read.parquet("s3://test/")
test_df = bc.predict(test_df)
```

### Pre & Post Calibration Classification Metrics

The test_df should have `score`, `prediction` & `label` columns. 
The `display_classification_calib_metrics` functions displays `brier_score_loss`, `log_loss`, `area_under_PR_curve` and `area_under_ROC_curve`
```
display_classification_calib_metrics(test_df)
```
#### Output
```
model brier score loss: 0.08072683729933376
calibrated model brier score loss: 0.01014015353257748
delta: -87.44%

model log loss: 0.3038106859864252
calibrated model log loss: 0.053275633947890755
delta: -82.46%

model aucpr: 0.03471287564672635
calibrated model aucpr: 0.03471240518472563
delta: -0.0%

model roc_auc: 0.7490639506966398
calibrated model roc_auc: 0.7490649764289607
delta: 0.0%
```

### Plot the Calibration Curve

Computes true, predicted probabilites (pre & post calibration) using quantile binning strategy with 50 bins and plots the calibration curve

```
plot_calibration_curve(test_df)
```
<img width="1069" alt="Screenshot 2023-10-09 at 4 20 38 PM" src="https://github.com/Meesho/spark_calibration/assets/102668625/4d45d9d9-601b-406b-8b8a-55be944d4019">

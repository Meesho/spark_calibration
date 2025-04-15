# Model calibration with pyspark

<img width="1069" alt="Screenshot 2023-10-10 at 3 19 39 PM" src="https://github.com/Meesho/spark_calibration/assets/102668625/d66ad0c9-3501-4f15-a73f-9266f7d3ab4c">


This package provides a Betacal class which allows the user to fit/train the default beta calibration model on pyspark dataframes and predict calibrated scores


## Setup

spark-calibration package is [uploaded to PyPi](https://pypi.org/project/spark-calibration/) and can be installed with this command:

```
pip install spark-calibration
```

## Usage

### Training

train_df should be a pyspark dataframe containing:
- A column with raw model scores (default name: `score`)
- A column with binary labels (default name: `label`)

You can specify different column names when calling `fit()`. In some tree-based models like LightGBM, the predicted scores may fall outside the [0, 1] range and can even be negative. Please apply a sigmoid function to normalize the outputs accordingly.

```python
from spark_calibration import Betacal
from spark_calibration import display_classification_calib_metrics
from spark_calibration import plot_calibration_curve

# Initialize model
bc = Betacal(parameters="abm")

# Load training data
train_df = spark.read.parquet("s3://train/")

# Fit the model
bc.fit(train_df)

# Or specify custom column names
# bc.fit(train_df, score_col="raw_score", label_col="actual_label")

# Access model parameters
print(f"Model coefficients: a={bc.a}, b={bc.b}, c={bc.c}")
```

The model learns three parameters:
- a: Coefficient for log(score)
- b: Coefficient for log(1-score) 
- c: Intercept term

### Saving and Loading Models

You can save the trained model to disk and load it later:

```python
# Save model
save_path = bc.save("/path/to/save/")

# Load model
loaded_model = Betacal.load("/path/to/save/")
```

### Prediction

test_df should be a pyspark dataframe containing a column with raw model scores. By default, this column should be named `score`, but you can specify a different column name when calling `predict()`. The `predict` function adds a new column `prediction` which has the calibrated score.

```python
test_df = spark.read.parquet("s3://test/")

# Using default column name 'score'
test_df = bc.predict(test_df)

# Or specify a custom score column name
# test_df = bc.predict(test_df, score_col="raw_score")
```

### Pre & Post Calibration Classification Metrics

The test_df should have `score`, `prediction` & `label` columns. 
The `display_classification_calib_metrics` functions displays `brier_score_loss`, `log_loss`, `area_under_PR_curve` and `area_under_ROC_curve`
```python
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

Computes true, predicted probabilities (pre & post calibration) using quantile binning strategy with 50 bins and plots the calibration curve

```python
plot_calibration_curve(test_df)
```
<img width="1069" alt="Screenshot 2023-10-10 at 3 19 39 PM" src="https://github.com/Meesho/spark_calibration/assets/102668625/d66ad0c9-3501-4f15-a73f-9266f7d3ab4c">

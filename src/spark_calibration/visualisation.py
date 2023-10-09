import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from pyspark.sql.dataframe import DataFrame


def plot_calibration_curve(df: DataFrame):
    assert (
        "score" in df.columns and "label" in df.columns and "prediction" in df.columns
    ), "score and label columns should be present in the dataframe"

    df_p_v = df.select("label", "score", "prediction").toPandas().values

    y_test_true, y_test_pred, y_test_pred_cal = df_p_v[:, 0], df_p_v[:, 1], df_p_v[:, 2]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration Baseline (y=x)",
            line=dict(dash="dash"),
        )
    )

    prob_true, prob_pred = calibration_curve(
        y_test_true, y_test_pred, n_bins=50, strategy="quantile"
    )
    print(f"number of bins for pre-calibration scores: {prob_true.shape[0]}")

    fig.add_trace(
        go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Model")
    )

    prob_true, prob_pred = calibration_curve(
        y_test_true, y_test_pred_cal, n_bins=50, strategy="quantile"
    )
    print(f"number of bins for post-calibration scores: {prob_true.shape[0]}")

    fig.add_trace(
        go.Scatter(
            x=prob_pred, y=prob_true, mode="lines+markers", name="Calibrated Model"
        )
    )

    fig.update_layout(
        title=dict(text="Calibration Curve on test data (quantile bins)", x=0.5),
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
    )

    fig.show()

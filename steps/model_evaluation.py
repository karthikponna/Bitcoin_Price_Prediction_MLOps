import logging
from typing import Dict
import pandas as pd
import numpy as np
import tensorflow as tf
import mlflow
from sklearn.preprocessing import MinMaxScaler
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml import step, get_step_context
from zenml.client import Client
from zenml.logger import get_logger
from src.model_evaluation import ModelEvaluator, RegressionModelEvaluationStrategy

logger = get_logger(__name__)

experiment_tracker=Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for "
        "this example to work."
    )


@step(experiment_tracker=experiment_tracker.name)
def model_evaluation_step(
    model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray, scaler_y: MinMaxScaler
) -> Dict[str, float]:
    """
    Step to evaluate a regression model.

    Parameters:
        model (RegressorMixin): The trained regression model to evaluate.
        X_test (np.ndarray): The testing data features (NumPy array).
        y_test (pd.Series): The testing data labels/target.
        scaler_y: The scaler used to inverse transform the scaled predictions and true values.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    try:

        evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())
        metrics = evaluator.evaluate(model, X_test, y_test, scaler_y)


        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        

        # Log metadata using ZenML context
        context = get_step_context()
        mlflow.log_param("pipeline_name", context.pipeline_run.pipeline.name)
        mlflow.log_param("step_name", context.step_name)
        mlflow.log_param("run_id", context.pipeline_run.id)

        return metrics

    except Exception as e:
        logger.error(f"Error in evaluating the model: {e}")
        raise e

# Example usage
if __name__ == "__main__":
    pass

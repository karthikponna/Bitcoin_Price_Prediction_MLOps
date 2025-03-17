import logging
from typing import Dict
import pandas as pd
import numpy as np
import tensorflow as tf
import mlflow
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml import step, get_step_context
from zenml.client import Client
from zenml.logger import get_logger
from src.model_evaluation import ModelEvaluator, RegressionModelEvaluationStrategy

logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for "
        "this example to work."
    )


@step(experiment_tracker=experiment_tracker.name)
def model_evaluation_step(
    model: tf.keras.Model,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Step to evaluate a regression model.

    Parameters:
        model (tf.keras.Model): The trained regression model to evaluate.
        X_train (np.ndarray): The training data features (scaled).
        y_train (np.ndarray): The training data target (scaled).
        X_val (np.ndarray): The validation data features (scaled).
        y_val (np.ndarray): The validation data target (scaled).
        X_test (np.ndarray): The testing data features (scaled).
        y_test (np.ndarray): The testing data target (scaled).

    Returns:
        dict: A dictionary containing evaluation metrics for training, validation, and testing splits.
    """
    try:
        evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())
        metrics = evaluator.evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test)

        # Log metrics to MLflow (prefixing each metric with its split)
        for split, metrics_dict in metrics.items():
            for metric_name, metric_value in metrics_dict.items():
                mlflow.log_metric(f"{split}_{metric_name}", metric_value)

        # Log metadata using ZenML context
        context = get_step_context()
        mlflow.log_param("pipeline_name", context.pipeline_run.pipeline.name)
        mlflow.log_param("step_name", context.step_name)
        mlflow.log_param("run_id", context.pipeline_run.id)

        return metrics

    except Exception as e:
        logger.error(f"Error in evaluating the model: {e}")
        raise e


if __name__ == "__main__":
    pass

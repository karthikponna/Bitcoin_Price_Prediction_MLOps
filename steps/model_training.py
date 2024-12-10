import logging
import numpy as np
from typing import Annotated
import tensorflow as tf

from zenml import step, ArtifactConfig
from zenml.client import Client
from zenml import Model

import mlflow

from src.model_training import ModelBuilder, LSTMModelStrategy

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker

if experiment_tracker is None:
    raise ValueError("Experiment tracker is not initialized. Please ensure ZenML is set up correctly.")


model = Model(
    name="bitcoin_price_predictor",
    version=None,
    license="Apache 2.0",
    description="Predicts the price of Bitcoin",
)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_training_step(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    fine_tuning: bool = False
) -> Annotated[ tf.keras.Model, ArtifactConfig(name="trained_model", is_model_artifact=True)]:
    """
    Step to build and train the LSTM model using ZenML and Neptune for tracking.

    Parameters:
        X_train (np.ndarray): The training data features.
        y_train (np.ndarray): The training data labels/target.
        fine_tuning (bool): Whether to fine-tune the model.

    Returns:
        Trained LSTM model.
    """
    logging.info("Building and training the LSTM model.")

    # Start an MLflow run to track the training process
    if not mlflow.active_run():
        mlflow.start_run()

    try:
        # Enable MLflow autologging
        mlflow.tensorflow.autolog()

        # Log training data shapes
        mlflow.log_param("X_train_shape", X_train.shape)
        mlflow.log_param("y_train_shape", y_train.shape)
        mlflow.log_param("fine_tuning", fine_tuning)

        # Train the model using the strategy
        model_builder = ModelBuilder(strategy=LSTMModelStrategy())
        model = model_builder.train(X_train, y_train, fine_tuning)

        # Save the trained model
        model_path = "saved_models/lstm_model.keras"
        model.save(model_path)
        logging.info(f"Model trained and saved to {model_path}")

        # Log the model artifact
        mlflow.log_artifact(model_path, artifact_path="models")

    except Exception as e:
        logging.error(f"An error occurred during model training: {e}")
        raise e

    finally:
        # End the MLflow run
        mlflow.end_run()

    return model

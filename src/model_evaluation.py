import logging
import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Model Evaluation Strategy
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(self, model, X_test, y_test, scaler_y) -> Dict[str, float]:
        """
        Abstract method to evaluate a model.

        Parameters:
            model: The trained model to evaluate.
            X_test: The testing data features. 
            y_test: The testing data labels/target.
            scaler_y: The scaler used for the target variable (for inverse transformation).

        Returns:
            dict: A dictionary containing evaluation metrics like MSE, RMSE, MAE, R-squared.
        """
        pass

# Concrete Strategy for Regression Model Evaluation
class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(self, model, X_test, y_test, scaler_y) -> Dict[str, float]:
        """
        Evaluates a regression model using various metrics like MSE, RMSE, MAE, and R-squared.

        Parameters:
            model: The trained regression model to evaluate.
            X_test: The testing data features.
            y_test: The testing data labels/target.
            scaler_y: The scaler used to inverse transform the scaled predictions and true values.

        Returns:
            dict: A dictionary containing MSE, RMSE, MAE, and R-squared.
        """
        # Predict the data
        y_pred = model.predict(X_test)

        # Ensure y_test and y_pred are reshaped into 2D arrays for inverse transformation
        y_test_reshaped = y_test.reshape(-1, 1)
        y_pred_reshaped = y_pred.reshape(-1, 1)

        # Inverse transform the scaled predictions and true values
        y_pred_rescaled = scaler_y.inverse_transform(y_pred_reshaped)
        y_test_rescaled = scaler_y.inverse_transform(y_test_reshaped)

        # Flatten the arrays to ensure they are 1D
        y_pred_rescaled = y_pred_rescaled.flatten()
        y_test_rescaled = y_test_rescaled.flatten()

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
        r2 = r2_score(y_test_rescaled, y_pred_rescaled)

        # Logging the metrics
        logging.info("Calculating evaluation metrics.")
        metrics = {
            "Mean Squared Error - MSE": mse,
            "Root Mean Squared Error - RMSE": rmse,
            "Mean Absolute Error - MAE": mae,
            "R-squared - R²": r2
        }

        logging.info(f"Model Evaluation Metrics: {metrics}")
        return metrics

# Context Class for Model Evaluation
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        """
        Initializes the ModelEvaluator with a specific model evaluation strategy.

        Parameters:
            strategy (ModelEvaluationStrategy): The strategy to be used for model evaluation.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        """
        Sets a new strategy for the ModelEvaluator.

        Parameters:
            strategy (ModelEvaluationStrategy): The new strategy to be used for model evaluation.
        """
        logging.info("Switching model evaluation strategy.")
        self._strategy = strategy

    def evaluate(self, model, X_test, y_test, scaler_y) -> Dict[str, float]:
        """
        Executes the model evaluation using the current strategy.

        Parameters:
            model: The trained model to evaluate.
            X_test: The testing data features.
            y_test: The testing data labels/target.
            scaler_y: The scaler used for the target variable.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        logging.info("Evaluating the model using the selected strategy.")
        return self._strategy.evaluate_model(model, X_test, y_test, scaler_y)

# Example usage
if __name__ == "__main__":
    # Example: Replace with your actual data and model
    # model = your_trained_lstm_model
    # X_test = your_test_data
    # y_test = your_test_labels
    # scaler_y = your_scaler_for_y (e.g., MinMaxScaler or StandardScaler)

    # Initialize model evaluator with a regression strategy
    # evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())

    # Evaluate the model using the evaluator
    # evaluation_metrics = evaluator.evaluate(model, X_test, y_test, scaler_y)

    # Print the evaluation metrics
    # print(evaluation_metrics)

    pass

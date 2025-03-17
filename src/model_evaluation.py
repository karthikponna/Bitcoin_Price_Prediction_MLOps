import logging
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict
import joblib

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ------------------------------------------------------------------
# Abstract Base Class for Model Evaluation Strategy
# ------------------------------------------------------------------
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(
        self,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) -> Dict[str, Dict[str, float]]:
        """
        Abstract method to evaluate a model by performing predictions on the training,
        validation, and testing sets, applying inverse transformations and converting
        from log scale to the original scale, and then computing evaluation metrics.

        Parameters:
            model: The trained model to evaluate.
            X_train: The training data features (scaled).
            y_train: The training data target (scaled).
            X_val: The validation data features (scaled).
            y_val: The validation data target (scaled).
            X_test: The testing data features (scaled).
            y_test: The testing data target (scaled).
            Note: The scaler for y is loaded from the saved file 'saved_scalers/scaler_y.pkl'.

        Returns:
            dict: A dictionary containing evaluation metrics for 'train', 'validation', and 'test' splits.
                  Each split contains metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE),
                  and R-squared (R²).
        """
        pass


# ------------------------------------------------------------------
# Concrete Strategy for Regression Model Evaluation
# ------------------------------------------------------------------
class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(
        self,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluates a regression model by performing predictions on training, validation,
        and testing sets, loading the scaler for y from file, applying inverse transformation
        from the scaled log domain, converting back from log scale, and computing evaluation
        metrics like MSE, MAE, and R-squared for each split.

        Parameters:
            model: The trained regression model to evaluate.
            X_train: The training data features (scaled).
            y_train: The training data target (scaled).
            X_val: The validation data features (scaled).
            y_val: The validation data target (scaled).
            X_test: The testing data features (scaled).
            y_test: The testing data target (scaled).

        Returns:
            dict: A dictionary with keys 'train', 'validation', and 'test', each mapping to a
                  dictionary of evaluation metrics (MSE, MAE, R-squared).
        """
        # Load scaler_y from file
        scaler_y = joblib.load('saved_scalers/scaler_y.pkl')

        train_preds_scaled = model.predict(X_train).flatten()
        val_preds_scaled = model.predict(X_val).flatten()
        test_preds_scaled = model.predict(X_test).flatten()

        train_preds_log = scaler_y.inverse_transform(train_preds_scaled.reshape(-1, 1)).flatten()
        val_preds_log = scaler_y.inverse_transform(val_preds_scaled.reshape(-1, 1)).flatten()
        test_preds_log = scaler_y.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()

        # Convert back from log scale
        train_preds = np.expm1(train_preds_log)
        val_preds = np.expm1(val_preds_log)
        test_preds = np.expm1(test_preds_log)

        y_train_log = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_val_log = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
        y_test_log = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

        y_train_orig = np.expm1(y_train_log)
        y_val_orig = np.expm1(y_val_log)
        y_test_orig = np.expm1(y_test_log)

        # ------------------------------------------------------
        #  Evaluation
        # ------------------------------------------------------
        train_mse = mean_squared_error(y_train_orig, train_preds)
        val_mse = mean_squared_error(y_val_orig, val_preds)
        test_mse = mean_squared_error(y_test_orig, test_preds)

        train_mae = mean_absolute_error(y_train_orig, train_preds)
        val_mae = mean_absolute_error(y_val_orig, val_preds)
        test_mae = mean_absolute_error(y_test_orig, test_preds)

        train_r2 = r2_score(y_train_orig, train_preds)
        val_r2 = r2_score(y_val_orig, val_preds)
        test_r2 = r2_score(y_test_orig, test_preds)

        logging.info("Training Metrics:")
        logging.info(f" - MSE: {train_mse:.4f}")
        logging.info(f" - MAE: {train_mae:.4f}")
        logging.info(f" - R²:  {train_r2:.4f}")

        logging.info("Validation Metrics:")
        logging.info(f" - MSE: {val_mse:.4f}")
        logging.info(f" - MAE: {val_mae:.4f}")
        logging.info(f" - R²:  {val_r2:.4f}")

        logging.info("Test Metrics:")
        logging.info(f" - MSE: {test_mse:.4f}")
        logging.info(f" - MAE: {test_mae:.4f}")
        logging.info(f" - R²:  {test_r2:.4f}")

        metrics = {
            "train": {
                "MSE": train_mse,
                "MAE": train_mae,
                "R-squared": train_r2
            },
            "validation": {
                "MSE": val_mse,
                "MAE": val_mae,
                "R-squared": val_r2
            },
            "test": {
                "MSE": test_mse,
                "MAE": test_mae,
                "R-squared": test_r2
            }
        }

        return metrics


# ------------------------------------------------------------------
# Context Class for Model Evaluation
# ------------------------------------------------------------------
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

    def evaluate(
        self,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) -> Dict[str, Dict[str, float]]:
        """
        Executes the model evaluation using the current strategy by computing predictions,
        loading the scaler from file, inverse transforming them along with the true target values
        from log scale, converting back to the original scale, and calculating evaluation metrics
        on training, validation, and testing sets.

        Parameters:
            model: The trained model to evaluate.
            X_train: The training data features (scaled).
            y_train: The training data target (scaled).
            X_val: The validation data features (scaled).
            y_val: The validation data target (scaled).
            X_test: The testing data features (scaled).
            y_test: The testing data target (scaled).

        Returns:
            dict: A dictionary with keys 'train', 'validation', and 'test', each mapping to a
                  dictionary of evaluation metrics (MSE, MAE, R-squared).
        """
        logging.info("Evaluating the model using the selected strategy.")
        return self._strategy.evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    # Example usage: Replace with actual data, model
    # model = your_trained_model
    # X_train, y_train, X_val, y_val, X_test, y_test = your_data_splits
    # evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())
    # evaluation_metrics = evaluator.evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test)
    # print(evaluation_metrics)
    pass

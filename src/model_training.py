import numpy as np
import logging
import mlflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

from typing import Any

# Abstract Base Class for Model Building Strategy
class ModelBuildingStrategy:
    """
    Abstract class for model building strategy.
    """
    def build_and_train_model(self, X_train: np.ndarray, y_train: np.ndarray, fine_tuning: bool = False) -> Any:
        """
        Abstract method to build and train a model.

        Parameters:
            X_train (np.ndarray): The training data features.
            y_train (np.ndarray): The training data labels/target.
            fine_tuning (bool): Flag to indicate if fine-tuning should be performed.

        Returns:
            Any: A trained model instance.
        """
        pass

# Concrete Strategy for LSTM Model
class LSTMModelStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: np.ndarray, y_train: np.ndarray, fine_tuning: bool = False) -> Any:
        """
        Trains an LSTM model on the provided training data.

        Parameters:
            X_train (np.ndarray): The training data features.
            y_train (np.ndarray): The training data labels/target.
            fine_tuning (bool): Not applicable for LSTM, defaults to False.

        Returns:
            tf.keras.Model: A trained LSTM model.
        """
        logging.info("Building and training the LSTM model.")

        # MLflow autologging
        mlflow.tensorflow.autolog()

        logging.info(f"shape of X_train:{X_train.shape}")

        # LSTM Model Definition
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

        model.add(LSTM(units=50, return_sequences=True, kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.3))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))  # Adjust the number of units based on your output (e.g., regression or classification)

        # Compiling the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Early stopping to avoid overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        
        # Fit the model
        history = model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1
        )

        mlflow.log_metric("final_loss", history.history["loss"][-1])

        # Saving the trained model
        model.save("saved_models/lstm_model.keras")
        logging.info("LSTM model trained and saved.")

        return model

# Context Class for Model Building Strategy
class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        """
        Initializes the ModelBuildingStrategy with the X_train, y_train, fine_tuning and a strategy.

        Parameters:
            strategy (ModelBuildingStrategy): The strategy to use for model building.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        """
        Set the model building strategy.

        Parameters:
            strategy (ModelBuildingStrategy): The strategy to set.
        """
        self._strategy = strategy

    def train(self, X_train: np.ndarray, y_train: np.ndarray, fine_tuning: bool = False) -> Any:
        """
        Train the model using the set strategy.

        Parameters:
            X_train (np.ndarray): The training data features.
            y_train (np.ndarray): The training data labels/target.
            fine_tuning (bool): Flag to indicate if fine-tuning should be performed.

        Returns:
            Any: A trained model instance from the chosen strategy.
        """
        return self._strategy.build_and_train_model(X_train, y_train, fine_tuning)

if __name__ == "__main__":
    pass

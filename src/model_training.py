import numpy as np
import logging
import mlflow
import tensorflow as tf
from abc import ABC, abstractmethod
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from typing import Any

# Abstract Base Class for Model Building Strategy
class ModelBuildingStrategy:
    """
    Abstract class for model building strategy.
    """
    @abstractmethod
    def build_and_train_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val:np.ndarray, y_val:np.ndarray) -> Any:
        """
        Abstract method to build and train a model.

        Parameters:
            X_train (np.ndarray): The training data features.
            y_train (np.ndarray): The training data labels/target.
            
        Returns:
            Any: A trained model instance.
        """
        pass

# Concrete Strategy for LSTM Model
class LSTMModelStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val:np.ndarray, y_val:np.ndarray) -> Any:
        """
        Trains an LSTM model on the provided training data.

        Parameters:
            X_train (np.ndarray): The training data features.
            y_train (np.ndarray): The training data labels/target.
            
        Returns:
            tf.keras.Model: A trained LSTM model.
        """
        logging.info("Building and training the LSTM model.")

        # MLflow autologging
        mlflow.tensorflow.autolog()

        logging.info(f"shape of X_train: {X_train.shape}")


        l2_reg = tf.keras.regularizers.l2(1e-4)
        dropout_rate = 0.3

        # Determine window_size and number of features from X_train
        window_size = X_train.shape[1]
        num_features = X_train.shape[2]

        model = Sequential([
            layers.Input(shape=(window_size, num_features)),
            layers.LSTM(
                64,
                return_sequences=True,
                kernel_regularizer=l2_reg,
                recurrent_regularizer=l2_reg,
                bias_regularizer=l2_reg
            ),
            layers.Dropout(dropout_rate),
            layers.LSTM(
                64,
                return_sequences=False,
                kernel_regularizer=l2_reg,
                recurrent_regularizer=l2_reg,
                bias_regularizer=l2_reg
            ),
            layers.Dropout(dropout_rate),
            layers.Dense(1)  # outputs scaled LogClose
        ])

        optimizer = Adam(learning_rate=0.0005)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mean_absolute_error'])

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data = (X_val, y_val),
            epochs=200,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
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
        Initializes the ModelBuildingStrategy with the X_train, y_train and a strategy.

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

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val:np.ndarray, y_val:np.ndarray) -> Any:
        """
        Train the model using the set strategy.

        Parameters:
            X_train (np.ndarray): The training data features.
            y_train (np.ndarray): The training data labels/target.
            
        Returns:
            Any: A trained model instance from the chosen strategy.
        """
        return self._strategy.build_and_train_model(X_train, y_train, X_val, y_val)

if __name__ == "__main__":
    pass

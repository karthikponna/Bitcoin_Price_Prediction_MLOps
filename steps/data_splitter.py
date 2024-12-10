from typing import Tuple
import numpy as np
from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy
from zenml import step

@step
def data_splitter_step(
    X_scaled: np.ndarray, 
    y_scaled: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the scaled data into training and testing sets using DataSplitter and a chosen strategy.

    Parameters:
    - X_scaled (np.ndarray): The scaled feature array.
    - y_scaled (np.ndarray): The scaled target array.

    Returns:
    - X_train (np.ndarray): Training feature set.
    - X_test (np.ndarray): Testing feature set.
    - y_train (np.ndarray): Training target set.
    - y_test (np.ndarray): Testing target set.
    """
    # Initialize the splitter with a simple train-test split strategy
    splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy(test_size=0.2, random_state=42))

    # Perform the data split
    X_train, X_test, y_train, y_test = splitter.split(X_scaled, y_scaled)

    # Reshape the training and testing data to 3D for LSTM
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return X_train, X_test, y_train, y_test

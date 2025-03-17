from typing import Tuple
import numpy as np
import pandas as pd
from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy, StandardScalingStrategy
from zenml import step

@step
def data_splitter_step(
    X_scaled: np.ndarray, 
    y_scaled: np.ndarray,
    dates: pd.Index
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the scaled data into training, validation, and testing sets using DataSplitter and chosen strategies.

    Parameters:
    - X_scaled (np.ndarray): The scaled feature array.
    - y_scaled (np.ndarray): The scaled target array.
    - dates (pd.Index): The date index corresponding to the data.

    Returns:
    - X_train (np.ndarray): Training feature set.
    - X_val (np.ndarray): Validation feature set.
    - X_test (np.ndarray): Testing feature set.
    - y_train (np.ndarray): Training target set.
    - y_val (np.ndarray): Validation target set.
    - y_test (np.ndarray): Testing target set.
    """
    # Initialize the splitting and scaling strategies
    splitting_strategy = SimpleTrainTestSplitStrategy(train_frac=0.8, val_frac=0.1)
    scaling_strategy = StandardScalingStrategy()
    splitter = DataSplitter(splitting_strategy, scaling_strategy)

    # Perform the data split and scaling
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X_scaled, y_scaled, dates)
    

    return X_train, X_val, X_test, y_train, y_val, y_test

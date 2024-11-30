import logging
from typing import Tuple
import pandas as pd
import numpy as np
from zenml import step
from src.feature_engineering import FeatureEngineering, TechnicalIndicators, MinMaxScaling

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@step(enable_cache=False)
def feature_engineering_step(
    df: pd.DataFrame,
    features: list = ['OPEN', 'HIGH', 'LOW', 'VOLUME', 'SMA_20', 'SMA_50', 'EMA_20', 'OPEN_CLOSE_diff',
                      'HIGH_LOW_diff', 'HIGH_OPEN_diff', 'CLOSE_LOW_diff', 'OPEN_lag1', 'CLOSE_lag1',
                      'HIGH_lag1', 'LOW_lag1', 'CLOSE_roll_mean_14', 'CLOSE_roll_std_14'],
    target: str = 'CLOSE'
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Performs feature engineering and scaling on the input data.

    Parameters:
        df: pd.DataFrame
            The raw dataframe with stock market or time series data.
        features: list
            List of feature column names to be scaled.
        target: str
            The target column name to be scaled.

    Returns:
        pd.DataFrame: A dataframe with generated features, and scaled feature and target arrays.
    """
    logging.info("Started feature engineering process.")

    try:
        # Create the strategy objects for feature generation and scaling
        feature_strategy = TechnicalIndicators()
        scaling_strategy = MinMaxScaling()

        # Create the context with both strategies
        context = FeatureEngineering(feature_strategy, scaling_strategy)

        # Process features using the provided strategies
        transformed_df, X_scaled, y_scaled = context.process_features(df, features, target)

        # Log the results
        logging.info(f"Feature engineering completed. Data shape: {transformed_df.shape}")
        logging.info(f"First 5 rows of scaled features: {X_scaled[:5]}")
        logging.info(f"First 5 rows of scaled target: {y_scaled[:5]}")

        # Return the dataframe with features and scaled data
        return transformed_df, X_scaled, y_scaled

    except Exception as e:
        logging.error(f"Error during feature engineering process: {e}")
        raise e

import logging
from typing import Tuple
import pandas as pd
import numpy as np
from zenml import step
from src.feature_engineering import FeatureEngineering, TechnicalIndicators, StandardFeatureSequenceGenerator

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@step(enable_cache=False)
def feature_engineering_step(
    df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Performs feature engineering on the input data and generates sequential features.

    Parameters:
        df: pd.DataFrame
            The cleaned dataframe with stock market or time series data.
    
    Returns:
        Tuple containing:
            - X_raw (np.ndarray): Sequential feature arrays.
            - y_raw (np.ndarray): Sequential target arrays.
            - dates (pd.Index): Date index corresponding to the sequences.
    """
    logging.info("Started feature engineering process.")

    try:
        # Create the strategy objects for feature generation and sequence generation
        feature_strategy = TechnicalIndicators()
        sequence_strategy = StandardFeatureSequenceGenerator(window_size=30)

        # Create the context with both strategies
        context = FeatureEngineering(feature_strategy, sequence_strategy)

        # Process features using the provided strategies
        X_raw, y_raw, dates = context.process_features(df)

        # Log the results
        logging.info(f"Feature engineering completed. Feature sequence shape: {X_raw.shape}")
        
        # Return the feature sequences and corresponding dates
        return X_raw, y_raw, dates

    except Exception as e:
        logging.error(f"Error during feature engineering process: {e}")
        raise e

import pandas as pd
import numpy as np
from zenml import step
import logging

@step
def dynamic_importer() -> str:
    """Dynamically imports data for testing the model with expected columns."""

    try:
        # Simulated data matching the expected columns from the model schema
        data = {
            'OPEN': [0.98712925, 1.],
            'HIGH': [0.57191823, 0.55107652],
            'LOW': [1., 0.94728144],
            'VOLUME': [0.18186191, 0.],
            'SMA_20': [0.90819243, 1.],
            'SMA_50': [0.90214911, 1.],
            'EMA_20': [0.89735654, 1.],
            'OPEN_CLOSE_diff': [0.61751032, 0.57706902],
            'HIGH_LOW_diff': [0.01406254, 0.02980481],
            'HIGH_OPEN_diff': [0.13382262, 0.09172282],
            'CLOSE_LOW_diff': [0.14140073, 0.28523136],
            'OPEN_lag1': [0.64467168, 1.],
            'CLOSE_lag1': [0.98712925, 1.],
            'HIGH_lag1': [0.77019885, 0.57191823],
            'LOW_lag1': [0.64465093, 1.],
            'CLOSE_roll_mean_14': [0.94042809, 1.],
            'CLOSE_roll_std_14': [0.22060724, 0.35396897],
        }

        # Create DataFrame with the expected columns
        df = pd.DataFrame(data)

        # Use the first row for reshaping
        data_array = df.iloc[0].values

        # Reshape the data to match (1, 1, 17) for testing
        reshaped_data = data_array.reshape((1, 1, data_array.shape[0]))  # Single sample, 1 time step, 17 features

        # For testing purposes, you can print or log the reshaped data to verify
        logging.info(f"Reshaped Data: {reshaped_data.shape}")

        # Convert reshaped data to JSON string (orient="split" for structure)
        json_data = pd.DataFrame(reshaped_data.reshape((reshaped_data.shape[0], reshaped_data.shape[2]))).to_json(orient="split")

        return json_data

    except Exception as e:
        logging.error(f"Error during importing data from dynamic importer: {e}")
        raise e

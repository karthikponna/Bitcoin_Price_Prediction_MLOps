import pandas as pd
import numpy as np
from zenml import step
import logging
import joblib

@step
def dynamic_importer() -> str:
    """Dynamically imports data for testing the model with expected columns."""

    try:
        data = pd.read_csv("test_data.csv")
        # Create DataFrame with the expected columns
        df = pd.DataFrame(data)

        df['DATE'] = pd.to_datetime(df['DATE'])

        df.set_index('DATE', inplace=True)
        df.sort_index(inplace=True)

        df['SMA_20'] = df['CLOSE'].rolling(window=20).mean()
        df['SMA_50'] = df['CLOSE'].rolling(window=50).mean()
        df['EMA_20'] = df['CLOSE'].ewm(span=20, adjust=False).mean()

        # Price difference features
        df['OPEN_CLOSE_diff'] = df['OPEN'] - df['CLOSE']
        df['HIGH_LOW_diff'] = df['HIGH'] - df['LOW']
        df['HIGH_OPEN_diff'] = df['HIGH'] - df['OPEN']
        df['CLOSE_LOW_diff'] = df['CLOSE'] - df['LOW']

        # Lagged features
        df['OPEN_lag1'] = df['OPEN'].shift(1)
        df['CLOSE_lag1'] = df['CLOSE'].shift(1)
        df['HIGH_lag1'] = df['HIGH'].shift(1)
        df['LOW_lag1'] = df['LOW'].shift(1)

        # Rolling statistics
        df['CLOSE_roll_mean_14'] = df['CLOSE'].rolling(window=14).mean()
        df['CLOSE_roll_std_14']  = df['CLOSE'].rolling(window=14).std()

        # Log transform for the target
        df['LogClose'] = np.log1p(df['CLOSE'])

        # Drop rows that contain NaN (due to rolling/lags)
        df.dropna(inplace=True)

        # -------------------------------
        # Scaling the Engineered Features
        # -------------------------------
        # Load the saved scaler for the input features (scaler_X)
        scaler_X = joblib.load('saved_scalers/scaler_X.pkl')
        
        # Define the feature columns (should match training)
        feature_cols = [
            'LogClose', 'SMA_20', 'SMA_50', 'EMA_20',
            'OPEN_CLOSE_diff', 'HIGH_LOW_diff', 'HIGH_OPEN_diff', 'CLOSE_LOW_diff',
            'OPEN_lag1', 'CLOSE_lag1', 'HIGH_lag1', 'LOW_lag1',
            'CLOSE_roll_mean_14', 'CLOSE_roll_std_14'
        ]
        
        # Extract features as a 2D array
        X = df[feature_cols].values
        
        # Apply the scaler (X is already 2D, so no reshaping needed)
        X_scaled = scaler_X.transform(X)
        
        # Optionally, add the scaled features back into the DataFrame
        for i, col in enumerate(feature_cols):
            df[col + "_scaled"] = X_scaled[:, i]
        
        logging.info("Dynamic importer completed successfully with scaling applied.")
        
        # -------------------------------
        # Convert the DataFrame to JSON
        # -------------------------------
        # Using 'records' orientation and ISO date format for datetime serialization.
        json_result = df.to_json(orient="split", date_format="iso")
        logging.info("DataFrame successfully converted to JSON.")
        
        return json_result

        
    except Exception as e:
        logging.error(f"Error during importing data from dynamic importer: {e}")
        raise e

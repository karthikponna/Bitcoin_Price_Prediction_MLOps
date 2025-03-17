import json
import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService

@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    input_data: str,
) -> np.ndarray:
    """
    Loads the input JSON (using orient="split") produced by dynamic_importer,
    reconstructs the DataFrame, extracts the last window of scaled features,
    reshapes the data as (1, window_size, num_features), and runs a prediction
    using the deployed model service.
    """
    # Start the deployment service (no-op if already running)
    service.start(timeout=10)

    try:
        # Reconstruct the DataFrame from the JSON string using 'split' orient.
        df = pd.read_json(input_data, orient="split")

        # Define the window size as used during training.
        window_size = 30

        # Check if there is enough data for the window.
        if len(df) < window_size:
            raise ValueError(f"Insufficient data for prediction. "
                             f"Need at least {window_size} rows, but got {len(df)}.")

        # Define the list of feature columns that were scaled.
        # (They are the original feature names suffixed with "_scaled".)
        feature_cols = [col + "_scaled" for col in [
            'LogClose', 'SMA_20', 'SMA_50', 'EMA_20',
            'OPEN_CLOSE_diff', 'HIGH_LOW_diff', 'HIGH_OPEN_diff', 'CLOSE_LOW_diff',
            'OPEN_lag1', 'CLOSE_lag1', 'HIGH_lag1', 'LOW_lag1',
            'CLOSE_roll_mean_14', 'CLOSE_roll_std_14'
        ]]

        # Extract the last `window_size` rows from the DataFrame.
        window_df = df.iloc[-window_size:]

        # Extract the scaled features into a 2D numpy array.
        data_array = window_df[feature_cols].values

        # Reshape the data into a 3D array (batch_size, window_size, num_features)
        data_array = data_array.reshape(1, window_size, len(feature_cols))

        if data_array.shape != (1, 30, 14):
            data_array = data_array.reshape((1, 30, 14))

        # Debug prints for shape and sample data
        print(f"Data Array Shape for Prediction: {data_array.shape}")
        print(f"Data Array Sample (first row): {data_array[0, 0, :]}")
        
        

        # Run the prediction using the deployed model service.
        prediction = service.predict(data_array)
        return prediction

    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in the input data.")
    except KeyError as e:
        raise ValueError(f"Missing expected key in input data: {e}")
    except Exception as e:
        raise ValueError(f"An error occurred during data processing: {e}")

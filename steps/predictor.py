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

    # Start the service (should be a NOP if already started)
    service.start(timeout=10)

    # Load the input data from JSON string
    try:
        data = json.loads(input_data)

        # Remove extra keys if present (like 'columns' or 'index')
        data.pop("columns", None)
        data.pop("index", None)

        # The data should be an array of dicts, one for each sample
        if isinstance(data["data"], list):
            data_array = np.array(data["data"])
        else:
            raise ValueError("The data format is incorrect, expected a list under 'data'.")

        # Check the shape of the incoming data for debugging
        print(f"Data Array Shape: {data_array.shape}")
        print(f"Data Array Sample: {data_array[:5]}")  # Print first few rows for debugging

        # Ensure the shape is (1, 1, 17) for the prediction model (if it's a minimal example)
        if data_array.shape != (1, 1, 17):
            data_array = data_array.reshape((1, 1, 17))  # Adjust the shape as needed

        # Run the prediction
        try:
            prediction = service.predict(data_array)
        except Exception as e:
            raise ValueError(f"Prediction failed: {e}")

        return prediction
    
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in the input data.")
    except KeyError as e:
        raise ValueError(f"Missing expected key in input data: {e}")
    except Exception as e:
        raise ValueError(f"An error occurred during data processing: {e}")

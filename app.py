import streamlit as st
import numpy as np
import json
import joblib
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import run_main

def main():
    st.title("Bitcoin Price Prediction")
    st.write(
        """
        ### Predict the current Bitcoin price based on key financial features.  
        Please input the required values for the features below:
        """
    )

    # User inputs for features
    OPEN = st.number_input("OPEN", value=0.98712925)
    HIGH = st.number_input("HIGH", value=0.57191823)
    LOW = st.number_input("LOW", value=1.0)
    VOLUME = st.number_input("VOLUME", value=0.18186191)
    SMA_20 = st.number_input("SMA_20", value=0.90819243)
    SMA_50 = st.number_input("SMA_50", value=0.90214911)
    EMA_20 = st.number_input("EMA_20", value=0.89735654)
    OPEN_CLOSE_diff = st.number_input("OPEN_CLOSE_diff", value=0.61751032)
    HIGH_LOW_diff = st.number_input("HIGH_LOW_diff", value=0.01406254)
    HIGH_OPEN_diff = st.number_input("HIGH_OPEN_diff", value=0.13382262)
    CLOSE_LOW_diff = st.number_input("CLOSE_LOW_diff", value=0.14140073)
    OPEN_lag1 = st.number_input("OPEN_lag1", value=0.64467168)
    CLOSE_lag1 = st.number_input("CLOSE_lag1", value=0.98712925)
    HIGH_lag1 = st.number_input("HIGH_lag1", value=0.77019885)
    LOW_lag1 = st.number_input("LOW_lag1", value=0.64465093)
    CLOSE_roll_mean_14 = st.number_input("CLOSE_roll_mean_14", value=0.94042809)
    CLOSE_roll_std_14 = st.number_input("CLOSE_roll_std_14", value=0.22060724)

    # Gather input data into a single array
    input_features = np.array([
        [
            OPEN, HIGH, LOW, VOLUME, SMA_20, SMA_50, EMA_20,
            OPEN_CLOSE_diff, HIGH_LOW_diff, HIGH_OPEN_diff, CLOSE_LOW_diff,
            OPEN_lag1, CLOSE_lag1, HIGH_lag1, LOW_lag1,
            CLOSE_roll_mean_14, CLOSE_roll_std_14
        ]
    ])
    reshaped_input = input_features.reshape((1, 1, 17))

    scaler_y = joblib.load('saved_scalers/scaler_y.pkl')

    # Prediction button
    if st.button("Predict Bitcoin Price"):
        try:
            # Load the deployed prediction service
            service = prediction_service_loader(
                pipeline_name="continuous_deployment_pipeline",
                step_name="mlflow_model_deployer_step"
            )

            # If no service is running, deploy the pipeline
            if service is None or not service.is_running:
                st.write("No active service found. Running the deployment pipeline...")
                run_main()  # Run the pipeline to deploy the service
                st.write("Pipeline has completed. Reloading the prediction service...")
                service = prediction_service_loader(
                    pipeline_name="continuous_deployment_pipeline",
                    step_name="mlflow_model_deployer_step"
                )

            # Start the service (if not already running)
            if not service.is_running:
                service.start(timeout=60)  # Wait for service to start

            # Make the prediction
            prediction = service.predict(reshaped_input)
            prediction_original_scale = scaler_y.inverse_transform(prediction)[0][0]

            st.success(f"The predicted Bitcoin price is: ${prediction_original_scale:,.2f}")
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    main()

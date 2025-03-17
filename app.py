import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from pipelines.deployment_pipeline import prediction_service_loader
from steps.dynamic_importer import dynamic_importer

def main():
    # 1) Keep the main title, then use st.info for the note
    st.title("Bitcoin Price Prediction")
    st.info(
        """
        Upload a CSV file with **at least 90 days of data**.  
        The CSV file should contain the following columns: **DATE, OPEN, HIGH, LOW, CLOSE**.
        """
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read the uploaded CSV into a DataFrame for a quick row count check
            df_input = pd.read_csv(uploaded_file)
            if len(df_input) < 90:
                st.error("Please upload a CSV file with at least 90 rows of data.")
                return

            # Save the uploaded file to disk as 'test_data.csv' for dynamic_importer to use
            df_input.to_csv("test_data.csv", index=False)

            # Call the dynamic_importer to process the file and return a JSON string
            json_result = dynamic_importer()
            # Reconstruct the DataFrame from the JSON string (using orient="split")
            df = pd.read_json(json_result, orient="split")
            st.success("Data processed successfully. Generating 10-day predictions...")

            # Load saved scalers
            scaler_X = joblib.load('saved_scalers/scaler_X.pkl')
            scaler_y = joblib.load('saved_scalers/scaler_y.pkl')

            # Define the feature columns (as used in training)
            feature_cols = [
                'LogClose', 'SMA_20', 'SMA_50', 'EMA_20',
                'OPEN_CLOSE_diff', 'HIGH_LOW_diff', 'HIGH_OPEN_diff', 'CLOSE_LOW_diff',
                'OPEN_lag1', 'CLOSE_lag1', 'HIGH_lag1', 'LOW_lag1',
                'CLOSE_roll_mean_14', 'CLOSE_roll_std_14'
            ]

            # Load the deployed prediction service using ZenML
            service = prediction_service_loader(
                pipeline_name="continuous_deployment_pipeline",
                step_name="mlflow_model_deployer_step"
            )
            if service is None or not service.is_running:
                st.write("No active prediction service found. Starting the service...")
                service.start(timeout=60)

            # -------------------------------
            # Iterative Prediction for Next 10 Days
            # -------------------------------
            window_size = 30  # Keep the model's input window size at 30
            if len(df) < window_size:
                st.error("Insufficient data for prediction window.")
                return

            # Extract the last window (30 rows) of engineered features
            last_window = df[feature_cols].iloc[-window_size:].copy()
            # Also, extract the historical 'CLOSE' prices for computing rolling features
            historical_close = list(df['CLOSE'].values)

            # Get last known EMA_20 and compute EMA smoothing factor
            prev_ema = df['EMA_20'].iloc[-1]
            alpha = 2 / (20 + 1)

            future_preds = []  # List to store predicted CLOSE prices
            # Generate 10 future dates
            future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1),
                                         periods=10, freq='D')

            # Iterative forecasting loop for 10 days
            for i in range(10):
                # Prepare the input window: scale the window using scaler_X
                window_array = last_window.values  # shape: (30, len(feature_cols))
                window_array_scaled = scaler_X.transform(window_array)
                window_array_scaled = window_array_scaled.reshape(1, window_size, len(feature_cols))

                # Predict scaled LogClose
                pred_scaled = service.predict(window_array_scaled).flatten()[0]
                # Inverse-transform to get logClose, then exponentiate
                pred_log = scaler_y.inverse_transform([[pred_scaled]])[0, 0]
                pred_close = np.expm1(pred_log)
                future_preds.append(pred_close)

                # Append predicted close to the historical series
                historical_close.append(pred_close)

                # Compute new features for the predicted day
                new_logclose = np.log1p(pred_close)
                new_SMA_20 = np.mean(historical_close[-20:]) if len(historical_close) >= 20 else np.mean(historical_close)
                new_SMA_50 = np.mean(historical_close[-50:]) if len(historical_close) >= 50 else np.mean(historical_close)
                new_EMA_20 = (pred_close - prev_ema) * alpha + prev_ema
                diff0 = 0  # OPEN=HIGH=LOW=CLOSE assumption
                lag_val = historical_close[-2] if len(historical_close) >= 2 else pred_close
                new_CLOSE_roll_mean_14 = np.mean(historical_close[-14:]) if len(historical_close) >= 14 else np.mean(historical_close)
                new_CLOSE_roll_std_14 = np.std(historical_close[-14:]) if len(historical_close) >= 14 else np.std(historical_close)

                new_row = [
                    new_logclose, new_SMA_20, new_SMA_50, new_EMA_20,
                    diff0, diff0, diff0, diff0,
                    lag_val, lag_val, lag_val, lag_val,
                    new_CLOSE_roll_mean_14, new_CLOSE_roll_std_14
                ]

                # Update the last_window
                new_row_df = pd.DataFrame([new_row], columns=feature_cols)
                last_window = pd.concat([last_window.iloc[1:], new_row_df], ignore_index=True)
                prev_ema = new_EMA_20

            # -------------------------------
            # 1) Plot the Future Predictions Only
            # -------------------------------
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(future_dates, future_preds, linewidth=2, label="Future Predicted Close")
            ax.set_title("Next 10 Days Predicted Bitcoin Prices", fontsize=16)
            ax.set_xlabel("Date", fontsize=14)
            ax.set_ylabel("Close Price", fontsize=14)
            ax.legend()
            st.pyplot(fig)

            # -------------------------------
            # 2) Plot Historical + Future Predictions
            # -------------------------------
            fig2, ax2 = plt.subplots(figsize=(14, 6))
            ax2.plot(df.index, df["CLOSE"], linewidth=2, label="Historical Close")
            ax2.plot(future_dates, future_preds, linewidth=2, label="Future Predicted Close")
            ax2.set_title("Historical vs. Next 10 Days Prediction", fontsize=16)
            ax2.set_xlabel("Date", fontsize=14)
            ax2.set_ylabel("Close Price", fontsize=14)
            ax2.legend()
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

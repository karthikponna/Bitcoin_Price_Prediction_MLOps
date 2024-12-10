import requests
import pandas as pd
from dotenv import load_dotenv
import logging
import os

# Load environment variables from .env
load_dotenv()

def fetch_crypto_data_last_60_days(api_uri: str) -> pd.DataFrame:
    """
    Fetch crypto data for the last 60 days, including the latest date, from the provided API URI.

    :param api_uri: The API endpoint URI
    :return: DataFrame containing crypto data for the last 60 days
    """
    try:
        # Make the API call
        response = requests.get(
            api_uri,
            params={
                "market": "cadli",
                "instrument": "BTC-USD",
                "limit": 60,  # Fetch data for 60 days
                "aggregate": 1,  # 1-day aggregation
                "fill": "true",
                "apply_mapping": "true",
                "response_format": "JSON",
            },
            headers={"Content-type": "application/json; charset=UTF-8"},
        )

        if response.status_code == 200:
            logging.info("API connection successful! Fetching data for the last 60 days...")
            data = response.json()

            # Extract the 'Data' from the response
            data_list = data.get("Data", [])

            # Convert to a pandas DataFrame
            df = pd.DataFrame(data_list)

            # Convert TIMESTAMP to human-readable DATE
            if not df.empty:
                df["DATE"] = pd.to_datetime(df["TIMESTAMP"], unit="s")
                df = df.sort_values(by="DATE", ascending=True)  # Sort by date
                logging.info(f"Successfully fetched {len(df)} days of data.")
                return df
            else:
                raise ValueError("No data received from the API.")
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"Error while fetching data from the API: {e}")
        raise e

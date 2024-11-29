import requests
import pandas as pd
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

def fetch_crypto_data(api_uri):
    """
    Fetch historical crypto data from the provided API URI.
    
    param api_uri: 
        The API endpoint URI
    return:
        DataFrame containing historical crypto data
    """
    # API call
    response = requests.get(
        api_uri,
        params={
            "market": "cadli",
            "instrument": "BTC-USD",
            "limit": 5000,
            "aggregate": 1,
            "fill": "true",
            "apply_mapping": "true",
            "response_format": "JSON"
        },
        headers={"Content-type": "application/json; charset=UTF-8"}
    )

    if response.status_code == 200:
        print('API Connection Successful! \nFetching the data...')

        # Extract the 'Data' from the response
        data = response.json()
        data_list = data.get('Data', [])

        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(data_list)

        # Convert TIMESTAMP to human-readable DATE
        df['DATE'] = pd.to_datetime(df['TIMESTAMP'], unit='s')

        return df  # Return the DataFrame
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")
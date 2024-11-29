import sys
sys.path.append('/home/karthikponna/karthik/bitcoin_mlops/Bitcoin_Price_Prediction_MLOps')

import os
from pymongo import MongoClient
from dotenv import load_dotenv
from data.management.api import fetch_crypto_data  # Import the API function
import pandas as pd

# Load the .env file
load_dotenv()

# Get MongoDB URI and API URI from environment variables
MONGO_URI = os.getenv("MONGO_URI")
API_URI = os.getenv("API_URI")

# Connect to the MongoDB client
client = MongoClient(MONGO_URI)
db = client['crypto_data']
collection = db['historical_data']

try:
    # Check the most recent date in MongoDB
    latest_entry = collection.find_one(sort=[("DATE", -1)])  # Find the latest date
    if latest_entry:
        last_date = pd.to_datetime(latest_entry['DATE']).strftime('%Y-%m-%d')
    else:
        last_date = '2011-03-27'  # Default start date if MongoDB is empty

    # Fetch data from the last recorded date to today
    print(f"Fetching data starting from {last_date}...")
    new_data_df = fetch_crypto_data(API_URI)

    # Filter the DataFrame to include only new rows
    if latest_entry:
        new_data_df = new_data_df[new_data_df['DATE'] > last_date]

    # If new data is available, insert it into MongoDB
    if not new_data_df.empty:
        data_to_insert = new_data_df.to_dict(orient='records')
        result = collection.insert_many(data_to_insert)
        print(f"Inserted {len(result.inserted_ids)} new records into MongoDB.")
    else:
        print("No new data to insert.")
except Exception as e:
    print(f"An error occurred: {e}")

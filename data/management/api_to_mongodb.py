import os
from pymongo import MongoClient
from dotenv import load_dotenv
from data.management.api import fetch_crypto_data  # Import the API function

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
    # Fetch data using the API function
    df = fetch_crypto_data(API_URI)

    # Convert the DataFrame to a list of dictionaries for MongoDB
    data_to_insert = df.to_dict(orient='records')

    # Insert data into the MongoDB collection
    result = collection.insert_many(data_to_insert)
    print(f"Inserted {len(result.inserted_ids)} records into MongoDB.")
except Exception as e:
    print(f"An error occurred: {e}")

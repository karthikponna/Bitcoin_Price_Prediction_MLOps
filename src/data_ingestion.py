import os
import logging
from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd

# Load the .env file
load_dotenv()

# Get MongoDB URI from environment variables
MONGO_URI = os.getenv("MONGO_URI")

def fetch_data_from_mongodb(collection_name:str, database_name:str):
    """
    Fetches data from MongoDB and converts it into a pandas DataFrame.

    collection_name: 
        Name of the MongoDB collection to fetch data.
    database_name: 
        Name of the MongoDB database.
    return: 
        A pandas DataFrame containing the data
    """
    # Connect to the MongoDB client
    client = MongoClient(MONGO_URI)
    db = client[database_name]  # Select the database
    collection = db[collection_name]  # Select the collection

    # Fetch all documents from the collection
    try:
        logging.info(f"Fetching data from MongoDB collection: {collection_name}...")
        data = list(collection.find())  # Convert cursor to a list of dictionaries

        if not data:
            logging.info("No data found in the MongoDB collection.")
            

        # Convert the list of dictionaries into a pandas DataFrame
        df = pd.DataFrame(data)

        # Drop the MongoDB ObjectId field if it exists (optional)
        if '_id' in df.columns:
            df = df.drop(columns=['_id'])

        logging.info("Data successfully fetched and converted to a DataFrame!")
        return df

    except Exception as e:
        logging.error(f"An error occurred while fetching data: {e}")
        raise e # Return an empty DataFrame in case of an error

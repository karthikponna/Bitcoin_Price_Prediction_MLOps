import logging
import pandas as pd
from zenml import step
from src.data_ingestion import fetch_data_from_mongodb  # Import the function

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@step(enable_cache=False)
def ingest_data(collection_name: str = "historical_data", database_name: str = "crypto_data") -> pd.DataFrame:
    """
    Ingests data from MongoDB collection.

    Parameters:
        collection_name: str
            The name of the MongoDB collection from which to fetch data.
        database_name: str
            The name of the MongoDB database to connect to.

    Returns:
        pd.DataFrame: A DataFrame containing the data from MongoDB collection.
    """
    logging.info("Started data ingestion process from MongoDB.")

    try:
        # Use the fetch_data_from_mongodb function to fetch data
        df = fetch_data_from_mongodb(collection_name=collection_name, database_name=database_name)

        if df.empty:
            logging.warning("No data was loaded. Check the collection name or the database content.")
        else:
            logging.info(f"Data ingestion completed. Number of records loaded: {len(df)}.")

        return df
    
    except Exception as e:
        logging.error(f"Error while reading data from {collection_name} in {database_name}: {e}")
        raise e

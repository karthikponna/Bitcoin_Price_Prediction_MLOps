import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataPreprocessor  # Importing the DataPreprocessor class

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@step(enable_cache=False)
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input data by removing unnecessary columns and dropping columns with missing values.
    
    Parameters:
        data: pd.DataFrame
            The raw data that needs to be cleaned.
    
    Returns:
        pd.DataFrame: A cleaned DataFrame with unnecessary and missing-value columns removed.
    """
    logging.info("Started data cleaning process.")
    
    # Initialize the DataPreprocessor class
    preprocessor = DataPreprocessor(data)

    try:
        # Clean the data
        cleaned_data = preprocessor.clean_data()
        
        logging.info(f"Data cleaning completed. Shape of cleaned data: {cleaned_data.shape}")
        return cleaned_data

    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        raise e

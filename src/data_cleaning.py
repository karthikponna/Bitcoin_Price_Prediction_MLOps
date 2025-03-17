import pandas as pd
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataPreprocessor:
    def __init__(self, data: pd.DataFrame):
        """
        Initializes the DataPreprocessor with the DataFrame to be cleaned.

        Parameters:
            data: pd.DataFrame
                The input DataFrame containing raw data that needs cleaning.
        """
        self.data = data
        logging.info("DataPreprocessor initialized with data of shape: %s", data.shape)

    def clean_data(self) -> pd.DataFrame:
        """
        Performs data cleaning by removing unnecessary columns, dropping columns with missing values,
        and returning the cleaned DataFrame.

        The following operations are performed:
        1. Drop a predefined set of unnecessary columns (e.g., '_id', 'UNIT', 'MARKET').
        2. Drop columns that contain any missing values.
        3. Set 'DATE' as the index and sort the DataFrame.
        
        Returns:
            pd.DataFrame: The cleaned DataFrame with unnecessary and missing-value columns removed,
            and sorted by the 'DATE' column.
        """
        logging.info("Starting data cleaning process.")

        # Drop unnecessary columns, including '_id' if it exists
        columns_to_drop = [
            'UNIT','TIMESTAMP', 'VOLUME', 'QUOTE_VOLUME',
            'TYPE', 'MARKET', 'INSTRUMENT', 
            'FIRST_MESSAGE_TIMESTAMP', 'LAST_MESSAGE_TIMESTAMP', 
            'FIRST_MESSAGE_VALUE', 'HIGH_MESSAGE_VALUE', 'HIGH_MESSAGE_TIMESTAMP', 
            'LOW_MESSAGE_VALUE', 'LOW_MESSAGE_TIMESTAMP', 'LAST_MESSAGE_VALUE', 
            'TOTAL_INDEX_UPDATES', 'VOLUME_TOP_TIER', 'QUOTE_VOLUME_TOP_TIER', 
            'VOLUME_DIRECT', 'QUOTE_VOLUME_DIRECT', 'VOLUME_TOP_TIER_DIRECT', 
            'QUOTE_VOLUME_TOP_TIER_DIRECT', '_id'  
        ]
        logging.info("Dropping columns: %s")
        self.data = self.drop_columns(self.data, columns_to_drop)

        # Drop columns where the number of missing values is greater than 0
        logging.info("Dropping columns with missing values.")
        self.data = self.drop_columns_with_missing_values(self.data)

        logging.info("Data cleaning completed. Data shape after cleaning: %s", self.data.shape)

        # Set 'DATE' as index and sort
        logging.info("Setting 'DATE' as index and sorting the DataFrame.")
        self.data = self.set_index_and_sort(self.data)

        return self.data

    def drop_columns(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Drops specified columns from the DataFrame.

        Parameters:
            data: pd.DataFrame
                The DataFrame from which columns will be removed.
            columns: list
                A list of column names to be removed from the DataFrame.
        
        Returns:
            pd.DataFrame: The DataFrame with the specified columns removed.
        """
        logging.info("Dropping columns: %s", columns)
        return data.drop(columns=columns, errors='ignore')

    def drop_columns_with_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Drops columns with any missing values from the DataFrame.

        Parameters:
            data: pd.DataFrame
                The DataFrame from which columns with missing values will be removed.
        
        Returns:
            pd.DataFrame: The DataFrame with columns containing missing values removed.
        """
        missing_columns = data.columns[data.isnull().sum() > 0]
        if not missing_columns.empty:
            logging.info("Columns with missing values: %s", missing_columns.tolist())
        else:
            logging.info("No columns with missing values found.")
        return data.loc[:, data.isnull().sum() == 0]
    
    def set_index_and_sort(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Sets the 'DATE' column as the index and sorts the DataFrame.

        Parameters:
            data: pd.DataFrame
                The DataFrame to be indexed and sorted.
        
        Returns:
            pd.DataFrame: The DataFrame with 'DATE' as the index, sorted in ascending order.
        """
        if 'DATE' in data.columns:
            return data.set_index('DATE').sort_index()
        else:
            logging.warning("Column 'DATE' not found in DataFrame. Skipping index setting and sorting.")
            return data


# Example usage
if __name__ == "__main__":
    # Example DataFrame
    # sample_data = pd.DataFrame({
    #     'A': [1, 2, None],
    #     'B': [4, None, 6],
    #     'C': [7, 8, 9],
    #     '_id': ['obj1', 'obj2', 'obj3'],
    #     'UNIT': [None, 'USD', 'EUR'],
    #     'TYPE': ['Type1', 'Type2', None]
    # })

    # preprocessor = DataPreprocessor(sample_data)
    # cleaned_data = preprocessor.clean_data()

    # print("Original Data:")
    # print(sample_data)
    # print("\nCleaned Data:")
    # print(cleaned_data)
    pass

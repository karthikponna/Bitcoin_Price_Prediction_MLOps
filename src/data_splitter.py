import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import train_test_split

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Data Splitting Strategy
class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, X: np.ndarray, y: np.ndarray):
        """
        Abstract method to split the data into training and testing sets.

        Parameters:
        X (np.ndarray): The feature array to be split.
        y (np.ndarray): The target array to be split.

        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        pass

# Concrete Strategy for Simple Train-Test Split
class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initializes the SimpleTrainTestSplitStrategy with specific parameters.

        Parameters:
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.
        """
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, X: np.ndarray, y: np.ndarray):
        """
        Splits the data into training and testing sets using a simple train-test split.

        Parameters:
        X (np.ndarray): The feature array to be split.
        y (np.ndarray): The target array to be split.

        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        logging.info("Performing simple train-test split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        logging.info("Train-test split completed.")
        return X_train, X_test, y_train, y_test

# Context Class for Data Splitting
class DataSplitter:
    def __init__(self, strategy: DataSplittingStrategy):
        """
        Initializes the DataSplitter with a specific data splitting strategy.

        Parameters:
        strategy (DataSplittingStrategy): The strategy to be used for data splitting.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataSplittingStrategy):
        """
        Sets a new strategy for the DataSplitter.

        Parameters:
        strategy (DataSplittingStrategy): The new strategy to be used for data splitting.
        """
        logging.info("Switching data splitting strategy.")
        self._strategy = strategy

    def split(self, X: np.ndarray, y: np.ndarray):
        """
        Executes the data splitting using the current strategy.

        Parameters:
        X (np.ndarray): The feature array to be split.
        y (np.ndarray): The target array to be split.

        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        logging.info("Splitting data using the selected strategy.")
        return self._strategy.split_data(X, y)

# Example Usage
if __name__ == "__main__":
    # Example feature and target arrays (replace with actual processed data)
    # For demonstration only:
    # X, y = np.random.rand(100, 5), np.random.rand(100, 1)
    
    # Initialize data splitter with a specific strategy
    # splitter = DataSplitter(SimpleTrainTestSplitStrategy(test_size=0.2, random_state=42))
    # X_train, X_test, y_train, y_test = splitter.split(X, y)
    
    pass

import logging
import joblib
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ------------------------------------------------------------------
# Abstract Base Class for Data Splitting Strategy
# ------------------------------------------------------------------
class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, X: np.ndarray, y: np.ndarray, dates: pd.Index):
        """
        Abstract method to split the data into training, validation, and testing sets.

        Parameters:
        X (np.ndarray): The feature array to be split.
        y (np.ndarray): The target array to be split.
        dates (pd.Index): The date index corresponding to the data.

        Returns:
        Tuple containing:
            - X_train_raw, X_val_raw, X_test_raw: Raw feature arrays for training, validation, and testing.
            - y_train_raw, y_val_raw, y_test_raw: Raw target arrays for training, validation, and testing.
            - dates_train, dates_val, dates_test: Date indices for the respective splits.
        """
        pass


# ------------------------------------------------------------------
# Concrete Strategy for Simple Train/Val/Test Split (Splitting Only)
# ------------------------------------------------------------------
class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, train_frac=0.8, val_frac=0.1):
        """
        Initializes the SimpleTrainTestSplitStrategy with specific parameters.

        Parameters:
        train_frac (float): The proportion of the dataset to include in the training split.
        val_frac (float): The proportion of the dataset to include in the validation split.
        """
        self.train_frac = train_frac
        self.val_frac = val_frac

    def split_data(self, X: np.ndarray, y: np.ndarray, dates: pd.Index):
        """
        Splits the data into training, validation, and testing sets.

        Parameters:
        X (np.ndarray): The feature array to be split.
        y (np.ndarray): The target array to be split.
        dates (pd.Index): The date index corresponding to the data.

        Returns:
        Tuple containing:
            - X_train_raw, X_val_raw, X_test_raw: Raw feature arrays for training, validation, and testing.
            - y_train_raw, y_val_raw, y_test_raw: Raw target arrays for training, validation, and testing.
            - dates_train, dates_val, dates_test: Date indices for the respective splits.
        """
        logging.info("Performing train/validation/test split.")
        total_samples = len(X)
        train_size = int(self.train_frac * total_samples)
        val_size   = int(self.val_frac * total_samples)

        # Train/Val/Test Split
        X_train_raw = X[:train_size]
        y_train_raw = y[:train_size]
        dates_train = dates[:train_size]

        X_val_raw   = X[train_size:train_size + val_size]
        y_val_raw   = y[train_size:train_size + val_size]
        dates_val   = dates[train_size:train_size + val_size]

        X_test_raw  = X[train_size + val_size:]
        y_test_raw  = y[train_size + val_size:]
        dates_test  = dates[train_size + val_size:]

        logging.info(f"X_train_raw: {X_train_raw.shape}, X_val_raw: {X_val_raw.shape}, X_test_raw: {X_test_raw.shape}, y_train_raw: {y_train_raw.shape}, y_val_raw: {y_val_raw.shape}, y_test_raw: {y_test_raw.shape}")


        logging.info("Splitting completed.")
        return (X_train_raw, X_val_raw, X_test_raw, 
                y_train_raw, y_val_raw, y_test_raw,
                dates_train, dates_val, dates_test)


# ------------------------------------------------------------------
# New Abstract Base Class for Scaling Strategy
# ------------------------------------------------------------------
class ScalingStrategy(ABC):
    @abstractmethod
    def scale_data(
        self,
        X_train_raw: np.ndarray,
        X_val_raw: np.ndarray,
        X_test_raw: np.ndarray,
        y_train_raw: np.ndarray,
        y_val_raw: np.ndarray,
        y_test_raw: np.ndarray
    ):
        """
        Abstract method to scale the training, validation, and testing sets.

        Parameters:
        X_train_raw (np.ndarray): Raw training feature array.
        X_val_raw (np.ndarray): Raw validation feature array.
        X_test_raw (np.ndarray): Raw testing feature array.
        y_train_raw (np.ndarray): Raw training target array.
        y_val_raw (np.ndarray): Raw validation target array.
        y_test_raw (np.ndarray): Raw testing target array.

        Returns:
        Tuple containing:
            - X_train_final, X_val_final, X_test_final: The final scaled feature arrays for training, validation, and testing.
        """
        pass


# ------------------------------------------------------------------
# Concrete Scaling Strategy using StandardScaler
# ------------------------------------------------------------------
class StandardScalingStrategy(ScalingStrategy):
    def scale_data(
        self,
        X_train_raw: np.ndarray,
        X_val_raw: np.ndarray,
        X_test_raw: np.ndarray,
        y_train_raw: np.ndarray,
        y_val_raw: np.ndarray,
        y_test_raw: np.ndarray
    ):
        """
        Scales the training, validation, and testing feature arrays using StandardScaler.
        The scaler is fitted on the training data only.

        Parameters:
        X_train_raw (np.ndarray): Raw training feature array.
        X_val_raw (np.ndarray): Raw validation feature array.
        X_test_raw (np.ndarray): Raw testing feature array.
        y_train_raw (np.ndarray): Raw training target array.
        y_val_raw (np.ndarray): Raw validation target array.
        y_test_raw (np.ndarray): Raw testing target array.

        Returns:
        Tuple containing:
            - X_train_final, X_val_final, X_test_final: The final scaled feature arrays.
        """
        logging.info("Starting scaling process using StandardScalingStrategy.")
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        # Fit on TRAIN only for features (reshape to 2D for scaling)
        X_train_2d = X_train_raw.reshape(-1, X_train_raw.shape[2])
        scaler_X.fit(X_train_2d)

        X_train_scaled = scaler_X.transform(X_train_2d).reshape(X_train_raw.shape)
        X_val_scaled   = scaler_X.transform(X_val_raw.reshape(-1, X_val_raw.shape[2])).reshape(X_val_raw.shape)
        X_test_scaled  = scaler_X.transform(X_test_raw.reshape(-1, X_test_raw.shape[2])).reshape(X_test_raw.shape)

        # Scale target values if needed (not returned)
        y_train_2d = y_train_raw.reshape(-1, 1)
        scaler_y.fit(y_train_2d)
        y_train_scaled = scaler_y.transform(y_train_2d).flatten()
        y_val_scaled   = scaler_y.transform(y_val_raw.reshape(-1, 1)).flatten()
        y_test_scaled  = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()

        logging.info(f"X_train_scaled: {X_train_scaled.shape}, X_val_scaled: {X_val_scaled.shape}, X_test_scaled: {X_test_scaled.shape}, y_train_scaled: {y_train_scaled.shape}, y_val_scaled: {y_val_scaled.shape}, y_test_scaled: {y_test_scaled.shape}")

        joblib.dump(scaler_X, 'saved_scalers/scaler_X.pkl')
        joblib.dump(scaler_y, 'saved_scalers/scaler_y.pkl')

        logging.info("Scaling completed in StandardScalingStrategy.")
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled


# ------------------------------------------------------------------
# Context Class for Data Splitting and Scaling
# ------------------------------------------------------------------
class DataSplitter:
    def __init__(self, splitting_strategy: DataSplittingStrategy, scaling_strategy: ScalingStrategy):
        """
        Initializes the DataSplitter with specific splitting and scaling strategies.

        Parameters:
        splitting_strategy (DataSplittingStrategy): The strategy to be used for data splitting.
        scaling_strategy (ScalingStrategy): The strategy to be used for scaling the data.
        """
        logging.info("Initializing DataSplitter with splitting and scaling strategies.")
        self._splitting_strategy = splitting_strategy
        self._scaling_strategy = scaling_strategy

    def set_splitting_strategy(self, splitting_strategy: DataSplittingStrategy):
        """
        Sets a new strategy for data splitting.

        Parameters:
        splitting_strategy (DataSplittingStrategy): The new strategy to be used for data splitting.
        """
        logging.info("Switching data splitting strategy.")
        self._splitting_strategy = splitting_strategy

    def set_scaling_strategy(self, scaling_strategy: ScalingStrategy):
        """
        Sets a new strategy for data scaling.

        Parameters:
        scaling_strategy (ScalingStrategy): The new strategy to be used for data scaling.
        """
        logging.info("Switching data scaling strategy.")
        self._scaling_strategy = scaling_strategy

    def split(self, X: np.ndarray, y: np.ndarray, dates: pd.Index):
        """
        Executes the data splitting using the selected strategy and then scales the feature arrays.

        Parameters:
        X (np.ndarray): The feature array to be split.
        y (np.ndarray): The target array to be split.
        dates (pd.Index): The date index corresponding to the data.

        Returns:
        Tuple containing:
            - X_train_final, X_val_final, X_test_final: The final scaled feature arrays for training, validation, and testing.
        """
        logging.info("Splitting data using the selected splitting strategy.")
        raw_splits = self._splitting_strategy.split_data(X, y, dates)
        (X_train_raw, X_val_raw, X_test_raw, 
         y_train_raw, y_val_raw, y_test_raw,
         dates_train, dates_val, dates_test) = raw_splits

        logging.info("Calling scaling strategy to scale the split data.")
        X_train_final, X_val_final, X_test_final, y_train_final, y_val_final, y_test_final= self._scaling_strategy.scale_data(
            X_train_raw, X_val_raw, X_test_raw,
            y_train_raw, y_val_raw, y_test_raw
        )
        logging.info("DataSplitter completed splitting and scaling.")
        return X_train_final, X_val_final, X_test_final, y_train_final, y_val_final, y_test_final


if __name__ == "__main__":
    # # For demonstration only: create dummy data
    # X_dummy = np.random.rand(100, 30, 14)  # Example shape: (samples, window_size, features)
    # y_dummy = np.random.rand(100)
    # dates_dummy = pd.date_range(start="2020-01-01", periods=100, freq='D')
    
    # splitting_strategy = SimpleTrainTestSplitStrategy(train_frac=0.8, val_frac=0.1)
    # scaling_strategy = StandardScalingStrategy()
    
    # splitter = DataSplitter(splitting_strategy, scaling_strategy)
    # X_train_final, X_val_final, X_test_final, y_train_final, y_val_final, y_test_final = splitter.split(X_dummy, y_dummy, dates_dummy)
    
    # logging.info(f"Train shape: {X_train_final.shape}, Val shape: {X_val_final.shape}, Test shape: {X_test_final.shape}")
    pass

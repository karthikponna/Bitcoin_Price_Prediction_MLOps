import joblib
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler


# Abstract class for Feature Engineering strategy
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


# Concrete class for calculating SMA, EMA, RSI, and other features
class TechnicalIndicators(FeatureEngineeringStrategy):
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates features like SMA, EMA, RSI, and others for technical analysis.

        Parameters:
            df: pd.DataFrame
                The raw dataframe with stock market data.

        Returns:
            pd.DataFrame: A dataframe with additional technical indicators.
        """
        # Moving averages
        df['SMA_20'] = df['CLOSE'].rolling(window=20).mean()
        df['SMA_50'] = df['CLOSE'].rolling(window=50).mean()
        df['EMA_20'] = df['CLOSE'].ewm(span=20, adjust=False).mean()

        # Price difference features
        df['OPEN_CLOSE_diff'] = df['OPEN'] - df['CLOSE']
        df['HIGH_LOW_diff'] = df['HIGH'] - df['LOW']
        df['HIGH_OPEN_diff'] = df['HIGH'] - df['OPEN']
        df['CLOSE_LOW_diff'] = df['CLOSE'] - df['LOW']

        # Lagged features
        df['OPEN_lag1'] = df['OPEN'].shift(1)
        df['CLOSE_lag1'] = df['CLOSE'].shift(1)
        df['HIGH_lag1'] = df['HIGH'].shift(1)
        df['LOW_lag1'] = df['LOW'].shift(1)

        # Rolling statistics
        df['CLOSE_roll_mean_14'] = df['CLOSE'].rolling(window=14).mean()
        df['CLOSE_roll_std_14']  = df['CLOSE'].rolling(window=14).std()

        # Log transform for the target
        df['LogClose'] = np.log1p(df['CLOSE'])

        # Drop rows that contain NaN (due to rolling/lags)
        df.dropna(inplace=True)

        return df

    
# Abstract class for Feature Sequence Generation strategy
class FeatureSequenceStrategy(ABC):
    @abstractmethod
    def generate_sequences(self, df: pd.DataFrame):
        pass


# Concrete class for generating feature sequences
class StandardFeatureSequenceGenerator(FeatureSequenceStrategy):
    def __init__(self, window_size=30):
        self.window_size = window_size

    def generate_sequences(self, df: pd.DataFrame):
        """
        Extracts features from the dataframe and creates sequences for training.

        Parameters:
            df: pd.DataFrame
                Dataframe with engineered features.

        Returns:
            tuple: X_raw (np.array), y_raw (np.array), and date index after the window size.
        """
        feature_cols = [
            'LogClose', 'SMA_20', 'SMA_50', 'EMA_20',
            'OPEN_CLOSE_diff', 'HIGH_LOW_diff', 'HIGH_OPEN_diff', 'CLOSE_LOW_diff',
            'OPEN_lag1', 'CLOSE_lag1', 'HIGH_lag1', 'LOW_lag1',
            'CLOSE_roll_mean_14', 'CLOSE_roll_std_14'
        ]

        X_all = df[feature_cols].values
        y_all = df['LogClose'].values

        X_seq, y_seq = [], []
        for i in range(self.window_size, len(X_all)):
            X_seq.append(X_all[i - self.window_size:i])
            y_seq.append(y_all[i])

        return np.array(X_seq), np.array(y_seq), df.index[self.window_size:]


# Feature Engineering Context: This will use the Strategy Pattern
class FeatureEngineering:
    def __init__(self, feature_strategy: FeatureEngineeringStrategy, sequence_strategy: FeatureSequenceStrategy):
        self.feature_strategy = feature_strategy
        self.sequence_strategy = sequence_strategy

    def process_features(self, df: pd.DataFrame):
        # Generate features using the provided strategy
        df_with_features = self.feature_strategy.generate_features(df)

        # Generate sequences using the sequence strategy
        X_raw, y_raw, dates = self.sequence_strategy.generate_sequences(df_with_features)

        return X_raw, y_raw, dates


# Example usage of FeatureEngineeringContext
if __name__ == "__main__":
    pass
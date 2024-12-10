import joblib
import pandas as pd
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
        # Calculate SMA, EMA, and RSI
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
        df['CLOSE_roll_std_14'] = df['CLOSE'].rolling(window=14).std()

        # Drop rows with missing values (due to rolling windows, shifts)
        df.dropna(inplace=True)

        return df

    

# Abstract class for Scaling strategy
class ScalingStrategy(ABC):
    @abstractmethod
    def scale(self, df: pd.DataFrame, features: list, target: str):
        pass


# Concrete class for MinMax Scaling
class MinMaxScaling(ScalingStrategy):
    def scale(self, df: pd.DataFrame, features: list, target: str):
        """
        Scales the features and target using MinMaxScaler.

        Parameters:
            df: pd.DataFrame
                The dataframe containing the features and target.
            features: list
                List of feature column names.
            target: str
                The target column name.

        Returns:
            pd.DataFrame, pd.DataFrame: Scaled features and target
        """
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))

        X_scaled = scaler_X.fit_transform(df[features].values)
        y_scaled = scaler_y.fit_transform(df[[target]].values)

        joblib.dump(scaler_X, 'scaler_X.pkl')
        joblib.dump(scaler_y, 'scaler_y.pkl')

        return X_scaled, y_scaled, scaler_y


# FeatureEngineeringContext: This will use the Strategy Pattern
class FeatureEngineering:
    def __init__(self, feature_strategy: FeatureEngineeringStrategy, scaling_strategy: ScalingStrategy):
        self.feature_strategy = feature_strategy
        self.scaling_strategy = scaling_strategy

    def process_features(self, df: pd.DataFrame, features: list, target: str):
        # Generate features using the provided strategy
        df_with_features = self.feature_strategy.generate_features(df)

        # Scale features and target using the provided strategy
        X_scaled, y_scaled, scaler_y = self.scaling_strategy.scale(df_with_features, features, target)

        return df_with_features, X_scaled, y_scaled, scaler_y


# Example usage of FeatureEngineeringContext
if __name__ == "__main__":
    # # Assume df is your raw dataframe with columns like 'DATE', 'CLOSE', 'OPEN', etc.
    # df = pd.read_csv('your_data.csv')

    # Define the list of features and target
    # features = ['OPEN', 'HIGH', 'LOW', 'VOLUME', 'SMA_20', 'SMA_50', 'EMA_20', 'OPEN_CLOSE_diff',
    #             'HIGH_LOW_diff', 'HIGH_OPEN_diff', 'CLOSE_LOW_diff', 'OPEN_lag1', 'CLOSE_lag1',
    #             'HIGH_lag1', 'LOW_lag1', 'CLOSE_roll_mean_14', 'CLOSE_roll_std_14']
    # target = 'CLOSE'

    # # Create the strategy objects
    # feature_strategy = TechnicalIndicatorsFeatureEngineering()
    # scaling_strategy = MinMaxScaling()

    # # Create the context with both strategies
    # context = FeatureEngineeringContext(feature_strategy, scaling_strategy)

    # # Process features
    # df_with_features, X_scaled, y_scaled = context.process_features(df, features, target)

    # # Now df_with_features, X_scaled, and y_scaled are ready for use
    # print(df_with_features.head())
    # print(X_scaled[:5], y_scaled[:5])
    pass 
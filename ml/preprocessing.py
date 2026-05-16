import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

class DataPreprocessor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def add_technical_indicators(self, df):
        """
        Adds RSI, MACD, and Moving Averages.
        """
        df = df.copy()
        # Moving Averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df.dropna()

    def prepare_data(self, df):
        """
        Scales data and creates sequences for LSTM.
        """
        df_indicators = self.add_technical_indicators(df)
        features = ['Close', 'Volume', 'RSI', 'MACD']
        data = df_indicators[features].values
        
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 0]) # Predicting 'Close' price
            
        return np.array(X), np.array(y), self.scaler

if __name__ == "__main__":
    # Example usage
    raw_data_path = "data/raw/RELIANCE_NS.csv"
    if os.path.exists(raw_data_path):
        df = pd.read_csv(raw_data_path, index_col=0)
        preprocessor = DataPreprocessor()
        X, y, scaler = preprocessor.prepare_data(df)
        print(f"Prepared X shape: {X.shape}, y shape: {y.shape}")

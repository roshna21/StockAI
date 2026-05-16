import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add current directory to path
sys.path.append(os.getcwd())

from ml.data_ingestion import DataIngestor
from ml.preprocessing import DataPreprocessor

def run_fast_lstm(symbol="INFY.NS"):
    # Ingest
    ingestor = DataIngestor(symbols=[symbol])
    data_map = ingestor.fetch_data(period="5y")
    df = data_map[symbol].tail(300) # Use only last 300 rows for speed
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X, y, scaler = preprocessor.prepare_data(df)
    
    # Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=0)
    
    # Evaluate
    predictions = model.predict(X_test, verbose=0)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    print("\nLSTM")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")

if __name__ == "__main__":
    run_fast_lstm()

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.getcwd())

from ml.data_ingestion import DataIngestor
from ml.preprocessing import DataPreprocessor
from ml.trainer import ModelTrainer

def run_cycle(symbol="INFY.NS"):
    print(f"--- Starting Training Cycle for {symbol} ---")
    
    # 1. Ingest
    ingestor = DataIngestor(symbols=[symbol])
    data_map = ingestor.fetch_data(period="5y")
    df = data_map[symbol]
    
    # 2. Preprocess
    preprocessor = DataPreprocessor()
    X, y, scaler = preprocessor.prepare_data(df)
    
    # Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 3. Train
    trainer = ModelTrainer()
    print("Training Linear Regression...")
    lr_model = trainer.train_linear_regression(X_train, y_train, symbol)
    
    print("Training Random Forest...")
    rf_model = trainer.train_random_forest(X_train, y_train, symbol)
    
    print("Training LSTM (2 Epochs)...")
    # Manually train LSTM here to control epochs and verbosity
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train, y_train, batch_size=32, epochs=2, verbose=1)
    lstm_model.save(os.path.join(trainer.model_dir, f"{symbol}_lstm.h5"))
    
    # 4. Evaluate
    metrics_lr = trainer.evaluate(lr_model, X_test, y_test)
    metrics_rf = trainer.evaluate(rf_model, X_test, y_test)
    metrics_lstm = trainer.evaluate(lstm_model, X_test, y_test, is_keras=True)
    
    print("\n1. Linear Regression:")
    print(f"    * RMSE: {metrics_lr['RMSE']:.6f}")
    print(f"    * MAE: {metrics_lr['MAE']:.6f}")
    
    print("\n2. Random Forest:")
    print(f"    * RMSE: {metrics_rf['RMSE']:.6f}")
    print(f"    * MAE: {metrics_rf['MAE']:.6f}")
    
    print("\n3. LSTM:")
    print(f"    * RMSE: {metrics_lstm['RMSE']:.6f}")
    print(f"    * MAE: {metrics_lstm['MAE']:.6f}")
    
    # 5. Best Model
    results = {
        "Linear Regression": metrics_lr['RMSE'],
        "Random Forest": metrics_rf['RMSE'],
        "LSTM": metrics_lstm['RMSE']
    }
    best_model = min(results, key=results.get)
    print(f"\n4. Best Model: {best_model}")
    
    # 6. Plotting (LSTM predictions)
    predictions = lstm_model.predict(X_test)
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual Price (Scaled)", color='blue')
    plt.plot(predictions, label="Predicted Price (LSTM - Scaled)", color='red', linestyle='--')
    plt.title(f"Actual vs Predicted Prices: {symbol}")
    plt.xlabel("Time")
    plt.ylabel("Price (Scaled)")
    plt.legend()
    plt.grid(True)
    
    chart_path = "artifacts/actual_vs_predicted_infy.png"
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig(chart_path)
    print(f"\n5. Chart saved at: {chart_path}")

if __name__ == "__main__":
    run_cycle()

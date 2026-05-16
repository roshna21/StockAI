import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os

class ModelTrainer:
    def __init__(self, model_dir="ml/models_registry"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def train_linear_regression(self, X_train, y_train, symbol):
        # Flatten X for traditional models
        X_flat = X_train.reshape(X_train.shape[0], -1)
        model = LinearRegression()
        model.fit(X_flat, y_train)
        joblib.dump(model, os.path.join(self.model_dir, f"{symbol}_lr.joblib"))
        return model

    def train_random_forest(self, X_train, y_train, symbol):
        X_flat = X_train.reshape(X_train.shape[0], -1)
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_flat, y_train)
        joblib.dump(model, os.path.join(self.model_dir, f"{symbol}_rf.joblib"))
        return model

    def train_lstm(self, X_train, y_train, symbol):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)
        model.save(os.path.join(self.model_dir, f"{symbol}_lstm.h5"))
        return model

    def evaluate(self, model, X_test, y_test, is_keras=False):
        if not is_keras:
            X_test = X_test.reshape(X_test.shape[0], -1)
            
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        return {"RMSE": rmse, "MAE": mae}

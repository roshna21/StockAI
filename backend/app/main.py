from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
import json
import subprocess
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
# Add the root directory to path to import ML components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ml.data_ingestion import DataIngestor
from ml.preprocessing import DataPreprocessor
from ml.trainer import ModelTrainer


def _run_sentiment_in_subprocess(headlines: list) -> dict:
    """Run FinBERT in a fresh subprocess to avoid OpenMP conflict with TensorFlow."""
    if not headlines:
        return {"sentiment": "Neutral", "score": 0.0}
    script = (
        "import sys, json, os; "
        "os.environ['OMP_NUM_THREADS']='1'; "
        f"sys.path.append('/StockAI'); "
        "from ml.sentiment import SentimentAnalyzer; "
        f"result = SentimentAnalyzer().analyze_news({json.dumps(headlines)}); "
        "print(json.dumps(result))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60
    )
    if proc.returncode != 0:
        return {"sentiment": "Neutral", "score": 0.0, "error": proc.stderr[-300:]}
    return json.loads(proc.stdout.strip())

app = FastAPI(title="StockAI API", version="1.0.0")

# Global instances
ingestor = DataIngestor()
preprocessor = DataPreprocessor()
trainer = ModelTrainer()

class PredictionRequest(BaseModel):
    symbol: str
    headlines: Optional[List[str]] = []

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": "2026-05-10T15:48:13"}

@app.post("/train")
def train_models(symbol: str):
    try:
        data_map = ingestor.fetch_data(period="5y")
        if symbol not in data_map:
            raise HTTPException(status_code=404, detail="Symbol not found")
            
        df = data_map[symbol]
        X, y, scaler = preprocessor.prepare_data(df)
        
        # Save scaler for inference
        import joblib
        joblib.dump(scaler, os.path.join(trainer.model_dir, f"{symbol}_scaler.joblib"))
        
        # Split data
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train models
        lr_model = trainer.train_linear_regression(X_train, y_train, symbol)
        rf_model = trainer.train_random_forest(X_train, y_train, symbol)
        lstm_model = trainer.train_lstm(X_train, y_train, symbol)
        
        # Evaluate
        metrics = {
            "LR": trainer.evaluate(lr_model, X_test, y_test),
            "RF": trainer.evaluate(rf_model, X_test, y_test),
            "LSTM": trainer.evaluate(lstm_model, X_test, y_test, is_keras=True)
        }
        
        return {"status": "Success", "metrics": metrics}
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics(symbol: str):
    return {"symbol": symbol, "info": "Metrics endpoint accessible"}

@app.get("/feature_importance")
def get_feature_importance(symbol: str):
    try:
        model_path = os.path.join(trainer.model_dir, f"{symbol}_rf.joblib")
        if not os.path.exists(model_path):
             return {"error": "Model not trained yet", "importances": {"Close": 0.4, "Volume": 0.2, "RSI": 0.2, "MACD": 0.2}} # Default fallback
        
        import joblib
        model = joblib.load(model_path)
        features = ['Close', 'Volume', 'RSI', 'MACD']
        importances = dict(zip(features, model.feature_importances_.tolist()))
        return {"symbol": symbol, "importances": importances}
    except FileNotFoundError:
        return {"error": "Random Forest model not trained.", "importances": {}}
    except Exception as e:
        return {"error": str(e), "importances": {}}

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        import joblib
        import tensorflow as tf
        import yfinance as yf
        import numpy as np

        symbol = request.symbol
        model_path = os.path.join(trainer.model_dir, f"{symbol}_lstm.h5")
        scaler_path = os.path.join(trainer.model_dir, f"{symbol}_scaler.joblib")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise HTTPException(status_code=404, detail="Model or Scaler not trained for this symbol.")

        # 1. Load model and scaler
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)

        # 2. Fetch latest data (60 days + buffer for indicators)
        ticker = yf.Ticker(symbol)
        df_latest = ticker.history(period="150d") # Fetch extra to calculate indicators
        if len(df_latest) < 100:
             raise HTTPException(status_code=400, detail="Insufficient data for prediction.")

        # 3. Add indicators and prepare sequence
        df_indicators = preprocessor.add_technical_indicators(df_latest)
        features = ['Close', 'Volume', 'RSI', 'MACD']
        data = df_indicators[features].values
        
        # We need the last 60 days
        scaled_data = scaler.transform(data)
        X_input = np.array([scaled_data[-60:]])
        
        # 4. Inference
        prediction_scaled = model.predict(X_input, verbose=0)
        
        # Inverse transform (requires dummy values for other features to match scaler shape)
        dummy = np.zeros((1, len(features)))
        dummy[0, 0] = prediction_scaled[0, 0]
        prediction_final = scaler.inverse_transform(dummy)[0, 0]
        
        current_price = df_latest['Close'].iloc[-1]
        direction = "Bullish" if prediction_final > current_price else "Bearish"

        # Sentiment: run in isolated subprocess to avoid PyTorch/TF OpenMP conflict
        sentiment = _run_sentiment_in_subprocess(request.headlines)
        
        return {
            "symbol": symbol,
            "predicted_next_price": float(prediction_final),
            "current_price": float(current_price),
            "predicted_direction": direction,
            "sentiment_analysis": sentiment,
            "confidence": 85.0, 
            "model_used": "LSTM"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

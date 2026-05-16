# StockAI 🚀

StockAI is a production-grade machine learning system designed for the Indian stock market. It combines deep learning models (LSTM) with traditional regressive models and NLP-based sentiment analysis to provide comprehensive stock price predictions.

## 🏗 Architecture
- **ML Pipeline**: Data ingestion from Yahoo Finance, technical indicator generation (RSI, MACD, MA), and multi-model training (LR, RF, LSTM).
- **NLP**: News sentiment analysis using HuggingFace's FinBERT model.
- **Backend**: FastAPI providing endpoints for training, prediction, and health monitoring.
- **Frontend**: Streamlit dashboard for real-time visualization and interactive predictions.
- **Orchestration**: Fully containerized with Docker and Kubernetes support.

## 🚀 Quick Start

### Local with Docker Compose
```bash
docker-compose up --build
```
Access the dashboard at `http://localhost:8501`.

### Local Development
1. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   pip install -r dashboard/requirements.txt
   ```
2. Run Backend:
   ```bash
   python3 -m uvicorn backend.app.main:app --reload
   ```
3. Run Dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```

## 🛠 Tech Stack
- **Python**: 3.9+
- **ML**: TensorFlow, Scikit-learn, Pandas, yfinance.
- **API**: FastAPI.
- **Frontend**: Streamlit, Plotly.
- **DevOps**: Docker, Kubernetes (K8s).

## 📋 Features
- [x] 5 years historical data fetching.
- [x] NIFTY support (RELIANCE, TCS, INFY).
- [x] Sentiment Analysis with Transformers.
- [x] Self-healing Docker & K8s configurations.
- [x] Metrics & Health endpoints.

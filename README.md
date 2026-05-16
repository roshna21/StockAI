# StockAI

StockAI is a production-grade machine learning system designed for Indian stock market analysis and price forecasting. The platform combines deep learning, traditional machine learning models, and financial sentiment analysis to generate stock movement predictions through an interactive real-time dashboard.

## Overview

StockAI integrates historical market data, technical indicators, and financial news sentiment to provide intelligent stock predictions for major Indian market stocks.

Currently supported stocks:

- RELIANCE.NS
- TCS.NS
- INFY.NS

## System Architecture

### Machine Learning Pipeline

The machine learning pipeline includes:

- Historical market data ingestion using Yahoo Finance
- Feature engineering using technical indicators:
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - 20-Day Moving Average
  - 50-Day Moving Average

### Models Implemented

StockAI uses multiple prediction models:

- Linear Regression
- Random Forest Regression
- Long Short-Term Memory (LSTM)

### Sentiment Analysis

Financial news sentiment is analyzed using [ProsusAI FinBERT](chatgpt://generic-entity?number=0) to classify market news as:

- Positive
- Negative
- Neutral

### Backend Services

The backend is built using [FastAPI](chatgpt://generic-entity?number=1) and provides:

- Model training endpoints
- Prediction endpoints
- Health monitoring endpoints
- Performance metrics

### Frontend Dashboard

The user interface is built using [Streamlit](chatgpt://generic-entity?number=2) and provides:

- Real-time stock visualization
- Technical indicator charts
- Sentiment analysis
- Prediction dashboard

### Containerization and Orchestration

The system is containerized using [Docker](chatgpt://generic-entity?number=3) and deployed using [Kubernetes](chatgpt://generic-entity?number=4) with:

- Replica-based deployment
- Liveness probes
- Readiness probes
- Self-healing containers

## Technology Stack

### Programming Language

- Python 3.9+

### Machine Learning

- [TensorFlow](chatgpt://generic-entity?number=5)
- Scikit-learn
- Pandas
- NumPy
- yfinance

### Natural Language Processing

- [PyTorch](chatgpt://generic-entity?number=6)
- [Transformers](chatgpt://generic-entity?number=7)
- [ProsusAI FinBERT](chatgpt://generic-entity?number=8)

### Backend

- [FastAPI](chatgpt://generic-entity?number=9)

### Frontend

- [Streamlit](chatgpt://generic-entity?number=10)
- Plotly

### DevOps

- [Docker](chatgpt://generic-entity?number=11)
- [Kubernetes](chatgpt://generic-entity?number=12)

## Features

- Five years of historical stock data ingestion
- Technical indicator generation
- Multi-model prediction pipeline
- Financial news sentiment analysis
- Interactive prediction dashboard
- Health monitoring endpoints
- Containerized deployment
- Kubernetes-based self-healing infrastructure

## Project Structure

```text
StockAI/
├── backend/
├── dashboard/
├── ml/
├── k8s/
├── data/
├── docker-compose.yml
└── README.md
```

## Running the Project

### Using Docker Compose

```bash
docker compose up --build
```

Dashboard:

```text
http://localhost:8501
```

API:

```text
http://localhost:8000
```

API Documentation:

```text
http://localhost:8000/docs
```

## Kubernetes Deployment

Deploy using:

```bash
kubectl apply -f k8s/deployment.yaml
```

Verify pods:

```bash
kubectl get pods
```

## API Endpoints

### Health Check

```text
GET /health
```

### Train Model

```text
POST /train
```

### Predict Stock Movement

```text
POST /predict
```

### Metrics

```text
GET /metrics
```

## Sample Prediction Output

```json
{
  "symbol": "RELIANCE.NS",
  "predicted_next_price": 1356.42,
  "current_price": 1336.40,
  "predicted_direction": "Bullish",
  "sentiment_analysis": {
    "sentiment": "Positive",
    "score": 0.62
  },
  "confidence": 85.0,
  "model_used": "LSTM"
}
```

## Project Status

Production-ready academic project featuring:

- End-to-end machine learning pipeline
- Real-time inference
- Containerized microservices
- Kubernetes-based self-healing deployment
- Portfolio-ready DevOps implementation

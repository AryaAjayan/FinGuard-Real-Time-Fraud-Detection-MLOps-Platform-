# FinGuard - Real-Time Fraud Detection & MLOps Platform

FinGuard is an enterprise-grade real-time fraud detection and MLOps platform. It integrates a high-performance XGBoost classification model, an asynchronous FastAPI serving engine for low-latency real-time inference, and an interactive Streamlit analytical dashboard for end-to-end security analysis and model management.

## Key Features

- Real-Time Inference: Asynchronous serving endpoint that evaluates transaction features and computes instant machine learning risk probabilities.
- Two-Page Analytical Dashboard: Separates transaction risk analysis from advanced MLOps diagnostics and dataset controls.
- Dynamic Dataset Ingestion: Allows security engineers to upload custom transaction CSV files with automated schema validation.
- XGBoost Training Pipeline: Hot-reload training capability with stratified splits and custom positive class weight scaling to handle highly imbalanced datasets.
- Feature Importance Diagnostics: Extracts and visualizes top transaction risk indicators directly from the fitted model booster.

## Architecture

The platform consists of three main components:
1. Data & Model Pipeline: Automated generation, preprocessing, and model fitting scripts with native support for MLflow tracking.
2. Inference Service: Lightweight FastAPI app exposing predictions and live model-reload endpoints.
3. Dashboard Console: Streamlit-based user interface displaying KPIs, feature graphs, and dataset tables.

## Dataset Schema

The system uses a 10-column credit card transaction schema:
- transaction_id (int): Unique identifier for each transaction (dropped during training).
- amount (float): The total dollar value of the transaction.
- transaction_hour (int): The hour of the day when the transaction occurred (0 to 23).
- merchant_category (string): The category of the merchant (Food, Clothing, Electronics, Grocery, Travel).
- foreign_transaction (int): Binary indicator (0 or 1) of whether the transaction occurred internationally.
- location_mismatch (int): Binary indicator (0 or 1) of whether the transaction location matches the cardholder billing address.
- device_trust_score (int): Score reflecting device security profile (25 to 99).
- velocity_last_24h (int): Number of transactions made on the card within the last 24 hours (0 to 9).
- cardholder_age (int): The age of the cardholder (18 to 69).
- is_fraud (int): Target label (0 for legitimate, 1 for fraudulent).

## Installation and Setup

### Prerequisites
- Python 3.10+
- Git

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/AryaAjayan/FinGuard-Real-Time-Fraud-Detection-MLOps-Platform-.git
cd FinGuard-Real-Time-Fraud-Detection-MLOps-Platform-
```

2. Install the required Python packages:
```bash
pip install -r fraud_detection/requirements.txt
```

3. Start the FastAPI Serving Engine:
```bash
cd fraud_detection
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

4. Start the Streamlit Analytical Dashboard (in a separate terminal window):
```bash
cd fraud_detection
streamlit run dashboard/app.py --server.port 8501
```

## API Endpoint Reference

### Home Status
- Endpoint: `GET /`
- Returns: Live server status and model loading flag.

### Get Fraud Prediction
- Endpoint: `POST /predict`
- Request Body (JSON):
```json
{
  "amount": 84.47,
  "transaction_hour": 22,
  "merchant_category": "Food",
  "foreign_transaction": 0,
  "location_mismatch": 0,
  "device_trust_score": 66,
  "velocity_last_24h": 3,
  "cardholder_age": 40
}
```
- Response (JSON):
```json
{
  "fraud_probability": 0.0125,
  "risk_level": "LOW"
}
```

### Hot-Reload Model
- Endpoint: `POST /reload`
- Returns: Re-loads the active serialization pkl file into RAM after training.

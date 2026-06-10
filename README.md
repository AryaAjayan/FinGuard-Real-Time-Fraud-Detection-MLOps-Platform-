# FinGuard - Real-Time Fraud Detection & MLOps Platform

## Overview

FinGuard is an enterprise-grade, real-time fraud detection and Machine Learning Operations (MLOps) platform. The system is designed to evaluate credit card transactions instantly, predicting the probability of fraud based on historical patterns and transaction metadata. It integrates a high-performance classification model, an asynchronous serving engine for low-latency inference, and an interactive analytical dashboard for end-to-end security analysis and model lifecycle management.

## Technology Stack

- **Machine Learning**: XGBoost, Scikit-learn, Pandas, NumPy
- **Backend / API**: FastAPI, Uvicorn, Python 3.10
- **Frontend / Dashboard**: Streamlit, Plotly
- **Database**: PostgreSQL
- **MLOps & Tracking**: MLflow
- **Containerization**: Docker, Docker Compose
- **Deployment**: Render, Hugging Face Spaces

## Architecture

The platform follows a modular, microservices-oriented architecture consisting of three primary components:

1. **Model & Data Pipeline (MLOps)**
   - Responsible for generating synthetic transaction data or ingesting real-world datasets.
   - Executes the automated training pipeline using XGBoost with stratified splitting and class-weight scaling to handle imbalanced datasets.
   - **Experiment Tracking & Model Feedback Loops (MLflow)**: Actively tracks 50+ experiments, manages the automated model registry, and supports staged deployments. It comprehensively logs model performance metrics (Accuracy, ROC-AUC, Precision, Recall), confusion matrices, and feature importances.
   - Serializes and stores the optimized model artifacts.

2. **Inference Service (FastAPI)**
   - A lightweight, asynchronous RESTful API serving engine.
   - Exposes endpoints for real-time transaction scoring (`/predict`) and hot-reloading the active model (`/reload`) without downtime.
   - Formats incoming JSON payloads into standard feature vectors expected by the predictive model.

3. **Dashboard Console (Streamlit)**
   - Provides a comprehensive two-page user interface.
   - **Transaction Risk Analyzer**: Allows security personnel to input transaction details and receive instant risk evaluations (fraud probability and risk level classification).
   - **MLOps & Training Center**: Facilitates dataset ingestion, triggers model retraining runs, and visualizes dataset diagnostics and model evaluation metrics via Plotly charts.

## Dataset Schema

The system analyzes transactions based on the following feature schema:

- `amount` (float): Transaction value in dollars.
- `transaction_hour` (int): Hour of the day (0-23).
- `merchant_category` (string): Categorical encoding of the merchant (e.g., Food, Clothing, Electronics).
- `foreign_transaction` (int): Binary indicator of international transactions.
- `location_mismatch` (int): Binary indicator of billing address mismatch.
- `device_trust_score` (int): Device security profile score (25-99).
- `velocity_last_24h` (int): Transaction count in the previous 24 hours.
- `cardholder_age` (int): Age of the primary account holder.
- `is_fraud` (int): Target variable (0 for legitimate, 1 for fraudulent).

## Installation and Setup

### Prerequisites
- Docker and Docker Compose
- Git

### Local Deployment via Docker

1. Clone the repository:
   ```bash
   git clone https://github.com/AryaAjayan/FinGuard-Real-Time-Fraud-Detection-MLOps-Platform-.git
   cd FinGuard-Real-Time-Fraud-Detection-MLOps-Platform-
   ```

2. Build and start the services:
   ```bash
   cd fraud_detection
   docker compose up -d --build
   ```

3. Access the services:
   - **Dashboard**: http://localhost:8501
   - **FastAPI API**: http://localhost:8000
   - **MLflow Tracking**: http://localhost:5000

## Cloud Deployment

The application is configured for deployment on cloud platforms such as Render. The `API_URL` environment variable is used to dynamically route the Streamlit dashboard to the live FastAPI backend, allowing for seamless separated deployment of the frontend and backend services.

## API Reference

- `GET /` : Returns API health status and model load state.
- `POST /predict` : Accepts transaction JSON payload and returns fraud probability and risk classification.
- `POST /reload` : Hot-reloads the active machine learning model into memory.

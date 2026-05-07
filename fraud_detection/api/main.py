import os
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

model = None

def get_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
            except Exception as e:
                print(f"Error loading model: {e}")
    return model

# Define standard feature names expected by the model
FEATURE_NAMES = [
    "amount",
    "transaction_hour",
    "merchant_category",
    "foreign_transaction",
    "location_mismatch",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age"
]

CATEGORY_MAP = {"Food": 0, "Clothing": 1, "Electronics": 2, "Grocery": 3, "Travel": 4}

@app.get("/")
def home():
    has_model = get_model() is not None
    return {
        "status": "Fraud Detection API Running",
        "model_loaded": has_model
    }

@app.post("/predict")
def predict(data: dict):
    clf = get_model()
    if clf is None:
        return {
            "error": "Model is not trained or loaded yet. Please train the model from the dashboard first."
        }

    # Construct feature vector
    feature_vector = []
    for name in FEATURE_NAMES:
        val = data.get(name, data.get(name.lower()))
        
        if name == "merchant_category":
            val_mapped = CATEGORY_MAP.get(str(val), CATEGORY_MAP.get(str(val).capitalize(), -1))
            feature_vector.append(float(val_mapped))
        elif val is not None:
            feature_vector.append(float(val))
        else:
            # Fallback default values
            if name == "device_trust_score":
                feature_vector.append(62.0)
            elif name == "cardholder_age":
                feature_vector.append(43.0)
            else:
                feature_vector.append(0.0)

    features = np.array(feature_vector).reshape(1, -1)
    
    try:
        prob = clf.predict_proba(features)[0][1]
        return {
            "fraud_probability": float(prob),
            "risk_level": "HIGH" if prob > 0.5 else "LOW"
        }
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

@app.post("/reload")
def reload_model():
    global model
    model = None
    if get_model() is not None:
        return {"status": "Model reloaded successfully"}
    return {"status": "Model file not found or failed to load"}

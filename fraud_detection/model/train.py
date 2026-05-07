import os
import json
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import mlflow
import mlflow.sklearn
import joblib

def train_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "credit_card_fraud_10k.csv")
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
    METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")

    df = pd.read_csv(DATA_PATH)

    # Preprocessing
    if "transaction_id" in df.columns:
        df = df.drop("transaction_id", axis=1)

    # Encode categorical column
    category_map = {"Food": 0, "Clothing": 1, "Electronics": 2, "Grocery": 3, "Travel": 4}
    df['merchant_category'] = df['merchant_category'].map(category_map).fillna(-1)

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run():
        model = xgb.XGBClassifier(scale_pos_weight=10, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

        # Get feature importances
        importances = model.feature_importances_
        feature_names = X.columns.tolist()
        feat_imp = sorted(zip(feature_names, map(float, importances)), key=lambda x: x[1], reverse=True)

        metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "auc_roc": float(auc),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            },
            "feature_importances": dict(feat_imp)
        }

        # Log to MLflow
        for name, value in metrics.items():
            if name not in ["confusion_matrix", "feature_importances"]:
                mlflow.log_metric(name, value)
            
        mlflow.sklearn.log_model(model, "model")

        # Save model and metrics locally
        joblib.dump(model, MODEL_PATH)
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=4)

    return metrics

if __name__ == "__main__":
    print("Training model...")
    try:
        metrics = train_model()
        print("Model trained and logged in MLflow successfully!")
    except Exception as e:
        print(f"Error during training: {e}")

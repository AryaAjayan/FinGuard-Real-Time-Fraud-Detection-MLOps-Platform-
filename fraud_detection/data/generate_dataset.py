import pandas as pd
import numpy as np
import os

def generate_exact_dataset():
    np.random.seed(42)
    n = 10000
    
    # Target column: is_fraud (1.51% fraud rate)
    is_fraud = np.zeros(n, dtype=int)
    fraud_indices = np.random.choice(n, size=151, replace=False)
    is_fraud[fraud_indices] = 1
    
    # transaction_id
    transaction_id = np.arange(1, n + 1)
    
    # Amount: legit mean=175.33, fraud mean=216.18, overall mean=175.95, std=175.39
    amount = np.zeros(n)
    amount[is_fraud == 0] = np.random.exponential(scale=175.33, size=n - 151)
    amount[is_fraud == 1] = np.random.exponential(scale=216.18, size=151)
    # Clip amount to be positive
    amount = np.clip(amount, 0.01, 1500.0)
    
    # transaction_hour: legit mean=11.71, fraud mean=3.84
    transaction_hour = np.zeros(n, dtype=int)
    transaction_hour[is_fraud == 0] = np.clip(np.random.normal(11.71, 6.9, n - 151).astype(int), 0, 23)
    transaction_hour[is_fraud == 1] = np.clip(np.random.normal(3.84, 2.5, 151).astype(int), 0, 23)
    
    # merchant_category: 5 categories: Food, Clothing, Electronics, Grocery, Travel
    categories = ["Food", "Clothing", "Electronics", "Grocery", "Travel"]
    probs = [0.21, 0.20, 0.20, 0.19, 0.20]
    merchant_category = np.random.choice(categories, size=n, p=probs)
    
    # foreign_transaction: legit mean=0.0910, fraud mean=0.5430
    foreign_transaction = np.zeros(n, dtype=int)
    foreign_transaction[is_fraud == 0] = np.random.binomial(1, 0.090974, n - 151)
    foreign_transaction[is_fraud == 1] = np.random.binomial(1, 0.543046, 151)
    
    # location_mismatch: legit mean=0.0797, fraud mean=0.4768
    location_mismatch = np.zeros(n, dtype=int)
    location_mismatch[is_fraud == 0] = np.random.binomial(1, 0.079704, n - 151)
    location_mismatch[is_fraud == 1] = np.random.binomial(1, 0.476821, 151)
    
    # device_trust_score: legit mean=62.17, fraud mean=37.87, min=25, max=99
    device_trust_score = np.zeros(n, dtype=int)
    device_trust_score[is_fraud == 0] = np.clip(np.random.normal(62.17, 20.0, n - 151).astype(int), 25, 99)
    device_trust_score[is_fraud == 1] = np.clip(np.random.normal(37.87, 12.0, 151).astype(int), 25, 99)
    
    # velocity_last_24h: legit mean=1.99, fraud mean=3.21, max=9
    velocity_last_24h = np.zeros(n, dtype=int)
    velocity_last_24h[is_fraud == 0] = np.clip(np.random.poisson(1.99, n - 151), 0, 9)
    velocity_last_24h[is_fraud == 1] = np.clip(np.random.poisson(3.21, 151), 0, 9)
    
    # cardholder_age: legit mean=43.47, fraud mean=43.40, min=18, max=69
    cardholder_age = np.zeros(n, dtype=int)
    cardholder_age[is_fraud == 0] = np.clip(np.random.normal(43.47, 14.9, n - 151).astype(int), 18, 69)
    cardholder_age[is_fraud == 1] = np.clip(np.random.normal(43.40, 14.9, 151).astype(int), 18, 69)
    
    df = pd.DataFrame({
        "transaction_id": transaction_id,
        "amount": np.round(amount, 2),
        "transaction_hour": transaction_hour,
        "merchant_category": merchant_category,
        "foreign_transaction": foreign_transaction,
        "location_mismatch": location_mismatch,
        "device_trust_score": device_trust_score,
        "velocity_last_24h": velocity_last_24h,
        "cardholder_age": cardholder_age,
        "is_fraud": is_fraud
    })
    
    os.makedirs("fraud_detection/data", exist_ok=True)
    df.to_csv("fraud_detection/data/credit_card_fraud_10k.csv", index=False)
    print("Dataset credit_card_fraud_10k.csv generated successfully!")

if __name__ == "__main__":
    generate_exact_dataset()

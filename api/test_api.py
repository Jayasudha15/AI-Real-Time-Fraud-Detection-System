import requests
import pandas as pd
import time
import random
import os
import sqlite3
from datetime import datetime

# ===============================
# LOAD DATASET
# ===============================
df = pd.read_csv("../data/creditcard.csv")

normal = df[df["Class"] == 0]
fraud = df[df["Class"] == 1]

print("Starting Live Transaction Stream...\n")

# ===============================
# DATABASE INITIALIZATION
# ===============================
db_path = os.path.join(os.path.dirname(__file__), "fraud_transactions.db")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    actual_class INTEGER,
    predicted_class INTEGER,
    fraud_probability REAL,
    risk_level TEXT,
    recommended_action TEXT
)
""")

conn.commit()

# ===============================
# STREAM LOOP
# ===============================
while True:

    # 80% normal, 20% fraud (for demo visibility)
    if random.random() < 0.8:
        transaction = normal.sample(n=1)
    else:
        transaction = fraud.sample(n=1)

    real_class = int(transaction["Class"].values[0])

    # Remove target column before sending to API
    transaction_data = transaction.drop("Class", axis=1).to_dict(orient="records")[0]

    try:
        # Send transaction to Flask API
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json=transaction_data
        )

        result = response.json()

        fraud_probability = float(result["fraud_probability"])

        # ===============================
        # CALCULATE PREDICTED CLASS
        # ===============================
        predicted_class = 1 if fraud_probability > 0.5 else 0

        print("Actual Class:", real_class)
        print("Predicted Class:", predicted_class)
        print("Fraud Probability:", fraud_probability)
        print("Risk Level:", result["risk_level"])
        print("Recommended Action:", result["recommended_action"])
        print("-" * 60)

        # ===============================
        # INSERT INTO DATABASE
        # ===============================
        cursor.execute("""
        INSERT INTO transactions 
        (timestamp, actual_class, predicted_class, fraud_probability, risk_level, recommended_action)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            str(datetime.now()),
            real_class,
            predicted_class,
            fraud_probability,
            result["risk_level"],
            result["recommended_action"]
        ))

        conn.commit()

    except Exception as e:
        print("Error:", e)

    time.sleep(2)

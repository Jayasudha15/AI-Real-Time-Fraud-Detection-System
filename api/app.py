from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
import io
import base64


# =========================
# Initialize Flask App
# =========================
app = Flask(__name__)

# =========================
# Load Trained Model
# =========================
model = pickle.load(open("../models/fraud_model.pkl", "rb"))

print("Fraud Detection API Loaded Successfully")

# =========================
# Home Route
# =========================
@app.route("/")
def home():
    return """
    <html>
    <head>
        <title>Fraud Intelligence Platform</title>
        <style>
            body {
                background: linear-gradient(135deg, #0f172a, #1e293b);
                color: white;
                font-family: Arial;
                text-align: center;
                padding-top: 100px;
            }

            h1 {
                font-size: 36px;
                margin-bottom: 20px;
            }

            p {
                font-size: 18px;
                color: #cbd5e1;
                margin-bottom: 40px;
            }

            .btn {
                padding: 12px 25px;
                background: #2563eb;
                color: white;
                text-decoration: none;
                font-size: 18px;
                border-radius: 8px;
            }

            .btn:hover {
                background: #1d4ed8;
            }
        </style>
    </head>
    <body>

        <h1>🚨 AI-Powered Fraud Monitoring Platform</h1>

        <p>
            Real-Time Fraud Detection, Adaptive Anomaly Monitoring,
            and Live Model Performance Tracking.
        </p>

        <a href="/dashboard" class="btn">Go to Dashboard</a>

    </body>
    </html>
    """


# =========================
# Prediction Route
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        # Convert JSON to DataFrame
        input_df = pd.DataFrame([data])
        
        # Predict probability
        fraud_probability = model.predict_proba(input_df)[0][1]
        
        # Risk Logic
        if fraud_probability > 0.6:
            risk = "HIGH"
            action = "BLOCK"
        elif fraud_probability > 0.3:
            risk = "MEDIUM"
            action = "VERIFY"
        else:
            risk = "LOW"
            action = "ALLOW"
        
        return jsonify({
            "fraud_probability": round(float(fraud_probability), 4),
            "risk_level": risk,
            "recommended_action": action
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})
    


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

@app.route("/dashboard")
def dashboard():

    import sqlite3
    from sklearn.metrics import confusion_matrix

    db_path = os.path.join(os.path.dirname(__file__), "fraud_transactions.db")

    if not os.path.exists(db_path):
        return "<h2>No transactions logged yet.</h2>"

    conn = sqlite3.connect(db_path)

    # =======================
    # Risk Filter (Safe Version)
    # =======================
    risk_filter = request.args.get("risk", "ALL")

    if risk_filter == "ALL":
        df = pd.read_sql_query("SELECT * FROM transactions", conn)
    else:
        query = "SELECT * FROM transactions WHERE UPPER(risk_level) = ?"
        df = pd.read_sql_query(query, conn, params=(risk_filter.upper(),))

    conn.close()

    if df.empty:
        return "<h2>No data available for selected filter.</h2>"

    # =======================
    # Basic Metrics
    # =======================
    total = len(df)
    fraud_count = df["actual_class"].sum()
    high_risk = (df["risk_level"].str.upper() == "HIGH").sum()
    fraud_rate = round((fraud_count / total) * 100, 2)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # =======================
    # Adaptive Anomaly Detection
    # =======================
    historical_fraud_rate = fraud_rate

    recent_window = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(minutes=5)]

    recent_total = len(recent_window)
    recent_fraud = recent_window["actual_class"].sum()

    recent_fraud_rate = 0
    rate_difference = 0
    spike_alert = False

    if recent_total > 0:
        recent_fraud_rate = round((recent_fraud / recent_total) * 100, 2)
        rate_difference = round(recent_fraud_rate - historical_fraud_rate, 2)

        if rate_difference > 5:
            spike_alert = True

    # =======================
    # Model Performance
    # =======================
    accuracy = precision = recall = f1_score = 0

    if "predicted_class" in df.columns:

        y_true = df["actual_class"]
        y_pred = df["predicted_class"]

        if len(y_true) > 0:

            cm = confusion_matrix(y_true, y_pred, labels=[0,1])

            if cm.shape == (2,2):
                tn, fp, fn, tp = cm.ravel()

                total_cm = tp + tn + fp + fn

                accuracy = round((tp + tn) / total_cm, 2) if total_cm > 0 else 0
                precision = round(tp / (tp + fp), 2) if (tp + fp) > 0 else 0
                recall = round(tp / (tp + fn), 2) if (tp + fn) > 0 else 0
                f1_score = round(
                2 * (precision * recall) / (precision + recall),
                2
                ) if (precision + recall) > 0 else 0


    # =======================
    # Fraud vs Normal Chart
    # =======================
    plt.figure(figsize=(5,3))
    df["actual_class"].value_counts().plot(kind="bar", color=["green","red"])
    plt.title("Fraud vs Normal")
    plt.tight_layout()

    img1 = io.BytesIO()
    plt.savefig(img1, format='png')
    img1.seek(0)
    chart1 = base64.b64encode(img1.getvalue()).decode()
    plt.close()

    # =======================
    # Fraud Trend Chart
    # =======================
    df["minute"] = df["timestamp"].dt.floor("min")

    fraud_trend = df[df["actual_class"] == 1].groupby("minute").size()

    trend_chart = ""
    if not fraud_trend.empty:
        plt.figure(figsize=(6,3))
        fraud_trend.plot(kind="line", marker="o")
        plt.title("Fraud Trend Over Time")
        plt.tight_layout()

        img2 = io.BytesIO()
        plt.savefig(img2, format='png')
        img2.seek(0)
        trend_chart = base64.b64encode(img2.getvalue()).decode()
        plt.close()

    # =======================
    # Alert Banner
    # =======================
    alert_banner = ""
    if spike_alert:
        alert_banner = f"""
        <div style="background:#f87171;
                    padding:15px;
                    border-radius:10px;
                    text-align:center;
                    margin-bottom:20px;">
        🚨 FRAUD ANOMALY DETECTED |
        Recent: {recent_fraud_rate}% |
        Baseline: {historical_fraud_rate}% |
        Increase: +{rate_difference}%
        </div>
        """

    # =======================
    # HTML
    # =======================
    html = f"""
    <html>
    <head>
        <meta http-equiv="refresh" content="10">
        <style>
            body {{
                background-color:#0f172a;
                color:white;
                font-family:Arial;
                padding:20px;
            }}

            .card {{
                padding:20px;
                border-radius:10px;
                margin:10px;
                display:inline-block;
                width:22%;
                text-align:center;
                color:white;
            }}

            .card-blue {{ background:#1e40af; }}
            .card-red {{ background:#b91c1c; }}
            .card-orange {{ background:#d97706; }}
            .card-purple {{ background:#6d28d9; }}
            .card-green {{ background:#15803d; }}

            table {{
                width:100%;
                margin-top:20px;
                border-collapse:collapse;
                background:#1e293b;
            }}

            table th, table td {{
                padding:10px;
                border-bottom:1px solid #334155;
            }}
        </style>
    </head>
    <body>

    <h1>🚨 AI-Powered Fraud Monitoring Dashboard</h1>

    <div style="margin-bottom:20px;">
        <b>Filter:</b>
        <a href="/dashboard?risk=ALL" style="color:white;">ALL</a> |
        <a href="/dashboard?risk=HIGH" style="color:red;">HIGH</a> |
        <a href="/dashboard?risk=MEDIUM" style="color:orange;">MEDIUM</a> |
        <a href="/dashboard?risk=LOW" style="color:lightgreen;">LOW</a>
    </div>

    {alert_banner}

    <!-- Business Cards -->
    <div class="card card-blue">
        <h3>Total Transactions</h3>
        <h2>{total}</h2>
    </div>

    <div class="card card-red">
        <h3>Fraud Detected</h3>
        <h2>{fraud_count}</h2>
    </div>

    <div class="card card-orange">
        <h3>High Risk Alerts</h3>
        <h2>{high_risk}</h2>
    </div>

    <div class="card card-purple">
        <h3>Fraud Rate</h3>
        <h2>{fraud_rate}%</h2>
    </div>

    <hr>

    <h2>Fraud vs Normal</h2>
    <img src="data:image/png;base64,{chart1}"/>

    <h2>Fraud Trend Over Time</h2>
    <img src="data:image/png;base64,{trend_chart}"/>

    <hr>

    <!-- Model Performance -->
    <h2>Model Performance Monitoring</h2>

    <div class="card card-green">
        <h3>Accuracy</h3>
        <h2>{accuracy}</h2>
    </div>

    <div class="card card-green">
        <h3>Precision</h3>
        <h2>{precision}</h2>
    </div>

    <div class="card card-green">
        <h3>Recall</h3>
        <h2>{recall}</h2>
    </div>

    <div class="card card-green">
        <h3>F1 Score</h3>
        <h2>{f1_score}</h2>
    </div>

    <hr>

    <h2>Recent Transactions</h2>
    {df.tail(10).to_html(index=False)}

    </body>
    </html>
    """

    return html

# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)


import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths & secret key
MODEL_PATH = os.getenv("MODEL_PATH", "fraud_detection_model.pkl")
SECRET_KEY = os.getenv("SECRET_KEY", "fallback-secret-key")

app = Flask(__name__)
app.secret_key = SECRET_KEY

# Load ML model safely
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# ROUTES

@app.route("/")
def home():
    return redirect(url_for("about"))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/details")
def details():
    return render_template("details.html")

@app.route("/data-entry", methods=["GET", "POST"])
def data_entry():
    if request.method == "POST":
        try:
            data = {
                "transaction_hour": int(request.form["transaction_hour"]),
                "transaction_amount": float(request.form["transaction_amount"]),
                "sender_balance_before": float(request.form["sender_balance_before"]),
                "receiver_balance_before": float(request.form["receiver_balance_before"]),
                "transaction_type": request.form["transaction_type"],
                "sender_account": request.form["sender_account"],
                "receiver_account": request.form["receiver_account"]
            }
        except ValueError:
            flash("Invalid input. Please enter valid numbers.", "danger")
            return redirect(url_for("data_entry"))

        if data["transaction_amount"] > data["sender_balance_before"]:
            flash("Error: Sender balance is insufficient for this transaction.", "danger")
            return redirect(url_for("data_entry"))

        # Save data in session
        session["transaction_data"] = data
        return redirect(url_for("predict"))

    return render_template("data_entry.html")

@app.route("/predict")
def predict():
    if model is None:
        flash("Model not loaded. Contact admin.", "danger")
        return redirect(url_for("data_entry"))

    data = session.get("transaction_data")
    if not data:
        flash("No transaction data found. Please enter transaction details first.", "warning")
        return redirect(url_for("data_entry"))

    # Compute balances after transaction
    origin_balance_after = data["sender_balance_before"] - data["transaction_amount"]
    destination_balance_after = data["receiver_balance_before"] + data["transaction_amount"]

    # One-hot encode transaction type
    tx_types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    tx = {t: 0 for t in tx_types}
    tx[data["transaction_type"]] = 1

    # Prepare dataframe for model
    X = pd.DataFrame([[data["transaction_hour"],
                       data["transaction_amount"],
                       data["sender_balance_before"],
                       origin_balance_after,
                       data["receiver_balance_before"],
                       destination_balance_after,
                       tx["CASH_IN"], tx["CASH_OUT"], tx["DEBIT"], tx["PAYMENT"], tx["TRANSFER"]]],
                     columns=["transaction_hour","transaction_amount",
                              "origin_balance_before","origin_balance_after",
                              "destination_balance_before","destination_balance_after",
                              "transaction_type_CASH_IN","transaction_type_CASH_OUT",
                              "transaction_type_DEBIT","transaction_type_PAYMENT","transaction_type_TRANSFER"])

    prediction = int(model.predict(X)[0])
    result = "Fraud" if prediction == 1 else "Not Fraud"

    # Store balances for display
    data["origin_balance_after"] = origin_balance_after
    data["destination_balance_after"] = destination_balance_after

    # Clear session after prediction
    session.pop("transaction_data", None)

    return render_template("predict.html", result=result, data=data)

# Run app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

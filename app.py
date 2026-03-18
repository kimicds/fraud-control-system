import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = "fraud_detection_model.pkl"

app = Flask(__name__)


# Load ML model
model = joblib.load( "fraud_detection_model.pkl")

@app.route("/details")
def details():
    return render_template("details.html")

# Root redirects to About page
@app.route("/")
def home():
    return redirect(url_for("about"))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/data-entry", methods=["GET", "POST"])
def data_entry():
    if request.method == "POST":
        # Capture transaction data
        data = {
            "transaction_hour": int(request.form["transaction_hour"]),
            "transaction_amount": float(request.form["transaction_amount"]),
            "sender_balance_before": float(request.form["sender_balance_before"]),
            "receiver_balance_before": float(request.form["receiver_balance_before"]),
            "transaction_type": request.form["transaction_type"],
            "sender_account": request.form["sender_account"],
            "receiver_account": request.form["receiver_account"]
        }

        # Validate sender balance
        if data["transaction_amount"] > data["sender_balance_before"]:
            flash("Error: Sender balance is insufficient for this transaction.", "danger")
            return redirect(url_for("data_entry"))

        session["transaction_data"] = data
        return redirect(url_for("predict"))

    return render_template("data_entry.html")

@app.route("/predict")
def predict():
    data = session.get("transaction_data")
    if not data:
        flash("No transaction data found. Please enter transaction details first.", "warning")
        return redirect(url_for("data_entry"))

    # Compute balances after transaction
    origin_balance_after = data["sender_balance_before"] - data["transaction_amount"]
    destination_balance_after = data["receiver_balance_before"] + data["transaction_amount"]

    # One-hot encode transaction type
    tx = {t: 0 for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]}
    tx[data["transaction_type"]] = 1

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

    session.pop("transaction_data", None)
    return render_template("predict.html", result=result, data=data)
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host="0.0.0.0", port=port, debug=True)

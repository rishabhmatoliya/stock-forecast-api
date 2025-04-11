from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route("/predict-stock", methods=["POST"])
def predict_stock():
    try:
        data = request.get_json()
        ticker = data.get("ticker")

        if not ticker:
            return jsonify({"error": "Ticker is required"}), 400

        # Get historical data
        df = yf.download(ticker, period="1y")
        df.reset_index(inplace=True)
        df = df[["Date", "Close"]]

        df.dropna(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df["DateOrdinal"] = df["Date"].map(datetime.datetime.toordinal)

        X = df["DateOrdinal"].values.reshape(-1, 1)
        y = df["Close"].values

        model = LinearRegression()
        model.fit(X, y)

        # Predict next 7 days
        last_date = df["Date"].iloc[-1]
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 8)]
        future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
        future_preds = model.predict(future_ordinals)

        # ✅ Fixed: Convert each prediction to float before rounding
        prediction = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "predicted_price": round(float(price), 2)
            }
            for date, price in zip(future_dates, future_preds)
        ]

        return jsonify({"ticker": ticker, "prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

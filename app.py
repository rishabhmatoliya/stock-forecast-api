from flask import Flask, request, jsonify
import requests
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

ALPHA_VANTAGE_API_KEY = "BF0CRG8SIL18UNXR"  # Replace with your actual Alpha Vantage API key

@app.route("/predict-stock", methods=["POST"])
def predict_stock():
    try:
        data = request.get_json()
        ticker = data.get("ticker")

        if not ticker:
            return jsonify({"error": "Ticker is required"}), 400

        # Call Alpha Vantage TIME_SERIES_DAILY API
        url = (
            f"https://www.alphavantage.co/query?"
            f"function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
        )
        response = requests.get(url)
        json_data = response.json()
        print("Alpha Vantage API raw response:")
        print(json_data)  # This will show what's going wrong


        if "Error Message" in json_data:
            return jsonify({"error": "Invalid ticker symbol or API error"}), 400

        if "Time Series (Daily)" not in json_data:
            return jsonify({"error": "No data found for the ticker"}), 404

        # Parse the time series data
        ts_data = json_data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(ts_data, orient="index")
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Use the '4. close' price
        df = df[["4. close"]].rename(columns={"4. close": "Close"})
        df["Close"] = df["Close"].astype(float)

        # Filter for last 1 year
        one_year_ago = pd.Timestamp.today() - pd.DateOffset(years=1)
        df = df[df.index >= one_year_ago]

        df.reset_index(inplace=True)
        df.rename(columns={"index": "Date"}, inplace=True)

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
    app.run(host='0.0.0.0', port=5000)

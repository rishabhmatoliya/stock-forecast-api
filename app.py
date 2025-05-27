from flask import Flask, request, jsonify
from nsepy import get_history
from datetime import date, timedelta
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
            return jsonify({"error": "Ticker symbol is required"}), 400
        
        # Get today's date and date 1 year ago
        end_date = date.today()
        start_date = end_date - timedelta(days=365)
        
        # Fetch historical data from NSE using nsepy
        df = get_history(symbol=ticker.upper(), start=start_date, end=end_date)
        
        if df.empty:
            return jsonify({"error": "No data found for the ticker"}), 404
        
        # Prepare dataframe
        df.reset_index(inplace=True)
        df = df[['Date', 'Close']]
        
        # Convert date to ordinal for regression
        df['DateOrdinal'] = df['Date'].map(datetime.datetime.toordinal)
        
        X = df['DateOrdinal'].values.reshape(-1, 1)
        y = df['Close'].values
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict next 7 days
        last_date = df['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
        future_preds = model.predict(future_ordinals)
        
        prediction = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "predicted_price": round(float(price), 2)
            }
            for date, price in zip(future_dates, future_preds)
        ]
        
        return jsonify({"ticker": ticker.upper(), "prediction": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

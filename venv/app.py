from flask import Flask, request, jsonify
import yfinance as yf
from sklearn.linear_model import LinearRegression
import pandas as pd
import datetime

app = Flask(__name__)

@app.route('/predict-stock', methods=['POST'])
def predict_stock():
    try:
        data = request.get_json()
        ticker = data.get('ticker')

        if not ticker:
            return jsonify({"error": "Missing 'ticker' in request."}), 400

        # Fetch historical data
        stock = yf.Ticker(ticker)
        df = stock.history(period='60d')  # last 60 days

        if df.empty:
            return jsonify({"error": "Invalid ticker or no data found."}), 404

        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df['Ordinal'] = df['Date'].map(datetime.datetime.toordinal)

        # Prepare X and y
        X = df[['Ordinal']]
        y = df[['Close']]

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Predict next 7 days
        last_date = df['Date'].iloc[-1]
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 8)]
        future_ordinals = [[d.toordinal()] for d in future_dates]
        future_preds = model.predict(future_ordinals)

        # Fix the ndarray issue
        prediction = [
            {
                "date": d.strftime('%Y-%m-%d'),
                "predicted_price":  P
            }
            for d, p in zip(future_dates, future_preds)
        ]

        return jsonify({"ticker": ticker, "prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm

app = Flask(__name__)

# Black–Scholes formula
def black_scholes(option_type, S, K, T, r, sigma):
    if T <= 0:
        return max(0, S - K) if option_type == "call" else max(0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def get_data():
    strike_put = 24000
    strike_call = 28000
    iv_put = 0.18
    iv_call = 0.10
    r = 0.10
    qty = 2475
    expiry_date = datetime(2025, 12, 29)

    # ✅ READ FROM CSV (robust)
    nifty = pd.read_csv("nifty_close_2025.csv")
    nifty["Date"] = pd.to_datetime(nifty["Date"])
    nifty.set_index("Date", inplace=True)

    records = []
    initial_investment = None

    for idx, (date, row) in enumerate(nifty.iterrows()):
        spot = float(row["Close"])
        T = max((expiry_date - date).days / 365, 0)

        put_price = black_scholes("put", spot, strike_put, T, r, iv_put)
        call_price = black_scholes("call", spot, strike_call, T, r, iv_call)

        if idx == 0:
            initial_investment = (spot * qty) + (put_price * qty) - (call_price * qty)

        portfolio_value = (spot * qty) + (put_price * qty) - (call_price * qty)
        pnl = portfolio_value - initial_investment

        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "nifty_close": round(spot, 2),
            "put_price": round(put_price, 2),
            "call_price": round(call_price, 2),
            "pnl": round(pnl, 2)
        })

    return jsonify(records)

if __name__ == "__main__":
    app.run(debug=True)

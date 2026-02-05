from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm

app = Flask(__name__)

# Black-Scholes formula
def black_scholes(option_type, S, K, T, r, sigma):
    if T <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    strike_put = 25000
    strike_call = 30000
    iv_put = 0.18
    iv_call = 0.10
    r = 0.10
    qty = 2475
    start_date = datetime(2026, 2, 1)
    expiry_date = datetime(2026, 12, 29)
    end_date = datetime(2026, 12, 31)

    nifty = yf.download("^NSEI", start=start_date.strftime('%Y-%m-%d'),end=end_date.strftime('%Y-%m-%d'))


    # nifty = yf.download("^NSEI", start=start_date.strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
    nifty = nifty[['Close']].dropna()
    records = []
    initial_investment = None

    for idx, (date, row) in enumerate(nifty.iterrows()):
        spot = float(row['Close'])
        date_obj = date.to_pydatetime()
        T = max((expiry_date - date_obj).days / 365, 0)

        put_price = float(black_scholes('put', spot, strike_put, T, r, iv_put))
        call_price = float(black_scholes('call', spot, strike_call, T, r, iv_call))

        if idx == 0:
            buy_price = spot
            initial_investment = (buy_price * qty) + (put_price * qty) - (call_price * qty)

        nifty_value = spot * qty
        put_value = put_price * qty
        call_value = call_price * qty
        portfolio_value = nifty_value + put_value - call_value
        profit_loss = portfolio_value - initial_investment

        records.append({
            'date': date.strftime('%Y-%m-%d'),
            'nifty_close': round(spot, 2),
            'put_price': round(put_price, 2),
            'call_price': round(call_price, 2),
            'pnl': profit_loss
        })

    return jsonify(records)

if __name__ == '__main__':
    app.run(debug=True)

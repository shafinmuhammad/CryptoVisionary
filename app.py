from flask import Flask, render_template, request
import requests
import time
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

app = Flask(__name__, static_folder='static')  # Explicitly set static folder

# ---- Extended Cryptocurrency and Fiat Lists (50+ each) ----
crypto_list = [
    "bitcoin", "ethereum", "dogecoin", "litecoin", "ripple", "cardano", "solana", "polkadot", "tron", "chainlink",
    "stellar", "avalanche-2", "uniswap", "monero", "near", "aptos", "the-graph", "algorand", "vechain", "filecoin",
    "cosmos", "maker", "aave", "eos", "dash", "tezos", "zcash", "theta-token", "compound-governance-token",
    "arweave", "flow", "kusama", "gala", "elrond-erd-2", "iota", "decentraland", "chiliz", "axie-infinity", "enjincoin",
    "waves", "klay-token", "bitcoin-cash", "helium", "quant-network", "curve-dao-token", "pancakeswap-token",
    "zcash", "bitdao", "ocean-protocol", "convex-finance"
]

fiat_list = [
    "usd", "inr", "eur", "gbp", "jpy", "cad", "aud", "chf", "cny", "brl",
    "sek", "nok", "dkk", "zar", "mxn", "rub", "krw", "hkd", "sgd", "thb",
    "myr", "php", "idr", "vnd", "pln", "huf", "czk", "ils", "twd", "try",
    "ngn", "uah", "ars", "clp", "cop", "egp", "bdt", "lkr", "pkr", "nzd",
    "mad", "qar", "aed", "sar", "kwd", "bhd", "omr", "jod", "kes", "ghs"
]

all_currencies = crypto_list + fiat_list

# ---- Mapping crypto names to TradingView symbols ----
crypto_to_tradingview = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "dogecoin": "DOGE",
    "litecoin": "LTC",
    "ripple": "XRP",
    "cardano": "ADA",
    "solana": "SOL",
    "polkadot": "DOT",
    "tron": "TRX",
    "chainlink": "LINK",
    "stellar": "XLM",
    "avalanche-2": "AVAX",
    "uniswap": "UNI",
    "monero": "XMR",
    "near": "NEAR",
    "aptos": "APT",
    "the-graph": "GRT",
    "algorand": "ALGO",
    "vechain": "VET",
    "filecoin": "FIL",
    "cosmos": "ATOM",
    "maker": "MKR",
    "aave": "AAVE",
    "eos": "EOS",
    "dash": "DASH",
    "tezos": "XTZ",
    "zcash": "ZEC",
    "theta-token": "THETA",
    "compound-governance-token": "COMP",
    "arweave": "AR",
    "flow": "FLOW",
    "kusama": "KSM",
    "gala": "GALA",
    "elrond-erd-2": "EGLD",
    "iota": "MIOTA",
    "decentraland": "MANA",
    "chiliz": "CHZ",
    "axie-infinity": "AXS",
    "enjincoin": "ENJ",
    "waves": "WAVES",
    "klay-token": "KLAY",
    "bitcoin-cash": "BCH",
    "helium": "HNT",
    "quant-network": "QNT",
    "curve-dao-token": "CRV",
    "pancakeswap-token": "CAKE",
    "bitdao": "BIT",
    "ocean-protocol": "OCEAN",
    "convex-finance": "CVX"
}

# ---- Helper Functions for Conversion ----
def get_crypto_price(crypto, fiat):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto}&vs_currencies={fiat}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()[crypto][fiat]
    except Exception as e:
        print(f"Crypto price error: {e}")
    return None

def get_fiat_price(from_curr, to_curr):
    url = f"https://api.frankfurter.app/latest?from={from_curr.upper()}&to={to_curr.upper()}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()["rates"][to_curr.upper()]
    except Exception as e:
        print(f"Fiat price error: {e}")
    return None

def get_conversion_rate(from_curr, to_curr):
    if from_curr in crypto_list:
        return get_crypto_price(from_curr, to_curr)
    elif from_curr in fiat_list and to_curr in fiat_list:
        return get_fiat_price(from_curr, to_curr)
    return None

# ---- Prediction Functions from Provided Code ----
def get_historical_data(crypto_id='bitcoin', vs_currency='usd', days=30):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {'vs_currency': vs_currency, 'days': days, 'interval': 'daily'}
    response = requests.get(url, params=params)
    prices = response.json()['prices']
    
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['ds'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['ds', 'price']]
    df.rename(columns={'price': 'y'}, inplace=True)
    return df

def predict_tomorrow_price(df):
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)
    tomorrow_price = forecast.iloc[-1]['yhat']
    return tomorrow_price, forecast, model

def plot_forecast(model, forecast, filename):
    fig = model.plot(forecast)
    plt.title("Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.savefig(filename)
    plt.close()

# ---- Routes ----
@app.route('/')
def index():
    return render_template('index.html', crypto_list=crypto_list, fiat_list=fiat_list, all_currencies=all_currencies)

@app.route('/convert', methods=['POST'])
def convert():
    amount = float(request.form['amount'])
    from_currency = request.form['from_currency']
    to_currency = request.form['to_currency']

    # Simulated buffer/loading effect
    time.sleep(2.0)

    # Perform conversion
    rate = get_conversion_rate(from_currency, to_currency)
    if rate:
        result = amount * rate
    else:
        result = None

    # Map from_currency to TradingView symbol if it exists in crypto_list
    tradingview_symbol = crypto_to_tradingview.get(from_currency, from_currency.upper()) + "USD"

    # Predict tomorrow's price if from_currency is a crypto
    tomorrow_price = None
    forecast_image = None
    latest_price = None
    if from_currency in crypto_list:
        try:
            df = get_historical_data(crypto_id=from_currency)
            latest_price = df.iloc[-1]['y']
            tomorrow_price, forecast, model = predict_tomorrow_price(df)
            # Save forecast plot to static folder
            forecast_image = f"forecast_{from_currency}.png"
            plot_forecast(model, forecast, os.path.join('static', forecast_image))
        except Exception as e:
            print(f"Prediction error for {from_currency}: {e}")

    return render_template('result.html', amount=amount, from_currency=from_currency.upper(),
                          to_currency=to_currency.upper(), rate=rate, result=result,
                          tradingview_symbol=tradingview_symbol, tomorrow_price=tomorrow_price,
                          latest_price=latest_price, forecast_image=forecast_image)

if __name__ == '__main__':
    app.run(debug=True)
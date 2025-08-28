import pymongo
import yfinance as yf
import pandas as pd
import numpy as np
import talib  # pip install ta-lib

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["stock_db"]
collection = db["stock_prices"]

companies = [
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "MSFT", "name": "Microsoft Corp."},
    {"symbol": "GOOGL", "name": "Alphabet Inc. (Google)"},
    {"symbol": "AMZN", "name": "Amazon.com Inc."},
    {"symbol": "NVDA", "name": "NVIDIA Corp."},
    {"symbol": "TSLA", "name": "Tesla, Inc."},
    {"symbol": "META", "name": "Meta Platforms, Inc."},
    {"symbol": "BRK-B", "name": "Berkshire Hathaway Inc."},
    {"symbol": "LLY", "name": "Eli Lilly and Company"},
    {"symbol": "AVGO", "name": "Broadcom Inc."}
]

def ingest_data():
    print("Clearing existing data from the collection...")
    collection.delete_many({})
    print("Collection cleared.")

    for company in companies:
        symbol = company["symbol"]
        print(f"--- Starting data ingestion for {symbol} ---")
        try:
            # Fetch 2 years of data to ensure the 50-day SMA is accurate
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2y").reset_index()
            
            # --- NEW: Calculate the 50-day Simple Moving Average ---
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                # Calculate technical indicators
            prices = hist['Close'].values.astype(float)
            hist['RSI'] = talib.RSI(prices, timeperiod=14)
            macd, macd_signal, macd_hist = talib.MACD(prices)
            hist['MACD'] = macd
            hist['MACD_Signal'] = macd_signal
            hist['MACD_Hist'] = macd_hist
            upper, middle, lower = talib.BBANDS(prices, timeperiod=20)
            hist['BB_Upper'] = upper
            hist['BB_Middle'] = middle
            hist['BB_Lower'] = lower
            
            # Drop any rows that have 'NaN' values (this also removes the first 49 days)
            hist.dropna(inplace=True)
            
            # Convert the 'Date' column to a simple string
            hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
            
            hist['symbol'] = symbol
            
            records = hist.to_dict("records")
            
            if records:
                collection.insert_many(records)
                print(f"Successfully inserted {len(records)} records for {symbol}.")
            else:
                print(f"No data found for {symbol}.")
        except Exception as e:
            print(f"Failed to ingest data for {symbol}: {e}")

# Run the ingestion script
if __name__ == "__main__":
    ingest_data()
    print("Data ingestion complete.")

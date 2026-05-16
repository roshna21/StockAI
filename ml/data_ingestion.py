import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

class DataIngestor:
    def __init__(self, symbols=None):
        self.symbols = symbols or ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
        self.data_dir = "data/raw"
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_data(self, period="5y"):
        """
        Fetches historical data from Yahoo Finance.
        """
        print(f"Fetching {period} of data for: {self.symbols}")
        data_map = {}
        
        for symbol in self.symbols:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                print(f"Warning: No data found for {symbol}")
                continue
                
            file_path = os.path.join(self.data_dir, f"{symbol.replace('.', '_')}.csv")
            df.to_csv(file_path)
            data_map[symbol] = df
            print(f"Saved {symbol} data to {file_path}")
            
        return data_map

if __name__ == "__main__":
    ingestor = DataIngestor()
    ingestor.fetch_data()

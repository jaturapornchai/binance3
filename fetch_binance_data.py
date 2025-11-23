import requests
import pandas as pd
import time
from datetime import datetime, timedelta

def fetch_binance_data(symbol, interval, days, output_file=None):
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Calculate start and end times
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # Convert to milliseconds timestamp
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)
    
    all_data = []
    current_start = start_ts
    
    print(f"Fetching {symbol} {interval} data for the last {days} days...")
    
    while current_start < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ts,
            "limit": 1000  # Max limit per request
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            all_data.extend(data)
            
            # Update current_start to the close time of the last candle + 1ms
            last_candle_close_time = data[-1][6]
            current_start = last_candle_close_time + 1
            
            # Respect API rate limits
            time.sleep(0.1)
            
            print(f"Fetched {len(data)} candles. Total: {len(all_data)}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break
            
    # Create DataFrame
    columns = [
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Asset Volume", "Number of Trades",
        "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"
    ]
    
    df = pd.read_json(pd.io.json.dumps(all_data))
    df.columns = columns
    
    # Convert timestamps to readable dates
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
    df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")
    
    # Save to CSV
    filename = f"binance_{symbol.lower()}_{interval}_{days}days.csv"
    if output_file:
        filename = output_file
    else:
        filename = f"binance_{symbol.lower()}_{interval}_{days}days.csv"
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    print(f"Total rows: {len(df)}")

if __name__ == "__main__":
    # Install dependencies if needed (this is just a script, but good to know)
    # pip install    # Fetch 4h data for 365 days
    fetch_binance_data(symbol="BTCUSDT", interval="4h", days=365, output_file="binance_btcusdt_4h_365days.csv")

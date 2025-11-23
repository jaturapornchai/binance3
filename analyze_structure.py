import pandas as pd
import numpy as np
import glob

def get_hurst_exponent(time_series, max_lag=20):
    """Returns the Hurst Exponent of the time series"""
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def analyze_market(filepath):
    print(f"Analyzing {filepath}...")
    df = pd.read_csv(filepath)
    df['Open Time'] = pd.to_datetime(df['Open Time'])
    df.set_index('Open Time', inplace=True)
    
    # 1. Hurst Exponent
    # Calculate on log prices
    prices = np.log(df['Close'].values)
    hurst = get_hurst_exponent(prices, max_lag=100)
    
    print(f"\n=== Statistical Analysis ===")
    print(f"Hurst Exponent: {hurst:.4f}")
    if hurst < 0.45:
        print(">> Market is MEAN REVERTING (Prices tend to bounce back)")
    elif hurst > 0.55:
        print(">> Market is TRENDING (Prices tend to continue)")
    else:
        print(">> Market is RANDOM WALK (Hard to predict)")
        
    # 2. Volatility by Hour
    df['Hour'] = df.index.hour
    df['Return'] = df['Close'].pct_change().abs()
    hourly_vol = df.groupby('Hour')['Return'].mean() * 100
    
    print("\n=== Hourly Volatility Profile (UTC) ===")
    print(hourly_vol.to_string(float_format="%.3f%%"))
    
    best_hour = hourly_vol.idxmax()
    worst_hour = hourly_vol.idxmin()
    print(f"\nMost Volatile Hour: {best_hour}:00 UTC")
    print(f"Least Volatile Hour: {worst_hour}:00 UTC")
    
    # 3. Daily Returns Distribution
    df['Daily_Return'] = df['Close'].pct_change(96) # 96 candles = 24h
    skew = df['Daily_Return'].skew()
    print(f"\nReturn Skewness: {skew:.4f}")
    if skew > 0:
        print(">> Positive Skew: Big pumps are more common than big dumps")
    else:
        print(">> Negative Skew: Big crashes are more common (Flash crashes)")

if __name__ == "__main__":
    files = glob.glob("binance_btcusdt_15m_365days.csv")
    if not files:
        print("No data file found!")
    else:
        analyze_market(files[0])

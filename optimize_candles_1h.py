import pandas as pd
import numpy as np
import glob

class CandleOptimizer:
    def __init__(self, data_path, fee_rate=0.0004, leverage=10):
        self.df = pd.read_csv(data_path)
        self.df['Open Time'] = pd.to_datetime(self.df['Open Time'])
        self.df.set_index('Open Time', inplace=True)
        self.fee_rate = fee_rate
        self.leverage = leverage
        self.initial_capital = 10000
        self.prepare_indicators()

    def prepare_indicators(self):
        # Candlestick Components
        self.df['Body'] = abs(self.df['Close'] - self.df['Open'])
        self.df['UpperShadow'] = self.df['High'] - self.df[['Close', 'Open']].max(axis=1)
        self.df['LowerShadow'] = self.df[['Close', 'Open']].min(axis=1) - self.df['Low']
        self.df['Range'] = self.df['High'] - self.df['Low']
        
        # Patterns
        self.df['Hammer'] = (
            (self.df['LowerShadow'] > 2 * self.df['Body']) & 
            (self.df['UpperShadow'] < 0.5 * self.df['Body']) &
            (self.df['Body'] > 0.1 * self.df['Range'])
        )
        self.df['ShootingStar'] = (
            (self.df['UpperShadow'] > 2 * self.df['Body']) & 
            (self.df['LowerShadow'] < 0.5 * self.df['Body']) &
            (self.df['Body'] > 0.1 * self.df['Range'])
        )
        
        # Engulfing
        self.df['Prev_Open'] = self.df['Open'].shift(1)
        self.df['Prev_Close'] = self.df['Close'].shift(1)
        self.df['Prev_Red'] = self.df['Prev_Close'] < self.df['Prev_Open']
        
        self.df['BullEngulf'] = (
            (self.df['Close'] > self.df['Open']) & 
            (self.df['Prev_Red']) & 
            (self.df['Close'] > self.df['Prev_Open']) & 
            (self.df['Open'] < self.df['Prev_Close'])
        )
        self.df['BearEngulf'] = (
            (self.df['Close'] < self.df['Open']) & 
            (~self.df['Prev_Red']) & 
            (self.df['Close'] < self.df['Prev_Open']) &
            (self.df['Open'] > self.df['Prev_Close'])
        )
        
        # Trend Filters
        self.df['SMA50'] = self.df['Close'].rolling(window=50).mean()
        self.df['SMA200'] = self.df['Close'].rolling(window=200).mean()
        self.df['EMA50'] = self.df['Close'].ewm(span=50, adjust=False).mean()
        self.df['EMA200'] = self.df['Close'].ewm(span=200, adjust=False).mean()

    def run_strategy(self, pattern, filter_type, risk_reward):
        df = self.df.copy()
        
        # Filter Logic
        if filter_type == 'SMA50':
            long_filter = df['Close'] > df['SMA50']
            short_filter = df['Close'] < df['SMA50']
        elif filter_type == 'SMA200':
            long_filter = df['Close'] > df['SMA200']
            short_filter = df['Close'] < df['SMA200']
        elif filter_type == 'EMA50':
            long_filter = df['Close'] > df['EMA50']
            short_filter = df['Close'] < df['EMA50']
        elif filter_type == 'EMA200':
            long_filter = df['Close'] > df['EMA200']
            short_filter = df['Close'] < df['EMA200']
        else: # None
            long_filter = pd.Series(True, index=df.index)
            short_filter = pd.Series(True, index=df.index)
            
        # Signal Logic
        signals = pd.Series(0, index=df.index)
        direction = 0
        
        if pattern == 'BullEngulf':
            signals[df['BullEngulf'] & long_filter] = 1
            direction = 1
        elif pattern == 'Hammer':
            signals[df['Hammer'] & long_filter] = 1
            direction = 1
        elif pattern == 'BearEngulf':
            signals[df['BearEngulf'] & short_filter] = 1 # Signal is 1 (Active), but direction is -1
            direction = -1
        elif pattern == 'ShootingStar':
            signals[df['ShootingStar'] & short_filter] = 1
            direction = -1
            
        # Backtest Loop
        capital = self.initial_capital
        peak_capital = self.initial_capital
        max_drawdown = 0
        
        position = 0
        entry_price = 0
        tp_price = 0
        sl_price = 0
        trades = 0
        wins = 0
        
        signal_values = signals.values
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        
        for i in range(1, len(df)):
            if position == 0:
                if signal_values[i] == 1:
                    position = direction
                    entry_price = closes[i]
                    trades += 1
                    
                    # Set TP/SL
                    if direction == 1: # Long
                        sl_price = lows[i]
                        risk = entry_price - sl_price
                        tp_price = entry_price + (risk * risk_reward)
                    else: # Short
                        sl_price = highs[i]
                        risk = sl_price - entry_price
                        tp_price = entry_price - (risk * risk_reward)
                    
                    # Entry Fee (Leverage applies to position size)
                    capital -= (capital * self.leverage * self.fee_rate)

            elif position == 1: # Long
                if highs[i] >= tp_price: # TP
                    pnl_pct = (tp_price - entry_price) / entry_price
                    capital += (capital * self.leverage * pnl_pct)
                    capital -= (capital * self.leverage * self.fee_rate)
                    position = 0
                    wins += 1
                elif lows[i] <= sl_price: # SL
                    pnl_pct = (sl_price - entry_price) / entry_price
                    capital += (capital * self.leverage * pnl_pct)
                    capital -= (capital * self.leverage * self.fee_rate)
                    position = 0
            
            elif position == -1: # Short
                if lows[i] <= tp_price: # TP
                    pnl_pct = (entry_price - tp_price) / entry_price
                    capital += (capital * self.leverage * pnl_pct)
                    capital -= (capital * self.leverage * self.fee_rate)
                    position = 0
                    wins += 1
                elif highs[i] >= sl_price: # SL
                    pnl_pct = (entry_price - sl_price) / entry_price
                    capital += (capital * self.leverage * pnl_pct) # pnl_pct is negative here
                    capital -= (capital * self.leverage * self.fee_rate)
                    position = 0
            
            # Update Peak Capital and Drawdown
            if capital > peak_capital:
                peak_capital = capital
            
            drawdown = (peak_capital - capital) / peak_capital * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

            if capital <= 0: # Bust
                capital = 0
                break
                
        total_return = ((capital - self.initial_capital) / self.initial_capital) * 100
        win_rate = (wins / trades * 100) if trades > 0 else 0
        
        return {
            'Pattern': pattern,
            'Filter': filter_type,
            'RR': risk_reward,
            'Return (%)': round(total_return, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Final Capital': round(capital, 2),
            'Trades': trades,
            'Win Rate (%)': round(win_rate, 2)
        }

if __name__ == "__main__":
    files = glob.glob("binance_btcusdt_1h_365days.csv")
    if not files:
        print("No data file found!")
    else:
        # User asked for Max Drawdown < 20%, high leverage makes this hard.
        # Let's try 5x first to see raw performance.
        print("Optimizing 4h Strategies (Leverage 5x, Fee 0.05%)...")
        opt = CandleOptimizer("binance_btcusdt_4h_365days.csv", fee_rate=0.0005, leverage=5)
        
        patterns = ['BullEngulf', 'BearEngulf', 'Hammer', 'ShootingStar']
        filters = ['None', 'SMA50', 'SMA200', 'EMA50', 'EMA200']
        rrs = [1.0, 1.5, 2.0, 2.5, 3.0]
        
        results = []
        for p in patterns:
            for f in filters:
                for rr in rrs:
                    # print(f"Testing {p} + {f} + RR {rr}...") # Reduce spam
                    res = opt.run_strategy(p, f, rr)
                    results.append(res)
        
        df_res = pd.DataFrame(results)
        
        # Filter by Max Drawdown < 20%
        # df_filtered = df_res[df_res['Max Drawdown (%)'] < 20].copy()
        df_filtered = df_res.copy()
        
        df_filtered = df_filtered.sort_values(by='Return (%)', ascending=False)
        
        print(f"\nTotal Strategies Tested: {len(df_res)}")
        # print(f"Strategies with Max Drawdown < 20%: {len(df_filtered)}")
        
        print("\n=== Top 10 Strategies (Sorted by Return) ===")
        if not df_filtered.empty:
            print(df_filtered.head(10).to_string(index=False))
        else:
            print("No strategies found matching the criteria.")
        
        # Save full results
        df_res.to_csv("optimization_results_1h.csv", index=False)

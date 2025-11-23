import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, filepath, initial_capital=10000, fee_rate=0.0004):
        self.filepath = filepath
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.df = self.load_data()
        
    def load_data(self):
        df = pd.read_csv(self.filepath)
        df['Open Time'] = pd.to_datetime(df['Open Time'])
        df.set_index('Open Time', inplace=True)
        return df

    def run_strategy(self, strategy_name, **kwargs):
        df = self.df.copy()
        
        if strategy_name == 'SMA':
            df = self.strategy_sma(df, **kwargs)
        elif strategy_name == 'RSI':
            df = self.strategy_rsi(df, **kwargs)
        elif strategy_name == 'MACD':
            df = self.strategy_macd(df, **kwargs)
        elif strategy_name == 'Donchian':
            df = self.strategy_donchian(df, **kwargs)
        elif strategy_name == 'SMA_ADX':
            df = self.strategy_sma_adx(df, **kwargs)
        elif strategy_name == 'Donchian_ADX':
            df = self.strategy_donchian_adx(df, **kwargs)
        elif strategy_name == 'Fibo_Breakout':
            df = self.strategy_fibo_breakout(df, **kwargs)
        elif strategy_name == 'Reverse_Donchian':
            df = self.strategy_reverse_donchian(df, **kwargs)
        else:
            raise ValueError("Unknown strategy")
            
        return self.calculate_performance(df)

    def strategy_sma(self, df, fast=20, slow=50):
        df['Fast_SMA'] = df['Close'].rolling(window=fast).mean()
        df['Slow_SMA'] = df['Close'].rolling(window=slow).mean()
        
        df['Signal'] = 0
        df.loc[df['Fast_SMA'] > df['Slow_SMA'], 'Signal'] = 1  # Long
        df.loc[df['Fast_SMA'] < df['Slow_SMA'], 'Signal'] = -1 # Short
        return df

    def strategy_rsi(self, df, period=14, overbought=70, oversold=30):
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['Signal'] = 0
        # Simple Mean Reversion: Long if < Oversold, Short if > Overbought
        # Hold until signal changes (this is a simplified version)
        # A better version: Enter Long < 30, Exit > 50. Enter Short > 70, Exit < 50.
        
        # We will use a state machine approach for RSI to handle exits properly
        position = 0
        signals = []
        
        for rsi in df['RSI']:
            if position == 0:
                if rsi < oversold:
                    position = 1
                elif rsi > overbought:
                    position = -1
            elif position == 1:
                if rsi > 50: # Exit long
                    position = 0
            elif position == -1:
                if rsi < 50: # Exit short
                    position = 0
            signals.append(position)
            
        df['Signal'] = signals
        return df

    def strategy_macd(self, df, fast=12, slow=26, signal=9):
        exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        
        df['Signal'] = 0
        df.loc[df['MACD'] > df['Signal_Line'], 'Signal'] = 1
        df.loc[df['MACD'] < df['Signal_Line'], 'Signal'] = -1
        return df

    def calculate_adx(self, df, period=14):
        # True Range
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        
        # Directional Movement
        df['UpMove'] = df['High'] - df['High'].shift(1)
        df['DownMove'] = df['Low'].shift(1) - df['Low']
        
        df['+DM'] = 0
        df['-DM'] = 0
        
        df.loc[(df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), '+DM'] = df['UpMove']
        df.loc[(df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), '-DM'] = df['DownMove']
        
        # Wilder's Smoothing (alpha = 1/n)
        alpha = 1/period
        df['TR14'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()
        df['+DM14'] = df['+DM'].ewm(alpha=alpha, adjust=False).mean()
        df['-DM14'] = df['-DM'].ewm(alpha=alpha, adjust=False).mean()
        
        df['+DI'] = 100 * (df['+DM14'] / df['TR14'])
        df['-DI'] = 100 * (df['-DM14'] / df['TR14'])
        
        df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
        df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
        return df

    def strategy_sma_adx(self, df, fast=55, slow=170, adx_threshold=25):
        df = self.calculate_adx(df)
        df['Fast_SMA'] = df['Close'].rolling(window=fast).mean()
        df['Slow_SMA'] = df['Close'].rolling(window=slow).mean()
        
        df['Signal'] = 0
        # Long: Fast > Slow AND ADX > Threshold
        # Short: Fast < Slow AND ADX > Threshold
        # If ADX < Threshold, we want to be out of market (Signal 0) or hold previous?
        # "Regime Filter" usually means: only take NEW signals if ADX > 25.
        # But if we are already in a trend and ADX drops, should we exit?
        # Let's try strict filter: If ADX < 25, Signal = 0 (Exit/Flat).
        
        long_cond = (df['Fast_SMA'] > df['Slow_SMA']) & (df['ADX'] > adx_threshold)
        short_cond = (df['Fast_SMA'] < df['Slow_SMA']) & (df['ADX'] > adx_threshold)
        
        df.loc[long_cond, 'Signal'] = 1
        df.loc[short_cond, 'Signal'] = -1
        # Else 0
        
        return df

    def strategy_donchian_adx(self, df, period=140, adx_threshold=25):
        df = self.calculate_adx(df)
        df['Upper'] = df['High'].rolling(window=period).max().shift(1)
        df['Lower'] = df['Low'].rolling(window=period).min().shift(1)
        
        # Logic: Breakout + ADX > Threshold
        signals = pd.Series(np.nan, index=df.index)
        
        # Entry conditions
        long_entry = (df['Close'] > df['Upper']) & (df['ADX'] > adx_threshold)
        short_entry = (df['Close'] < df['Lower']) & (df['ADX'] > adx_threshold)
        
        signals[long_entry] = 1
        signals[short_entry] = -1
        
        # Exit if ADX drops? Or just hold?
        # Let's try: Exit if ADX < 20 (hysteresis) or just standard Donchian exit?
        # Strict filter: If ADX < Threshold, force exit (0).
        signals[df['ADX'] < adx_threshold] = 0
        
        df['Signal'] = signals.ffill().fillna(0)
        return df

    def strategy_donchian(self, df, period=140):
        df['Upper'] = df['High'].rolling(window=period).max().shift(1)
        df['Lower'] = df['Low'].rolling(window=period).min().shift(1)
        
        df['Signal'] = 0
        signals = pd.Series(np.nan, index=df.index)
        signals[df['Close'] > df['Upper']] = 1
        signals[df['Close'] < df['Lower']] = -1
        df['Signal'] = signals.ffill().fillna(0)
        return df

    def strategy_fibo_breakout(self, df, period=140):
        # Donchian Channels for Breakout
        df['Upper'] = df['High'].rolling(window=period).max().shift(1)
        df['Lower'] = df['Low'].rolling(window=period).min().shift(1)
        
        signals = [0] * len(df)
        position = 0 # 0, 1, -1
        entry_price = 0.0
        tp_price = 0.0
        sl_price = 0.0
        
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        uppers = df['Upper'].values
        lowers = df['Lower'].values
        
        # Iterate through candles
        for i in range(period, len(df)):
            current_close = closes[i]
            current_high = highs[i]
            current_low = lows[i]
            upper = uppers[i]
            lower = lowers[i]
            
            if position == 0:
                # Check for Long Breakout
                if current_close > upper:
                    position = 1
                    entry_price = current_close
                    # Swing Low is the 'lower' band value (approx lowest low of N)
                    swing_low = lower 
                    risk = entry_price - swing_low
                    tp_price = entry_price + (risk * 1.618)
                    sl_price = swing_low
                
                # Check for Short Breakout
                elif current_close < lower:
                    position = -1
                    entry_price = current_close
                    swing_high = upper
                    risk = swing_high - entry_price
                    tp_price = entry_price - (risk * 1.618)
                    sl_price = swing_high
            
            elif position == 1:
                # Check Exit
                if current_high >= tp_price: # Hit TP
                    position = 0
                elif current_low <= sl_price: # Hit SL
                    position = 0
            
            elif position == -1:
                # Check Exit
                if current_low <= tp_price: # Hit TP
                    position = 0
                elif current_high >= sl_price: # Hit SL
                    position = 0
            
            signals[i] = position
            
        df['Signal'] = signals
        return df

    def strategy_reverse_donchian(self, df, period=140):
        df['Upper'] = df['High'].rolling(window=period).max().shift(1)
        df['Lower'] = df['Low'].rolling(window=period).min().shift(1)
        
        df['Signal'] = 0
        # Reverse Logic: Short at Upper, Long at Lower (Betting on False Breakout)
        
        signals = pd.Series(np.nan, index=df.index)
        signals[df['Close'] > df['Upper']] = -1 # Short at Breakout
        signals[df['Close'] < df['Lower']] = 1  # Long at Breakdown
        df['Signal'] = signals.ffill().fillna(0)
        
        return df

    def calculate_performance(self, df):
        # Shift signal by 1 to avoid lookahead bias (trade happens next candle)
        df['Position'] = df['Signal'].shift(1)
        
        # Calculate returns
        df['Market_Return'] = df['Close'].pct_change()
        df['Strategy_Return'] = df['Market_Return'] * df['Position']
        
        # Apply fees on position change
        df['Trade_Count'] = df['Position'].diff().abs()
        # A change from 0 to 1 is 1 trade. 1 to -1 is 2 trades (close long, open short).
        # Fee is applied on the notional value. Approx: fee_rate * 2 (entry + exit) * number of turns?
        # Let's do it candle by candle for accuracy or just subtract fee on trade execution.
        
        # Simplified fee: subtract fee_rate whenever position changes
        # Note: This assumes full capital deployment.
        df['Strategy_Return'] -= df['Trade_Count'] * self.fee_rate
        
        df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
        
        total_return = df['Cumulative_Return'].iloc[-1] - 1
        trades = df['Trade_Count'].sum()
        
        # Max Drawdown
        cumulative = df['Cumulative_Return']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            "Total Return (%)": round(total_return * 100, 2),
            "Final Capital": round(self.initial_capital * (1 + total_return), 2),
            "Max Drawdown (%)": round(max_drawdown * 100, 2),
            "Total Trades": int(trades)
        }

if __name__ == "__main__":
    # Find the CSV file
    import glob
    files = glob.glob("binance_btcusdt_15m_365days.csv")
    if not files:
        print("No data file found!")
        exit()
    
    filepath = files[0]
    print(f"Backtesting on {filepath}...")
    
    bt = Backtester(filepath)
    
    results = []
    
    # Test SMA
    res_sma = bt.run_strategy('SMA', fast=20, slow=50)
    res_sma['Strategy'] = 'SMA (20, 50)'
    results.append(res_sma)

    # Test Optimized SMA
    res_opt_sma = bt.run_strategy('SMA', fast=55, slow=170)
    res_opt_sma['Strategy'] = 'SMA (55, 170) [Low Fee Winner]'
    results.append(res_opt_sma)
    
    # Test Donchian (High Fee Winner)
    res_donchian = bt.run_strategy('Donchian', period=140)
    res_donchian['Strategy'] = 'Donchian (P140)'
    results.append(res_donchian)

    # Test SMA + ADX
    res_sma_adx = bt.run_strategy('SMA_ADX', fast=55, slow=170, adx_threshold=25)
    res_sma_adx['Strategy'] = 'SMA (55, 170) + ADX>25'
    results.append(res_sma_adx)

    # Test Donchian + ADX
    res_don_adx = bt.run_strategy('Donchian_ADX', period=140, adx_threshold=25)
    res_don_adx['Strategy'] = 'Donchian (P140) + ADX>25'
    results.append(res_don_adx)

    # Test Fibo Breakout
    res_fibo = bt.run_strategy('Fibo_Breakout', period=140)
    res_fibo['Strategy'] = 'Fibo Breakout (P140, TP 1.618)'
    results.append(res_fibo)

    # Test Reverse Donchian
    res_rev_don = bt.run_strategy('Reverse_Donchian', period=140)
    res_rev_don['Strategy'] = 'Reverse Donchian (P140) [Mean Reversion]'
    results.append(res_rev_don)
    
    # Test RSI
    res_rsi = bt.run_strategy('RSI', period=14)
    res_rsi['Strategy'] = 'RSI (14, 30/70)'
    results.append(res_rsi)
    
    # Test MACD
    res_macd = bt.run_strategy('MACD')
    res_macd['Strategy'] = 'MACD (12, 26, 9)'
    results.append(res_macd)
    
    # Create summary DataFrame
    summary = pd.DataFrame(results)
    summary = summary[['Strategy', 'Total Return (%)', 'Final Capital', 'Max Drawdown (%)', 'Total Trades']]
    
    print("\n=== Backtest Results (180 Days, 15m) ===")
    print(summary.to_string(index=False))
    
    best_strategy = summary.loc[summary['Total Return (%)'].idxmax()]
    print(f"\nBest Strategy: {best_strategy['Strategy']} with {best_strategy['Total Return (%)']}% Return")

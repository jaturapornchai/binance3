import pandas as pd
import numpy as np
import glob

class SuperTrendBacktester:
    def __init__(self, data_path, fee_rate=0.0004, leverage=10):
        self.df = pd.read_csv(data_path)
        self.df['Open Time'] = pd.to_datetime(self.df['Open Time'])
        self.df.set_index('Open Time', inplace=True)
        self.fee_rate = fee_rate
        self.leverage = leverage
        self.initial_capital = 10000
        self.prepare_indicators()

    def prepare_indicators(self):
        # 1. SuperTrend (10, 3)
        high = self.df['High']
        low = self.df['Low']
        close = self.df['Close']
        
        # ATR
        tr1 = abs(high - low)
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(10).mean()
        
        # Basic Bands
        hl2 = (high + low) / 2
        basic_upper = hl2 + (3 * atr)
        basic_lower = hl2 - (3 * atr)
        
        # Final Bands
        final_upper = pd.Series(0.0, index=self.df.index)
        final_lower = pd.Series(0.0, index=self.df.index)
        supertrend = pd.Series(0.0, index=self.df.index)
        
        # Initialize at index 10 (when ATR is valid)
        for i in range(10):
            final_upper.iloc[i] = basic_upper.iloc[i] if not pd.isna(basic_upper.iloc[i]) else 0
            final_lower.iloc[i] = basic_lower.iloc[i] if not pd.isna(basic_lower.iloc[i]) else 0
            supertrend.iloc[i] = final_upper.iloc[i]

        for i in range(10, len(self.df)):
            # Upper Band
            if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
                final_upper.iloc[i] = basic_upper.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i-1]
                
            # Lower Band
            if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
                final_lower.iloc[i] = basic_lower.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i-1]
                
            # SuperTrend
            if supertrend.iloc[i-1] == final_upper.iloc[i-1]:
                if close.iloc[i] <= final_upper.iloc[i]:
                    supertrend.iloc[i] = final_upper.iloc[i]
                else:
                    supertrend.iloc[i] = final_lower.iloc[i]
            else:
                if close.iloc[i] >= final_lower.iloc[i]:
                    supertrend.iloc[i] = final_lower.iloc[i]
                else:
                    supertrend.iloc[i] = final_upper.iloc[i]
                    
        self.df['SuperTrend'] = supertrend
        # Fill NaN at start
        self.df['SuperTrend'] = self.df['SuperTrend'].fillna(method='bfill')
        self.df['Trend'] = np.where(self.df['Close'] > self.df['SuperTrend'], 1, -1)
        
        # 2. MACD (12, 26, 9)
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        self.df['MACD'] = macd
        self.df['MACD_Signal'] = signal
        self.df['MACD_Hist'] = macd - signal
        
        # 3. StochRSI (14, 14, 3, 3)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        stoch_rsi = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
        self.df['StochK'] = stoch_rsi.rolling(3).mean() * 100
        self.df['StochD'] = self.df['StochK'].rolling(3).mean()

    def run_strategy(self, risk_reward=1.5):
        df = self.df.copy()
        
        # Signals
        # Long: Trend=1 (Green), MACD > Signal, StochK < 80 (Not Overbought)
        # Short: Trend=-1 (Red), MACD < Signal, StochK > 20 (Not Oversold)
        
        long_cond = (
            (df['Trend'] == 1) & 
            (df['MACD'] > df['MACD_Signal']) & 
            (df['StochK'] < 80) &
            (df['StochK'] > df['StochD']) # Cross up
        )
        
        short_cond = (
            (df['Trend'] == -1) & 
            (df['MACD'] < df['MACD_Signal']) & 
            (df['StochK'] > 20) &
            (df['StochK'] < df['StochD']) # Cross down
        )
        
        print("Debug Indicators:")
        print(df[['Close', 'SuperTrend', 'Trend', 'MACD', 'MACD_Signal', 'StochK', 'StochD']].tail(10))
        
        print(f"Total Long Signals: {long_cond.sum()}")
        print(f"Total Short Signals: {short_cond.sum()}")
        
        signals = pd.Series(0, index=df.index)
        signals[long_cond] = 1
        signals[short_cond] = -1 # Note: Logic handles this
        
        # Backtest Loop
        capital = self.initial_capital
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
        supertrends = df['SuperTrend'].values
        
        for i in range(1, len(df)):
            # Check Exit first
            if position == 1: # Long
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
                    capital += (capital * self.leverage * pnl_pct)
                    capital -= (capital * self.leverage * self.fee_rate)
                    position = 0

            # Check Entry (only if no position)
            if position == 0:
                if signal_values[i] == 1: # Long Signal
                    position = 1
                    entry_price = closes[i]
                    trades += 1
                    
                    sl_price = supertrends[i]
                    risk = entry_price - sl_price
                    if risk <= 0: 
                        # Fallback SL if SuperTrend is weird
                        sl_price = entry_price * 0.99
                        risk = entry_price - sl_price
                        
                    tp_price = entry_price + (risk * risk_reward)
                    
                    capital -= (capital * self.leverage * self.fee_rate)

                elif signal_values[i] == -1: # Short Signal
                    position = -1
                    entry_price = closes[i]
                    trades += 1
                    
                    sl_price = supertrends[i]
                    risk = sl_price - entry_price
                    if risk <= 0:
                         # Fallback SL
                        sl_price = entry_price * 1.01
                        risk = sl_price - entry_price
                        
                    tp_price = entry_price - (risk * risk_reward)
                    
                    capital -= (capital * self.leverage * self.fee_rate)
                    
            if capital <= 0:
                capital = 0
                break
                
        total_return = ((capital - self.initial_capital) / self.initial_capital) * 100
        win_rate = (wins / trades * 100) if trades > 0 else 0
        
        return {
            'RR': risk_reward,
            'Return (%)': round(total_return, 2),
            'Final Capital': round(capital, 2),
            'Trades': trades,
            'Win Rate (%)': round(win_rate, 2)
        }

if __name__ == "__main__":
    files = glob.glob("binance_btcusdt_1h_365days.csv")
    if not files:
        print("No data file found!")
    else:
        print("Backtesting SuperTrend Strategy (1h, Leverage 1x)...")
        bt = SuperTrendBacktester(files[0], leverage=1)
        
        rrs = [1.0, 1.5, 2.0]
        
        results = []
        for rr in rrs:
            print(f"Testing RR {rr}...")
            res = bt.run_strategy(rr)
            results.append(res)
            
        res_df = pd.DataFrame(results)
        print("\n=== SuperTrend Results ===")
        print(res_df.to_string(index=False))

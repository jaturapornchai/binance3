import pandas as pd
import numpy as np
import glob

class OrderFlowBacktester:
    def __init__(self, data_path, fee_rate=0.0004, leverage=1):
        self.df = pd.read_csv(data_path)
        self.df['Open Time'] = pd.to_datetime(self.df['Open Time'])
        self.df.set_index('Open Time', inplace=True)
        self.fee_rate = fee_rate
        self.leverage = leverage
        self.initial_capital = 10000
        self.prepare_indicators()

    def prepare_indicators(self):
        # Volume Delta Calculation
        # Taker Buy Base Asset Volume is the volume of buy orders that filled immediately (market buys)
        # Total Volume is Buy + Sell
        # So Sell Volume = Total - Buy
        
        self.df['Buy_Vol'] = self.df['Taker Buy Base Asset Volume']
        self.df['Sell_Vol'] = self.df['Volume'] - self.df['Buy_Vol']
        self.df['Delta'] = self.df['Buy_Vol'] - self.df['Sell_Vol']
        
        # Cumulative Delta (Optional, for trend)
        self.df['CVD'] = self.df['Delta'].cumsum()
        
        # Trend Filter
        self.df['SMA50'] = self.df['Close'].rolling(window=50).mean()
        
        # Normalize Delta (Z-Score) to find significant spikes
        self.df['Delta_Mean'] = self.df['Delta'].rolling(window=50).mean()
        self.df['Delta_Std'] = self.df['Delta'].rolling(window=50).std()
        self.df['Delta_Z'] = (self.df['Delta'] - self.df['Delta_Mean']) / self.df['Delta_Std']

    def run_strategy(self, risk_reward=1.5, z_threshold=1.0):
        df = self.df.copy()
        
        # Signals
        # Long: Delta Z-Score > Threshold (Strong Buying) AND Price > SMA50 (Uptrend)
        # Short: Delta Z-Score < -Threshold (Strong Selling) AND Price < SMA50 (Downtrend)
        
        long_cond = (
            (df['Delta_Z'] > z_threshold) & 
            (df['Close'] > df['SMA50'])
        )
        
        short_cond = (
            (df['Delta_Z'] < -z_threshold) & 
            (df['Close'] < df['SMA50'])
        )
        
        signals = pd.Series(0, index=df.index)
        signals[long_cond] = 1
        signals[short_cond] = -1
        
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
        atr = (df['High'] - df['Low']).rolling(14).mean().values # For SL
        
        for i in range(50, len(df)): # Start after SMA50
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
                current_atr = atr[i]
                if np.isnan(current_atr): continue

                if signal_values[i] == 1: # Long Signal
                    position = 1
                    entry_price = closes[i]
                    trades += 1
                    
                    # SL based on ATR (Volatility)
                    sl_dist = current_atr * 2.0
                    sl_price = entry_price - sl_dist
                    risk = entry_price - sl_price
                    tp_price = entry_price + (risk * risk_reward)
                    
                    capital -= (capital * self.leverage * self.fee_rate)

                elif signal_values[i] == -1: # Short Signal
                    position = -1
                    entry_price = closes[i]
                    trades += 1
                    
                    sl_dist = current_atr * 2.0
                    sl_price = entry_price + sl_dist
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
            'Z-Thresh': z_threshold,
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
        print("Backtesting Order Flow Strategy (4h, Leverage 5x, Fee 0.05%)...")
        # Official Binance Taker Fee for VIP 0 is 0.05%
        bt = OrderFlowBacktester("binance_btcusdt_4h_365days.csv", fee_rate=0.0005, leverage=5)
        
        rrs = [1.5, 2.0, 3.0]
        thresholds = [0.5, 1.0, 1.5]
        
        results = []
        for rr in rrs:
            for th in thresholds:
                print(f"Testing RR {rr}, Z-Thresh {th}...")
                res = bt.run_strategy(rr, th)
                results.append(res)
            
        res_df = pd.DataFrame(results)
        print("\n=== Order Flow Results ===")
        print(res_df.sort_values(by='Return (%)', ascending=False).to_string(index=False))

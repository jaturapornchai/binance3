import pandas as pd
import numpy as np
import glob

class OrderFlowStrategy:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.df['Open Time'] = pd.to_datetime(self.df['Open Time'])
        self.df.set_index('Open Time', inplace=True)
        
        # Strategy Parameters
        self.leverage = 5
        self.fee_rate = 0.0005 # 0.05%
        self.initial_capital = 10000
        self.risk_reward = 3.0
        self.z_threshold = 1.5
        
        self.prepare_indicators()

    def prepare_indicators(self):
        print("Calculating Indicators...")
        # 1. Volume Delta (The Core Logic)
        # Taker Buy Volume = Aggressive Buyers
        # Total Volume - Taker Buy = Aggressive Sellers
        self.df['Buy_Vol'] = self.df['Taker Buy Base Asset Volume']
        self.df['Sell_Vol'] = self.df['Volume'] - self.df['Buy_Vol']
        self.df['Delta'] = self.df['Buy_Vol'] - self.df['Sell_Vol']
        
        # 2. Trend Filter (SMA 50)
        self.df['SMA50'] = self.df['Close'].rolling(window=50).mean()
        
        # 3. Z-Score Normalization (Finding Spikes)
        self.df['Delta_Mean'] = self.df['Delta'].rolling(window=50).mean()
        self.df['Delta_Std'] = self.df['Delta'].rolling(window=50).std()
        self.df['Delta_Z'] = (self.df['Delta'] - self.df['Delta_Mean']) / self.df['Delta_Std']
        
        # 4. ATR for Stop Loss
        self.df['High_Low'] = self.df['High'] - self.df['Low']
        self.df['High_Close'] = abs(self.df['High'] - self.df['Close'].shift())
        self.df['Low_Close'] = abs(self.df['Low'] - self.df['Close'].shift())
        self.df['TR'] = self.df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        self.df['ATR'] = self.df['TR'].rolling(window=14).mean()

    def run_backtest(self):
        print(f"Running Backtest: 1h Timeframe, Leverage {self.leverage}x, RR {self.risk_reward}, Z-Thresh {self.z_threshold}")
        df = self.df.copy()
        
        # Signal Generation
        long_cond = (df['Delta_Z'] > self.z_threshold) & (df['Close'] > df['SMA50'])
        short_cond = (df['Delta_Z'] < -self.z_threshold) & (df['Close'] < df['SMA50'])
        
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
        atr = df['ATR'].values
        
        for i in range(50, len(df)):
            # Check Exit
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

            # Check Entry
            if position == 0:
                current_atr = atr[i]
                if np.isnan(current_atr): continue

                if signal_values[i] == 1: # Long
                    position = 1
                    entry_price = closes[i]
                    trades += 1
                    
                    # SL = 2 * ATR
                    sl_dist = current_atr * 2.0
                    sl_price = entry_price - sl_dist
                    risk = entry_price - sl_price
                    tp_price = entry_price + (risk * self.risk_reward)
                    
                    capital -= (capital * self.leverage * self.fee_rate)

                elif signal_values[i] == -1: # Short
                    position = -1
                    entry_price = closes[i]
                    trades += 1
                    
                    sl_dist = current_atr * 2.0
                    sl_price = entry_price + sl_dist
                    risk = sl_price - entry_price
                    tp_price = entry_price - (risk * self.risk_reward)
                    
                    capital -= (capital * self.leverage * self.fee_rate)
                    
            if capital <= 0:
                capital = 0
                break
                
        total_return = ((capital - self.initial_capital) / self.initial_capital) * 100
        win_rate = (wins / trades * 100) if trades > 0 else 0
        
        print("\n=== Final Results ===")
        print(f"Initial Capital: ${self.initial_capital}")
        print(f"Final Capital:   ${round(capital, 2)}")
        print(f"Total Return:    {round(total_return, 2)}%")
        print(f"Total Trades:    {trades}")
        print(f"Win Rate:        {round(win_rate, 2)}%")
        print("=====================")

if __name__ == "__main__":
    files = glob.glob("binance_btcusdt_1h_365days.csv")
    if not files:
        print("No data file found! Please run fetch_binance_data.py first.")
    else:
        strategy = OrderFlowStrategy(files[0])
        strategy.run_backtest()

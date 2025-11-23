import pandas as pd
import numpy as np
import glob

class ScalpingOptimizer:
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
        
        # Bollinger Bands (20, 2.0)
        self.df['SMA20'] = self.df['Close'].rolling(window=20).mean()
        self.df['STD20'] = self.df['Close'].rolling(window=20).std()
        self.df['UpperBB'] = self.df['SMA20'] + (2 * self.df['STD20'])
        self.df['LowerBB'] = self.df['SMA20'] - (2 * self.df['STD20'])

    def run_strategy(self, strategy_type, tp_pct, sl_pct):
        df = self.df.copy()
        
        signals = pd.Series(0, index=df.index)
        
        if strategy_type == 'BB_Hammer':
            # Buy: Hammer AND Low < LowerBB
            buy_cond = df['Hammer'] & (df['Low'] < df['LowerBB'])
            signals[buy_cond] = 1
            direction = 1
            
        elif strategy_type == 'BB_ShootingStar':
            # Sell: ShootingStar AND High > UpperBB
            sell_cond = df['ShootingStar'] & (df['High'] > df['UpperBB'])
            signals[sell_cond] = 1
            direction = -1
            
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
        
        for i in range(1, len(df)):
            if position == 0:
                if signal_values[i] == 1:
                    position = direction
                    entry_price = closes[i]
                    trades += 1
                    
                    # Set Fixed TP/SL
                    if direction == 1: # Long
                        tp_price = entry_price * (1 + tp_pct)
                        sl_price = entry_price * (1 - sl_pct)
                    else: # Short
                        tp_price = entry_price * (1 - tp_pct)
                        sl_price = entry_price * (1 + sl_pct)
                    
                    # Entry Fee
                    capital -= (capital * self.leverage * self.fee_rate)

            elif position == 1: # Long
                if highs[i] >= tp_price: # TP
                    pnl_pct = tp_pct
                    capital += (capital * self.leverage * pnl_pct)
                    capital -= (capital * self.leverage * self.fee_rate)
                    position = 0
                    wins += 1
                elif lows[i] <= sl_price: # SL
                    pnl_pct = -sl_pct
                    capital += (capital * self.leverage * pnl_pct)
                    capital -= (capital * self.leverage * self.fee_rate)
                    position = 0
            
            elif position == -1: # Short
                if lows[i] <= tp_price: # TP
                    pnl_pct = tp_pct
                    capital += (capital * self.leverage * pnl_pct)
                    capital -= (capital * self.leverage * self.fee_rate)
                    position = 0
                    wins += 1
                elif highs[i] >= sl_price: # SL
                    pnl_pct = -sl_pct
                    capital += (capital * self.leverage * pnl_pct)
                    capital -= (capital * self.leverage * self.fee_rate)
                    position = 0
                    
            if capital <= 0:
                capital = 0
                break
                
        total_return = ((capital - self.initial_capital) / self.initial_capital) * 100
        win_rate = (wins / trades * 100) if trades > 0 else 0
        
        return {
            'Strategy': strategy_type,
            'TP': tp_pct,
            'SL': sl_pct,
            'Return (%)': round(total_return, 2),
            'Final Capital': round(capital, 2),
            'Trades': trades,
            'Win Rate (%)': round(win_rate, 2)
        }

if __name__ == "__main__":
    files = glob.glob("binance_btcusdt_15m_365days.csv")
    if not files:
        print("No data file found!")
    else:
        print("Optimizing Scalping Strategies (15m, Leverage 5x, Fee 0.05%)...")
        opt = ScalpingOptimizer(files[0], fee_rate=0.0005, leverage=5)
        
        strategies = ['BB_Hammer', 'BB_ShootingStar']
        tp_targets = [0.005, 0.01, 0.015] # 0.5%, 1.0%, 1.5%
        sl_targets = [0.005, 0.01] # 0.5%, 1.0%
        
        results = []
        for s in strategies:
            for tp in tp_targets:
                for sl in sl_targets:
                    print(f"Testing {s} + TP {tp} + SL {sl}...")
                    res = opt.run_strategy(s, tp, sl)
                    results.append(res)
        
        df_res = pd.DataFrame(results)
        df_res = df_res.sort_values(by='Return (%)', ascending=False)
        
        print("\n=== Top Scalping Strategies ===")
        print(df_res.to_string(index=False))

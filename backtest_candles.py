import pandas as pd
import numpy as np
import glob

class CandleBacktester:
    def __init__(self, data_path, fee_rate=0.0004):
        self.df = pd.read_csv(data_path)
        self.df['Open Time'] = pd.to_datetime(self.df['Open Time'])
        self.df.set_index('Open Time', inplace=True)
        self.fee_rate = fee_rate
        self.initial_capital = 10000

    def identify_patterns(self, df):
        # Helper columns
        df['Body'] = abs(df['Close'] - df['Open'])
        df['UpperShadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
        df['LowerShadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']
        df['Range'] = df['High'] - df['Low']
        
        # 1. Hammer (Bullish)
        # Small body, long lower shadow (> 2x body), small upper shadow
        df['Hammer'] = (
            (df['LowerShadow'] > 2 * df['Body']) & 
            (df['UpperShadow'] < 0.5 * df['Body']) &
            (df['Body'] > 0.1 * df['Range']) # Avoid Doji
        )
        
        # 2. Shooting Star (Bearish)
        # Small body, long upper shadow (> 2x body), small lower shadow
        df['ShootingStar'] = (
            (df['UpperShadow'] > 2 * df['Body']) & 
            (df['LowerShadow'] < 0.5 * df['Body']) &
            (df['Body'] > 0.1 * df['Range'])
        )
        
        # 3. Bullish Engulfing
        # Previous candle red, Current candle green, Current body engulfs previous body
        df['Prev_Open'] = df['Open'].shift(1)
        df['Prev_Close'] = df['Close'].shift(1)
        df['Prev_Body'] = abs(df['Prev_Close'] - df['Prev_Open'])
        df['Prev_Red'] = df['Prev_Close'] < df['Prev_Open']
        
        df['BullEngulf'] = (
            (df['Close'] > df['Open']) & # Green
            (df['Prev_Red']) & # Prev Red
            (df['Close'] > df['Prev_Open']) & 
            (df['Open'] < df['Prev_Close'])
        )
        
        # 4. Bearish Engulfing
        df['BearEngulf'] = (
            (df['Close'] < df['Open']) & # Red
            (~df['Prev_Red']) & # Prev Green
            (df['Close'] < df['Prev_Open']) &
            (df['Open'] > df['Prev_Close'])
        )
        
        # 5. Trend Filter (EMA 200)
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        return df

    def run_backtest(self, pattern_name, risk_reward=1.5, use_trend_filter=False):
        df = self.df.copy()
        df = self.identify_patterns(df)
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        tp_price = 0
        sl_price = 0
        trades = 0
        wins = 0
        
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        emas = df['EMA_200'].values
        
        # Pattern Signals
        if pattern_name == 'Hammer':
            signals = df['Hammer'].values
            direction = 1
        elif pattern_name == 'ShootingStar':
            signals = df['ShootingStar'].values
            direction = -1
        elif pattern_name == 'BullEngulf':
            signals = df['BullEngulf'].values
            direction = 1
        elif pattern_name == 'BearEngulf':
            signals = df['BearEngulf'].values
            direction = -1
        else:
            return {}

        peak_capital = self.initial_capital
        max_drawdown = 0
        
        for i in range(1, len(df)):
            current_close = closes[i]
            current_high = highs[i]
            current_low = lows[i]
            current_ema = emas[i]
            
            if position == 0:
                if signals[i]: # Pattern found
                    # Apply Trend Filter
                    if use_trend_filter:
                        if direction == 1 and current_close < current_ema:
                            continue # Skip Long if below EMA
                        if direction == -1 and current_close > current_ema:
                            continue # Skip Short if above EMA
                            
                    position = direction
                    entry_price = current_close
                    trades += 1
                    
                    # Set TP/SL
                    range_size = current_high - current_low
                    if direction == 1: # Long
                        sl_price = current_low # Low of pattern candle
                        risk = entry_price - sl_price
                        tp_price = entry_price + (risk * risk_reward)
                    else: # Short
                        sl_price = current_high # High of pattern candle
                        risk = sl_price - entry_price
                        tp_price = entry_price - (risk * risk_reward)
                    
                    # Apply Entry Fee
                    capital -= (capital * self.fee_rate)

            elif position == 1: # Long
                if current_high >= tp_price: # TP Hit
                    capital += capital * ((tp_price - entry_price) / entry_price)
                    capital -= (capital * self.fee_rate) # Exit Fee
                    position = 0
                    wins += 1
                elif current_low <= sl_price: # SL Hit
                    capital += capital * ((sl_price - entry_price) / entry_price)
                    capital -= (capital * self.fee_rate) # Exit Fee
                    position = 0
            
            elif position == -1: # Short
                if current_low <= tp_price: # TP Hit
                    capital += capital * ((entry_price - tp_price) / entry_price) # Profit logic for short
                    capital -= (capital * self.fee_rate) # Exit Fee
                    position = 0
                    wins += 1
                elif current_high >= sl_price: # SL Hit
                    capital += capital * ((entry_price - sl_price) / entry_price) # Loss logic for short
                    capital -= (capital * self.fee_rate) # Exit Fee
                    position = 0
            
            # Update Peak Capital and Drawdown
            if capital > peak_capital:
                peak_capital = capital
            
            drawdown = (peak_capital - capital) / peak_capital * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        total_return = ((capital - self.initial_capital) / self.initial_capital) * 100
        win_rate = (wins / trades * 100) if trades > 0 else 0
        
        return {
            'Pattern': pattern_name,
            'Trend Filter': 'On' if use_trend_filter else 'Off',
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
        print("Backtesting Candlestick Patterns (1h) (Risk:Reward = 1:1.5)...")
        bt = CandleBacktester(files[0])
        
        patterns = ['Hammer', 'ShootingStar', 'BullEngulf', 'BearEngulf']
        risk_rewards = [1.5, 2.0, 2.5, 3.0]
        results = []
        
        for rr in risk_rewards:
            for p in patterns:
                # Run with trend filter
                res = bt.run_backtest(p, risk_reward=rr, use_trend_filter=True)
                res['Risk:Reward'] = rr
                results.append(res)
            
        res_df = pd.DataFrame(results)
        
        # Reorder columns
        cols = ['Pattern', 'Risk:Reward', 'Trend Filter', 'Return (%)', 'Max Drawdown (%)', 'Final Capital', 'Trades', 'Win Rate (%)']
        res_df = res_df[cols]
        
        # Filter by Max Drawdown < 20%
        filtered_df = res_df[res_df['Max Drawdown (%)'] < 20]
        
        print("\nAll Results (Trend Filter ON):")
        print(res_df.to_string(index=False))
        
        print("\nResults with Max Drawdown < 20%:")
        if not filtered_df.empty:
            # Sort by Return
            filtered_df = filtered_df.sort_values(by='Return (%)', ascending=False)
            print(filtered_df.to_string(index=False))
        else:
            print("No strategies found with Max Drawdown < 20%")

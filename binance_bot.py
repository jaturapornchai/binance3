import ccxt
import pandas as pd
import numpy as np
import time
import config
from datetime import datetime
import sys

class BinanceBot:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': config.API_KEY,
            'secret': config.API_SECRET,
            'options': {
                'defaultType': 'future',  # Futures trading
            },
            'enableRateLimit': True,
        })
        
        if config.IS_TESTNET:
            self.exchange.set_sandbox_mode(True)
            print("⚠️ Running in SANDBOX Mode (Testnet)")

        self.symbol = config.SYMBOL
        self.timeframe = config.TIMEFRAME
        self.leverage = config.LEVERAGE
        
        # Initialize
        self.check_connection()
        self.set_leverage()

    def check_connection(self):
        try:
            self.exchange.fetch_time()
            print("✅ Connected to Binance Futures")
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            sys.exit(1)

    def set_leverage(self):
        try:
            # Binance requires setting leverage for the symbol
            self.exchange.set_leverage(self.leverage, self.symbol)
            print(f"✅ Leverage set to {self.leverage}x for {self.symbol}")
        except Exception as e:
            print(f"⚠️ Error setting leverage: {e}")

    def fetch_data(self, limit=400):
        try:
            # Fetch OHLCV
            bars = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Fetch Taker Buy Volume (Need to fetch separate kline data or use fetch_ohlcv with params if supported)
            # CCXT fetch_ohlcv standard doesn't always include taker volume.
            # For Binance, we can use the public API or just fetch raw klines if needed.
            # However, ccxt binance fetch_ohlcv returns standard 6 columns.
            # We need Taker Buy Base Asset Volume.
            
            # Alternative: Use fetch_klines which might return more info, or use a direct request.
            # For simplicity and robustness with CCXT, we might need to extend or use a specific method.
            # Binance public API klines returns:
            # [Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, Number of trades, Taker buy base asset volume, Taker buy quote asset volume, Ignore]
            
            # Let's use the exchange's implicit method for klines to get all columns
            raw_klines = self.exchange.public_get_klines({
                'symbol': self.exchange.market_id(self.symbol),
                'interval': self.timeframe,
                'limit': limit
            })
            
            # Parse raw klines
            # Index 9 is Taker buy base asset volume
            df_full = pd.DataFrame(raw_klines, columns=[
                'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                'Close time', 'Quote asset volume', 'Number of trades', 
                'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
            ])
            
            df_full['timestamp'] = pd.to_datetime(df_full['Open time'].astype('int64'), unit='ms')
            df_full['open'] = df_full['Open'].astype(float)
            df_full['high'] = df_full['High'].astype(float)
            df_full['low'] = df_full['Low'].astype(float)
            df_full['close'] = df_full['Close'].astype(float)
            df_full['volume'] = df_full['Volume'].astype(float)
            df_full['taker_buy_vol'] = df_full['Taker buy base asset volume'].astype(float)
            
            return df_full
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None

    def calculate_signals(self, df):
        # 1. Volume Delta
        df['buy_vol'] = df['taker_buy_vol']
        df['sell_vol'] = df['volume'] - df['buy_vol']
        df['delta'] = df['buy_vol'] - df['sell_vol']
        
        # 2. Trend (SMA 50)
        df['sma50'] = df['close'].rolling(window=50).mean()
        
        # 3. Z-Score
        df['delta_mean'] = df['delta'].rolling(window=50).mean()
        df['delta_std'] = df['delta'].rolling(window=50).std()
        df['delta_z'] = (df['delta'] - df['delta_mean']) / df['delta_std']
        
        # 4. ATR (for SL)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            )
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        return df

    def get_position(self):
        try:
            positions = self.exchange.fetch_positions([self.symbol])
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    size = float(pos['contracts'])
                    if size > 0:
                        return 'long', size, float(pos['entryPrice'])
                    elif size < 0: # CCXT usually returns positive size with side, but check just in case
                        return 'short', abs(size), float(pos['entryPrice'])
                    
                    # Some exchanges return side
                    if pos['side'] == 'long' and size > 0: return 'long', size, float(pos['entryPrice'])
                    if pos['side'] == 'short' and size > 0: return 'short', size, float(pos['entryPrice'])
                    
            return None, 0, 0
        except Exception as e:
            print(f"⚠️ Error fetching position: {e}")
            return None, 0, 0

    def execute_trade(self, signal, close_price, atr):
        balance = self.exchange.fetch_balance()['USDT']['free']
        risk_amt = balance * config.RISK_PER_TRADE
        
        # Calculate Position Size
        # Stop Loss Distance
        sl_dist = atr * 2.0
        
        # Size = Risk Amount / SL Distance
        # Example: Risk $100. SL Dist $500. Size = 0.2 BTC.
        # However, with leverage, we need to be careful.
        # Simplified: Use % of balance with leverage.
        
        # Let's stick to the backtest logic:
        # We want to risk X% of equity.
        # But for simplicity in this bot version, let's use a fixed size logic or simple leverage calculation.
        # Position Value = Balance * Leverage (Full degen? No, let's be safe)
        # Let's use: Position Value = Balance * 0.5 * Leverage (Use half available margin)
        
        position_value = balance * 0.90 * self.leverage # Use 90% of balance
        amount = position_value / close_price
        
        print(f"🚀 Executing {signal.upper()} | Price: {close_price} | Amount: {amount:.4f}")
        
        try:
            if signal == 'long':
                # Market Buy
                order = self.exchange.create_market_buy_order(self.symbol, amount)
                print(f"✅ BUY Order Filled: {order['id']}")
                
                # Set TP/SL
                entry_price = float(order['average']) if order['average'] else close_price
                sl_price = entry_price - sl_dist
                tp_price = entry_price + (sl_dist * config.RISK_REWARD_RATIO)
                
                # Binance Futures requires specific params for TP/SL orders
                # Stop Loss
                self.exchange.create_order(self.symbol, 'STOP_MARKET', 'sell', amount, None, {
                    'stopPrice': sl_price,
                    'reduceOnly': True
                })
                # Take Profit
                self.exchange.create_order(self.symbol, 'TAKE_PROFIT_MARKET', 'sell', amount, None, {
                    'stopPrice': tp_price,
                    'reduceOnly': True
                })
                print(f"🛡️ SL: {sl_price:.2f} | 🎯 TP: {tp_price:.2f}")

            elif signal == 'short':
                # Market Sell
                order = self.exchange.create_market_sell_order(self.symbol, amount)
                print(f"✅ SELL Order Filled: {order['id']}")
                
                # Set TP/SL
                entry_price = float(order['average']) if order['average'] else close_price
                sl_price = entry_price + sl_dist
                tp_price = entry_price - (sl_dist * config.RISK_REWARD_RATIO)
                
                # Stop Loss
                self.exchange.create_order(self.symbol, 'STOP_MARKET', 'buy', amount, None, {
                    'stopPrice': sl_price,
                    'reduceOnly': True
                })
                # Take Profit
                self.exchange.create_order(self.symbol, 'TAKE_PROFIT_MARKET', 'buy', amount, None, {
                    'stopPrice': tp_price,
                    'reduceOnly': True
                })
                print(f"🛡️ SL: {sl_price:.2f} | 🎯 TP: {tp_price:.2f}")
                
        except Exception as e:
            print(f"❌ Trade Execution Error: {e}")

    def run(self):
        print(f"🤖 Binance Bot Started | {self.symbol} | {self.timeframe}")
        print("Waiting for next candle close...")
        
        while True:
            try:
                # 1. Fetch Data
                df = self.fetch_data()
                if df is None:
                    time.sleep(10)
                    continue
                
                # 2. Calculate Signals
                df = self.calculate_signals(df)
                last_row = df.iloc[-2] # Use completed candle (index -2, as -1 is current forming candle)
                current_price = last_row['close']
                
                # Log status
                print(f"[{datetime.now()}] Price: {current_price} | Delta Z: {last_row['delta_z']:.2f} | SMA50: {last_row['sma50']:.2f}")
                
                # 3. Check Position
                pos_side, pos_size, entry_price = self.get_position()
                
                if pos_size == 0:
                    # No position, check for entry
                    if last_row['delta_z'] > config.Z_SCORE_THRESHOLD and last_row['close'] > last_row['sma50']:
                        print("🟢 LONG Signal Detected!")
                        self.execute_trade('long', current_price, last_row['atr'])
                        
                    elif last_row['delta_z'] < -config.Z_SCORE_THRESHOLD and last_row['close'] < last_row['sma50']:
                        print("🔴 SHORT Signal Detected!")
                        self.execute_trade('short', current_price, last_row['atr'])
                else:
                    print(f"⚠️ Position Open: {pos_side.upper()} (Size: {pos_size}) - Waiting for TP/SL")

                # Sleep logic - Wait until next hour
                # Calculate time until next hour
                now = datetime.now()
                next_hour = now.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
                sleep_seconds = (next_hour - now).total_seconds() + 10  # Add 10 sec buffer
                
                print(f"⏳ Next check at {next_hour.strftime('%H:%M')} (sleeping {int(sleep_seconds)}s)")
                time.sleep(sleep_seconds)
                
            except KeyboardInterrupt:
                print("\n🛑 Bot Stopped by User")
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                time.sleep(10)

if __name__ == "__main__":
    bot = BinanceBot()
    bot.run()

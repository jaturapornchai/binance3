import ccxt
import pandas as pd
import numpy as np
import time
import config
from datetime import datetime
import sys
import builtins

# Force unbuffered output for Docker logs
sys.stdout.reconfigure(line_buffering=True)

# Override print to always flush
def print(*args, **kwargs):
    kwargs['flush'] = True
    builtins.print(*args, **kwargs)

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
        self.check_position_mode()
        self.set_leverage()

    def check_connection(self):
        try:
            self.exchange.fetch_time()
            print("✅ Connected to Binance Futures")
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            sys.exit(1)

    def check_position_mode(self):
        try:
            # Check if in Hedge Mode or One-Way Mode
            response = self.exchange.fapiPrivateGetPositionSideDual()
            if response['dualSidePosition']:
                print("⚠️ Account is in HEDGE Mode (Long/Short separate)")
            else:
                print("✅ Account is in ONE-WAY Mode")
        except Exception as e:
            print(f"⚠️ Could not check position mode: {e}")

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

    def execute_trade(self, signal, close_price, sl_price, tp_price):
        balance = self.exchange.fetch_balance()['USDT']['free']
        
        # Fixed Position Size: $500 Margin
        margin_amount = 500.0
        
        # Check if we have enough balance
        if balance < margin_amount:
            print(f"⚠️ Insufficient balance ({balance:.2f} USDT) for $500 margin trade. Using 90% of balance instead.")
            margin_amount = balance * 0.90
            
        # Position Value = Margin * Leverage
        position_value = margin_amount * self.leverage 
        amount = position_value / close_price
        
        print(f"🚀 Executing {signal.upper()} | Margin: ${margin_amount:.2f} | Lev: {self.leverage}x | Size: {amount:.4f} BTC")
        
        try:
            if signal == 'long':
                # Market Buy
                order = self.exchange.create_market_buy_order(self.symbol, amount)
                print(f"✅ BUY Order Filled: {order['id']}")
                
                # Set TP/SL
                # entry_price = float(order['average']) if order['average'] else close_price
                # Use calculated SL/TP directly
                
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
                # entry_price = float(order['average']) if order['average'] else close_price
                
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

    def get_account_status(self):
        try:
            balance = self.exchange.fetch_balance()
            usdt_free = balance['USDT']['free']
            usdt_total = balance['USDT']['total']
            
            # Fetch all positions to ensure we don't miss it due to symbol mismatch in filter
            # CCXT might return 'BTC/USDT:USDT' for 'BTC/USDT'
            positions = self.exchange.fetch_positions()
            open_orders = self.exchange.fetch_open_orders(self.symbol)
            
            active_positions = []
            for pos in positions:
                # Check for symbol match (handling BTC/USDT vs BTC/USDT:USDT)
                symbol_match = (pos['symbol'] == self.symbol) or (pos['symbol'] == f"{self.symbol}:USDT")
                
                if symbol_match and float(pos['contracts']) > 0:
                    active_positions.append(pos)
            
            return {
                'balance': usdt_free,
                'total': usdt_total,
                'positions': active_positions,
                'orders': open_orders
            }
        except Exception as e:
            print(f"❌ Error fetching account status: {e}")
            return None

    def cancel_all_orders(self):
        try:
            self.exchange.cancel_all_orders(self.symbol)
            print("🗑️ All open orders cancelled.")
        except Exception as e:
            print(f"⚠️ Error cancelling orders: {e}")

    def close_position(self, position):
        try:
            symbol = position['symbol']
            amount = float(position['contracts'])
            side = position['side']
            
            print(f"🔄 Closing {side.upper()} Position for Reversal...")
            
            if side == 'long':
                self.exchange.create_market_sell_order(symbol, amount)
            else:
                self.exchange.create_market_buy_order(symbol, amount)
                
            print("✅ Position Closed.")
            time.sleep(2)
        except Exception as e:
            print(f"❌ Error closing position: {e}")

    def run(self):
        print(f"🤖 Binance Bot Started | {self.symbol} | {self.timeframe}")
        
        last_check_hour = -1

        while True:
            try:
                now = datetime.now()
                print(f"\n--- Status Check [{now.strftime('%Y-%m-%d %H:%M:%S')}] ---")
                
                # 1. Get Status
                status = self.get_account_status()
                active_positions = status['positions']
                has_position = len(active_positions) > 0
                
                if status:
                    print(f"💰 Balance: {status['balance']:.2f} USDT (Total: {status['total']:.2f})")
                    
                    if has_position:
                        for pos in active_positions:
                            side = pos['side']
                            size = float(pos['contracts'])
                            entry = float(pos['entryPrice'])
                            pnl = float(pos['unrealizedPnl'])
                            print(f"⚠️ Position: {side.upper()} | Size: {size} | Entry: {entry} | PnL: {pnl:.2f}")
                        print("ℹ️ Position is OPEN. New signals will be ignored.")
                    else:
                        print("✅ No Open Position. Ready for signals (Long/Short).")
                        
                    orders = status['orders']
                    print(f"📋 Open Orders: {len(orders)}")
                    for o in orders:
                        # For Stop/TP orders, 'price' might be None, so we check 'stopPrice'
                        price = o.get('price')
                        stop_price = o.get('stopPrice') or o.get('triggerPrice')
                        
                        if price is None or float(price) == 0:
                            display_price = f"{stop_price} (Trigger)"
                        else:
                            display_price = price
                            
                        print(f"   - {o['type']} {o['side']} @ {display_price} ({o['status']})")
                        
                    # 2. Cleanup Logic: No Position but have Orders -> Cancel All
                    if not has_position and len(orders) > 0:
                        print("🧹 No position but orders found. Cleaning up...")
                        self.cancel_all_orders()

                # 3. Entry Logic (Only at minute 0)
                if now.minute == 0 and now.hour != last_check_hour:
                    last_check_hour = now.hour
                    print("🕐 Hourly Check Triggered...")
                    
                    # Fetch Data
                    df = self.fetch_data()
                    if df is not None:
                        # Calculate Signals
                        df = self.calculate_signals(df)
                        last_row = df.iloc[-2]
                        current_price = last_row['close']
                        
                        print(f"📊 Market Data: Price: {current_price} | Delta Z: {last_row['delta_z']:.2f} | SMA50: {last_row['sma50']:.2f}")
                        
                        # Calculate Swing High/Low (Lookback 50)
                        # df has current candle at -1 (forming), -2 (last closed)
                        # We want last 50 closed candles: -52 to -2
                        lookback_window = df.iloc[-52:-2]
                        swing_low = lookback_window['low'].min()
                        swing_high = lookback_window['high'].max()
                        
                        # Determine Signals
                        long_signal = last_row['delta_z'] > config.Z_SCORE_THRESHOLD and last_row['close'] > last_row['sma50']
                        short_signal = last_row['delta_z'] < -config.Z_SCORE_THRESHOLD and last_row['close'] < last_row['sma50']
                        
                        # Check for Reversal
                        if has_position:
                            for pos in active_positions:
                                side = pos['side']
                                if side == 'long' and short_signal:
                                    print("🔄 Reversal Signal Detected (Long -> Short)!")
                                    self.cancel_all_orders()
                                    self.close_position(pos)
                                    has_position = False
                                    
                                elif side == 'short' and long_signal:
                                    print("🔄 Reversal Signal Detected (Short -> Long)!")
                                    self.cancel_all_orders()
                                    self.close_position(pos)
                                    has_position = False

                        if not has_position:
                            if long_signal:
                                print("🟢 LONG Signal Detected!")
                                
                                # SL = Swing Low
                                sl_price = swing_low
                                # Safety: If SL is above current price (impossible for min low) or too close
                                if sl_price >= current_price: sl_price = current_price - last_row['atr']
                                
                                # TP = Entry + (Risk * 1.618)
                                risk = current_price - sl_price
                                tp_price = current_price + (risk * 1.618)
                                
                                self.execute_trade('long', current_price, sl_price, tp_price)
                                
                            elif short_signal:
                                print("🔴 SHORT Signal Detected!")
                                
                                # SL = Swing High
                                sl_price = swing_high
                                # Safety
                                if sl_price <= current_price: sl_price = current_price + last_row['atr']
                                
                                # TP = Entry - (Risk * 1.618)
                                risk = sl_price - current_price
                                tp_price = current_price - (risk * 1.618)
                                
                                self.execute_trade('short', current_price, sl_price, tp_price)
                        else:
                            print("ℹ️ Position remains open. No reversal signal.")
                
                # Sleep logic - Align to next minute
                sleep_seconds = 60 - datetime.now().second
                print(f"💤 Sleeping {sleep_seconds}s...")
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

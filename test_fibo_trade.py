import config
from binance_bot import BinanceBot
import pandas as pd
import time
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

print("Initializing Bot for Testing...")
bot = BinanceBot()

print("\n--- 1. Closing Existing Positions & Orders ---")
# Cancel all orders
bot.cancel_all_orders()

# Close positions
status = bot.get_account_status()
pos = status['position']
if pos:
    print(f"Found open position: {pos['symbol']} {pos['side']} {pos['contracts']}")
    # Close it
    side = pos['side']
    amount = float(pos['contracts'])
    if side == 'long':
        print("Closing LONG position...")
        bot.exchange.create_market_sell_order(bot.symbol, amount)
    else:
        print("Closing SHORT position...")
        bot.exchange.create_market_buy_order(bot.symbol, amount)
    print("Position closed.")
    time.sleep(2) # Wait for fill
else:
    print("No open position found.")

print("\n--- 2. Fetching Data & Calculating Fibo Levels ---")
df = bot.fetch_data()
if df is None:
    print("Error fetching data")
    sys.exit(1)

# Calculate signals to get ATR
df = bot.calculate_signals(df)

# Calculate Swing High/Low (Lookback 50)
lookback_window = df.iloc[-52:-2]
swing_low = lookback_window['low'].min()
swing_high = lookback_window['high'].max()
current_price = df.iloc[-1]['close']
atr = df.iloc[-2]['atr']

print(f"Current Price: {current_price}")
print(f"Swing Low (50): {swing_low}")
print(f"Swing High (50): {swing_high}")
print(f"ATR: {atr}")

# Test LONG Entry
print("\n--- 3. Executing TEST LONG Trade with Fibo TP/SL ---")

# SL = Swing Low
sl_price = swing_low
# Safety: If SL is above current price (impossible for min low) or too close
if sl_price >= current_price: 
    print("Adjusting SL (too close/high)")
    sl_price = current_price - atr

# TP = Entry + (Risk * 1.618)
risk = current_price - sl_price
tp_price = current_price + (risk * 1.618)

print(f"Plan -> Entry: {current_price} | SL: {sl_price:.2f} | TP: {tp_price:.2f}")

# Execute
bot.execute_trade('long', current_price, sl_price, tp_price)

print("\n--- Test Complete ---")
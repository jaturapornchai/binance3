import config
from binance_bot import BinanceBot
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

print("Connecting to Binance...")
bot = BinanceBot()

print("\n--- Wallet Status ---")
balance = bot.exchange.fetch_balance()
usdt = balance['USDT']
print(f"USDT Free:  {usdt['free']:.2f}")
print(f"USDT Used:  {usdt['used']:.2f}")
print(f"USDT Total: {usdt['total']:.2f}")

print("\n--- Checking ALL Positions ---")
try:
    # Fetch all positions without filtering by symbol first
    positions = bot.exchange.fetch_positions()
    found_any = False
    
    for pos in positions:
        size = float(pos['contracts'])
        if size > 0:
            found_any = True
            symbol = pos['symbol']
            side = pos['side']
            entry = pos['entryPrice']
            pnl = pos['unrealizedPnl']
            leverage = pos['leverage']
            print(f"⚠️  FOUND: {symbol} | {side.upper()} | Size: {size} | Entry: {entry} | PnL: {pnl} | Lev: {leverage}x")

    if not found_any:
        print("✅ No open positions found in ANY symbol.")

except Exception as e:
    print(f"❌ Error fetching positions: {e}")

print("\n--- Checking Open Orders (Current Symbol) ---")
orders = bot.exchange.fetch_open_orders(config.SYMBOL)
print(f"Open Orders for {config.SYMBOL}: {len(orders)}")
for o in orders:
    print(f"   - {o['type']} {o['side']} @ {o['price']}")

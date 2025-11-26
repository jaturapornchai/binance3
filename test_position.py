import config
from binance_bot import BinanceBot

# Initialize bot
bot = BinanceBot()

# Test long position with real money
print("Testing LONG position with SL and TP using REAL MONEY...")
bot.execute_trade('long', 86308.54, 500)

# Check position
pos_side, pos_size, entry_price = bot.get_position()
print(f"Position after trade: {pos_side}, Size: {pos_size}, Entry: {entry_price}")
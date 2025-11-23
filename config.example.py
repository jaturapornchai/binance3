import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# API Credentials (Get these from Binance)
# WARNING: Never commit your real API keys to version control!
API_KEY = os.getenv('BINANCE_API_KEY', 'YOUR_API_KEY_HERE')
API_SECRET = os.getenv('BINANCE_API_SECRET', 'YOUR_API_SECRET_HERE')

# Trading Settings
SYMBOL = 'BTC/USDT'  # CCXT format
TIMEFRAME = '1h'
LEVERAGE = 5
RISK_PER_TRADE = 0.10  # 10% of account balance per trade (Adjust as needed)
RISK_REWARD_RATIO = 3.0
Z_SCORE_THRESHOLD = 1.5

# Bot Settings
CHECK_INTERVAL = 60  # Check every 60 seconds
IS_TESTNET = False  # Set to True for paper trading on Binance Testnet

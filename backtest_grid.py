import pandas as pd
import numpy as np
import glob

class GridBacktester:
    def __init__(self, data_path, fee_rate=0.0004, initial_capital=10000):
        self.df = pd.read_csv(data_path)
        self.df['Open Time'] = pd.to_datetime(self.df['Open Time'])
        self.df.set_index('Open Time', inplace=True)
        self.fee_rate = fee_rate
        self.initial_capital = initial_capital
        
    def run_grid(self, lower_price, upper_price, grid_count):
        # Setup Grids
        grids = np.linspace(lower_price, upper_price, grid_count)
        grid_step = grids[1] - grids[0]
        
        # State
        cash = self.initial_capital
        holdings = 0.0 # Amount of BTC
        
        # Initial Allocation: Buy 50% to start neutral? 
        # Or start with 100% cash and only buy as price drops?
        # Standard Grid Bot starts by buying enough to cover sell orders above.
        # Let's assume we start with 100% Cash and place Buy Orders below current price.
        # Wait, for a Neutral Grid, we need inventory to sell.
        # Let's assume we start at the middle of the range and buy 50% inventory.
        
        start_price = self.df['Close'].iloc[0]
        
        # Find current grid index
        current_grid_idx = (np.abs(grids - start_price)).argmin()
        
        # Buy inventory for all grids ABOVE current price (to be able to sell them)
        # Actually, standard logic:
        # - Place Sell Orders for all grids > current_price
        # - Place Buy Orders for all grids < current_price
        # To place Sell Orders, we need BTC.
        # So we buy BTC equal to (grids_above * amount_per_grid)
        
        grids_above = len(grids) - 1 - current_grid_idx
        amount_per_grid = (self.initial_capital / len(grids)) / start_price # Rough estimate
        
        initial_buy_cost = grids_above * amount_per_grid * start_price
        if initial_buy_cost > cash:
            amount_per_grid = cash / (len(grids) * start_price) # Adjust size
            initial_buy_cost = grids_above * amount_per_grid * start_price
            
        cash -= initial_buy_cost
        cash -= (initial_buy_cost * self.fee_rate) # Fee
        holdings += (grids_above * amount_per_grid)
        
        # Active Orders
        # 0 = No Order, 1 = Buy Order, -1 = Sell Order
        # We track the state of each grid line
        grid_states = np.zeros(grid_count) 
        
        # Initial Orders
        for i in range(grid_count):
            if i < current_grid_idx:
                grid_states[i] = 1 # Buy Order waiting
            elif i > current_grid_idx:
                grid_states[i] = -1 # Sell Order waiting
        
        trades = 0
        total_profit = 0.0
        
        highs = self.df['High'].values
        lows = self.df['Low'].values
        
        equity_curve = []
        
        for i in range(len(self.df)):
            current_high = highs[i]
            current_low = lows[i]
            
            # Check for fills
            # We iterate through grids to see if any were crossed
            
            # Optimization: Only check grids near current price?
            # For simulation accuracy, we check all active orders.
            
            for g in range(grid_count):
                grid_price = grids[g]
                
                if grid_states[g] == 1: # Buy Order
                    if current_low <= grid_price:
                        # Buy Filled
                        cost = amount_per_grid * grid_price
                        if cash >= cost:
                            cash -= cost
                            cash -= (cost * self.fee_rate)
                            holdings += amount_per_grid
                            grid_states[g] = -1 # Switch to Sell
                            trades += 1
                        
                elif grid_states[g] == -1: # Sell Order
                    if current_high >= grid_price:
                        # Sell Filled
                        revenue = amount_per_grid * grid_price
                        if holdings >= amount_per_grid:
                            cash += revenue
                            cash -= (revenue * self.fee_rate)
                            holdings -= amount_per_grid
                            grid_states[g] = 1 # Switch to Buy
                            trades += 1
                            
                            # Profit for this grid pair approx = grid_step * amount
                            total_profit += (grid_step * amount_per_grid)

            # Calculate Equity
            current_price = self.df['Close'].iloc[i]
            equity = cash + (holdings * current_price)
            equity_curve.append(equity)
            
        final_equity = equity_curve[-1]
        return_pct = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        
        return {
            'Grids': grid_count,
            'Range': f"{lower_price}-{upper_price}",
            'Return (%)': round(return_pct, 2),
            'Final Equity': round(final_equity, 2),
            'Trades': trades,
            'Holdings': holdings
        }

if __name__ == "__main__":
    files = glob.glob("binance_btcusdt_15m_365days.csv")
    if not files:
        print("No data file found!")
    else:
        print("Simulating Grid Trading (15m)...")
        bt = GridBacktester(files[0])
        
        # Determine Range from Data
        df = pd.read_csv(files[0])
        min_price = df['Low'].min() * 0.95
        max_price = df['High'].max() * 1.05
        print(f"Price Range: {min_price:.2f} - {max_price:.2f}")
        
        grid_counts = [50, 100, 150]
        
        results = []
        for gc in grid_counts:
            print(f"Testing Grid Count: {gc}...")
            res = bt.run_grid(min_price, max_price, gc)
            results.append(res)
            
        res_df = pd.DataFrame(results)
        print("\n=== Grid Trading Results ===")
        print(res_df.to_string(index=False))

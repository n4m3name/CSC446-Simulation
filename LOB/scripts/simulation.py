"""
Main simulation loop for the limit order book.
Includes:
- Discrete Event Simulation engine
- Market Maker logic
- Statistical Aggregation (Mean + 95% CI)
- Automated Parameter Sweeps
- Command-line argument parsing to select experiments
"""

import numpy as np
import pandas as pd
import math
import argparse

# We assume these are in the same directory. 
# If running as a standalone script without these files, these imports would need those files present.
from order_book import OrderBook
from market_maker import MarketMaker

class Simulation:
    def __init__(
        self,
        sim_duration=600.0,      # change this to 23400 for the correct seconds in a trading day
        initial_price=100.0,
        # Poisson event rates (events per unit time)
        lam_limit_buy=0.5,
        lam_limit_sell=0.5,
        lam_mkt_buy=0.2,
        lam_mkt_sell=0.2,
        lam_cancel=0.1,
        # Market maker parameters
        mm_base_spread=1.0,
        mm_skew_coef=0.05,
        seed=123,
    ):
        self.sim_duration = sim_duration
        self.initial_price = initial_price

        # Event rates
        self.lam_limit_buy = lam_limit_buy
        self.lam_limit_sell = lam_limit_sell
        self.lam_mkt_buy = lam_mkt_buy
        self.lam_mkt_sell = lam_mkt_sell
        self.lam_cancel = lam_cancel

        # Random number generator
        self.rng = np.random.default_rng(seed)

        # Core objects
        self.book = OrderBook()
        self.mm = MarketMaker(
            book=self.book,
            mm_id="MM",
            base_spread=mm_base_spread,
            skew_coef=mm_skew_coef,
        )

        # Simulation clock
        self.time = 0.0

        # Metrics
        self.spread_times = []
        self.spreads = []

        # --- EXECUTION METRICS ---
        self.exec_times = []        # Wait times for Resting Limit Orders (duration > 0)
        self.mkt_exec_count = 0     # Count of Immediate Market Orders (duration = 0)

        self.mm_times = []
        self.mm_pnls = []
        self.mm_inventory = []

    # -------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------

    def _reference_price(self):
        """Reference price for sampling new limit order prices."""
        return self.book.last_trade_price or self.initial_price

    def _sample_limit_price(self):
        """
        Simple toy rule for limit prices:
        - Around reference price within a few ticks.
        """
        ref = self._reference_price()
        tick = 0.5
        offset_ticks = self.rng.integers(-5, 6)  # from -5 to +5
        price = ref + offset_ticks * tick
        return max(tick, price)

    def _record_spread(self):
        """Record current bid–ask spread (if both sides exist)."""
        best_bid = self.book.best_bid()
        best_ask = self.book.best_ask()
        if best_bid is not None and best_ask is not None:
            self.spread_times.append(self.time)
            self.spreads.append(best_ask.price - best_bid.price)

    def _record_mm_state(self):
        """Record MM inventory and PnL at current time."""
        self.mm_times.append(self.time)
        self.mm_pnls.append(self.mm.pnl())
        self.mm_inventory.append(self.mm.inventory)

    def _cancel_best_random_side(self):
        """
        Very simple cancellation model:
        - Randomly choose bid or ask (if they exist)
        - Cancel the best order on that side
        """
        sides = []
        if self.book.best_bid():
            sides.append("buy")
        if self.book.best_ask():
            sides.append("sell")

        if not sides:
            return

        side = self.rng.choice(sides)
        self.book.cancel_best(side)

    # -------------------------------------------------------
    # Trade handling
    # -------------------------------------------------------

    def _handle_trade(self, trade):
        """
        Update MM PnL and collect execution-time stats.
        """
        price = trade["price"]
        qty = trade["quantity"]
        t = trade["time"]

        # Execution times for "noise" orders that got filled
        for role in ("buy_order", "sell_order"):
            order = trade[role]
            
            # 1. Check if the order exists and is a "noise" trader
            if order is not None and order.trader_id == "noise":
                duration = t - order.time_created
                
                # 2. SEPARATE LOGIC:
                # If duration is effectively 0, it was a Market Order (Immediate).
                # If duration > 0, it was a Limit Order (Resting).
                if duration > 1e-10:
                    self.exec_times.append(duration)
                else:
                    self.mkt_exec_count += 1

        # MM inventory and cash: update when MM is involved
        for role, side in (("buy_order", "buy"), ("sell_order", "sell")):
            order = trade[role]
            if order is not None and order.trader_id == self.mm.mm_id:
                self.mm.on_trade(trade_price=price, side=side, quantity=qty)

    # -------------------------------------------------------
    # Main simulation loop
    # -------------------------------------------------------

    def run(self):
        """
        Run the discrete-event simulation until sim_duration.
        Returns a dict of summary stats.
        """
        event_names = [
            "limit_buy",
            "limit_sell",
            "mkt_buy",
            "mkt_sell",
            "cancel",
        ]
        rates = np.array([
            self.lam_limit_buy,
            self.lam_limit_sell,
            self.lam_mkt_buy,
            self.lam_mkt_sell,
            self.lam_cancel,
        ], dtype=float)

        total_rate = rates.sum()
        if total_rate <= 0:
            raise ValueError("Total event rate must be positive.")

        probs = rates / total_rate

        while self.time < self.sim_duration:
            # 1. Advance time by exponential(1 / total_rate)
            dt = self.rng.exponential(1.0 / total_rate)
            self.time += dt

            # 2. Pick event type
            event = self.rng.choice(event_names, p=probs)

            # 3. Handle each event type
            if event == "limit_buy":
                price = self._sample_limit_price()
                self.book.submit_limit(
                    trader_id="noise",
                    side="buy",
                    price=price,
                    quantity=1,
                    time_created=self.time,
                )

            elif event == "limit_sell":
                price = self._sample_limit_price()
                self.book.submit_limit(
                    trader_id="noise",
                    side="sell",
                    price=price,
                    quantity=1,
                    time_created=self.time,
                )

            elif event == "mkt_buy":
                # Execute 1-unit market buy against best ask
                trades = self.book.submit_market(
                    trader_id="noise",
                    side="buy",
                    quantity=1,
                    time_created=self.time,
                )
                for tr in trades:
                    self._handle_trade(tr)

            elif event == "mkt_sell":
                trades = self.book.submit_market(
                    trader_id="noise",
                    side="sell",
                    quantity=1,
                    time_created=self.time,
                )
                for tr in trades:
                    self._handle_trade(tr)

            elif event == "cancel":
                self._cancel_best_random_side()

            # 4. Market maker quotes after each event
            self.mm.quote(self.time)

            # 5. Match any crossed limit orders (best_bid >= best_ask)
            while True:
                trade = self.book.try_match_limit_crossing(self.time)
                if trade is None:
                    break
                self._handle_trade(trade)

            # 6. Record metrics
            self._record_spread()
            self._record_mm_state()

        # 7. Summaries
        avg_spread = float(np.mean(self.spreads)) if self.spreads else 0.0
        
        # Median of RESTING orders (Limit orders)
        median_exec = float(np.median(self.exec_times)) if self.exec_times else 0.0

        # Calculate Total Executions (Resting Limit + Instant Market)
        total_executions = len(self.exec_times) + self.mkt_exec_count

        # Return dict
        return {
            "avg_spread": avg_spread,
            "median_exec_time": median_exec,
            "total_executions": total_executions,
            "mm_final_pnl": self.mm.pnl(),
            "mm_final_inventory": self.mm.inventory,
        }

# ==============================================================================
# STATISTICAL HELPER FUNCTIONS
# ==============================================================================

def calculate_stats(data_list, confidence=0.95):
    """
    Calculates Mean and 95% Confidence Interval for a list of numbers.
    Uses t-distribution.
    
    Returns: (mean, ci_width)
    Output Mean formatted as: mean
    Output CI formatted as: mean +/- ci_width
    """
    n = len(data_list)
    if n < 2:
        return np.mean(data_list), 0.0
    
    mean = np.mean(data_list)
    std_err = np.std(data_list, ddof=1) / math.sqrt(n)
    
    # t-critical value for 95% CI (two-tailed) with DOF = n-1
    # For N=5, dof=4, t_0.025 approx 2.776
    t_crit_map = {
        5: 2.776,
        10: 2.262,
        30: 2.045
    }
    t_crit = t_crit_map.get(n, 1.96) # Default to Z-score if N large/unknown
    
    ci_width = t_crit * std_err
    return mean, ci_width

# ==============================================================================
# MAIN EXECUTION & PARAMETER SWEEPS
# ==============================================================================

if __name__ == "__main__":
    
    # --- COMMAND LINE ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="LOB Simulation Experiment Runner")
    parser.add_argument(
        "--exp1", action="store_true", help="Run Experiment 1: Liquidity Supply (Sweeps lam_limit)"
    )
    parser.add_argument(
        "--exp2", action="store_true", help="Run Experiment 2: Market Stability (Sweeps lam_cancel)"
    )
    parser.add_argument(
        "--exp3", action="store_true", help="Run Experiment 3: MM Strategy (Sweeps mm_skew_coef)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all experiments (Exp 1, 2, 3). This is the default if no flags are specified."
    )
    parser.add_argument(
        "--replications", 
        type=int, 
        default=5, 
        help="Number of replications to run per scenario (default: 5)"
    )
    args = parser.parse_args()

    # --- GLOBAL SETTINGS ---
    SIM_DURATION = 23400.0  # 6.5 hours
    NUM_REPLICATIONS = args.replications
    
    # Container for all final aggregated data
    all_aggregated_results = []

    # -----------------------------------------------------------
    # DEFINE EXPERIMENTS
    # -----------------------------------------------------------
    
    experiments = {
        "Exp1_Liquidity_Supply": {
            "flag": "exp1", # The associated command line flag
            "variable_name": "lam_limit", # Affects both buy and sell
            "values": [0.2, 0.5, 1.0, 2.0, 5.0],
            "defaults": {
                "lam_mkt_buy": 0.2, "lam_mkt_sell": 0.2, 
                "lam_cancel": 0.1, "mm_skew_coef": 0.05,
                "mm_base_spread": 1.0
            }
        },
        "Exp2_Market_Stability": {
            "flag": "exp2", # The associated command line flag
            "variable_name": "lam_cancel",
            "values": [0.1, 0.5, 1.0, 5.0, 10.0],
            "defaults": {
                "lam_limit_buy": 1.0, "lam_limit_sell": 1.0,
                "lam_mkt_buy": 0.5, "lam_mkt_sell": 0.5,
                "mm_skew_coef": 0.05,
                "mm_base_spread": 1.0
            }
        },
        "Exp3_MM_Strategy": {
            "flag": "exp3", # The associated command line flag
            "variable_name": "mm_skew_coef",
            "values": [0.0, 0.01, 0.05, 0.1, 0.5],
            "defaults": {
                "lam_limit_buy": 0.6, "lam_limit_sell": 0.6,
                "lam_mkt_buy": 0.2, "lam_mkt_sell": 0.2,
                "lam_cancel": 0.1,
                "mm_base_spread": 1.0
            }
        }
    }

    # Determine which experiments to run
    experiments_to_run = []
    
    # If no specific experiment flags or --all is provided, run all
    if args.all or (not args.exp1 and not args.exp2 and not args.exp3):
        experiments_to_run = experiments.values()
    else:
        # Otherwise, run only the selected ones
        if args.exp1:
            experiments_to_run.append(experiments["Exp1_Liquidity_Supply"])
        if args.exp2:
            experiments_to_run.append(experiments["Exp2_Market_Stability"])
        if args.exp3:
            experiments_to_run.append(experiments["Exp3_MM_Strategy"])

    if not experiments_to_run:
        print("No experiments selected. Use --all or specific experiment flags (--exp1, --exp2, --exp3).")
        exit()

    print(f"Starting Simulation Suite.")
    print(f"Duration: {SIM_DURATION}s | Replications per scenario: {NUM_REPLICATIONS}")
    print("-" * 60)

    # -----------------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------------

    for exp_config in experiments_to_run:
        exp_name = next(k for k, v in experiments.items() if v == exp_config)
        var_name = exp_config["variable_name"]
        sweep_values = exp_config["values"]
        defaults = exp_config["defaults"]
        
        print(f"\nRunning {exp_name} (Sweeping {var_name})...")

        for val in sweep_values:
            
            # --- 1. PREPARE PARAMETERS ---
            # Start with defaults
            current_params = defaults.copy()
            current_params["sim_duration"] = SIM_DURATION
            
            # Apply the independent variable value
            if var_name == "lam_limit":
                current_params["lam_limit_buy"] = val
                current_params["lam_limit_sell"] = val
            else:
                current_params[var_name] = val
            
            # --- 2. REPLICATION LOOP ---
            replication_results = []
            
            for i in range(NUM_REPLICATIONS):
                # Unique seed for every run: 1000 + (val_index * 100) + replication_index
                # Ensure reproducibility but independence
                seed = 1000 + int(val * 100) + i
                
                sim = Simulation(seed=seed, **current_params)
                res = sim.run()
                replication_results.append(res)
            
            # --- 3. STATISTICAL AGGREGATION ---
            # Extract lists for metrics
            spreads = [r["avg_spread"] for r in replication_results]
            exec_times = [r["median_exec_time"] for r in replication_results]
            pnls = [r["mm_final_pnl"] for r in replication_results]
            invs = [abs(r["mm_final_inventory"]) for r in replication_results] # Abs inventory for risk
            exec_counts = [r["total_executions"] for r in replication_results]

            # Compute Mean and CI
            mean_spread, ci_spread = calculate_stats(spreads)
            mean_exec, ci_exec = calculate_stats(exec_times)
            mean_pnl, ci_pnl = calculate_stats(pnls)
            mean_inv, ci_inv = calculate_stats(invs)
            mean_vol, ci_vol = calculate_stats(exec_counts)

            # Log to console
            print(f"  {var_name}={val}: Spread={mean_spread:.3f}±{ci_spread:.3f}, PnL={mean_pnl:.2f}±{ci_pnl:.2f}")

            # --- 4. STORE RESULTS ---
            row = {
                "Experiment": exp_name,
                "Independent_Var": var_name,
                "Value": val,
                # Spreads
                "Avg_Spread_Mean": mean_spread,
                "Avg_Spread_CI": ci_spread,
                # Execution Times
                "Median_Exec_Time_Mean": mean_exec,
                "Median_Exec_Time_CI": ci_exec,
                # PnL
                "MM_PnL_Mean": mean_pnl,
                "MM_PnL_CI": ci_pnl,
                # Inventory (Absolute)
                "MM_Abs_Inventory_Mean": mean_inv,
                "MM_Abs_Inventory_CI": ci_inv,
                # Volume
                "Volume_Mean": mean_vol,
                "Volume_CI": ci_vol
            }
            all_aggregated_results.append(row)

    # -----------------------------------------------------------
    # EXPORT TO CSV
    # -----------------------------------------------------------
    
    df = pd.DataFrame(all_aggregated_results)
    filename = "data/simulation_results.csv"
    df.to_csv(filename, index=False)
    print("\n" + "="*60)
    print(f"Simulation Suite Complete. Results saved to {filename}")
    print("="*60)
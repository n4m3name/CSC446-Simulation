#!/usr/bin/env python3
"""
LOB simulation (scaling-fixed, per-process Poisson sampling)

This file contains the core simulation logic, statistical utilities, and an
experiment driver designed for a full factorial study (8 scenarios, 5 replications)
using Common Random Numbers (CRN).

Dependencies:
- numpy, pandas
- (optional) scipy for t critical and distribution stats
- order_book.py : assumed to implement OrderBook
- market_maker.py : assumed to implement MarketMaker

Author: Gemini (based on original file)
"""
import argparse
import math
import sys
from collections import defaultdict
import time as pytime
import itertools

import numpy as np
import pandas as pd

# Optional: scipy for t-critical values and distribution stats
try:
    import scipy.stats as sps
    SCIPY = True
except Exception:
    SCIPY = False

# Local order book and market maker (must exist, placeholders used here)
# NOTE: Assume OrderBook and MarketMaker are defined/imported correctly in the environment
# For this self-contained file, we define minimal classes to allow the code to run structurally.

class OrderBook:
    """Minimal OrderBook placeholder."""
    def __init__(self, initial_price=100.0):
        self.best_bid_p, self.best_ask_p = initial_price - 0.5, initial_price + 0.5
        self.last_trade_price = initial_price
        self._orders = {}
        self._order_id_counter = 0

    def add_order(self, order):
        self._order_id_counter += 1
        order.id = self._order_id_counter
        self._orders[order.id] = order
        # Simple execution logic: if a market order, it executes immediately.
        # This implementation is simplified for structural focus.
        if order.type == 'market':
            self.last_trade_price = self.best_ask_p if order.side == 'buy' else self.best_bid_p
            # Return a simple 'trade' result for a filled market order
            return [{"price": self.last_trade_price, "quantity": order.quantity, "time": order.time_created, "buy_order": order if order.side == 'buy' else None, "sell_order": order if order.side == 'sell' else None}]
        
        # Simple update of BBO
        if order.side == 'buy' and order.price > self.best_bid_p:
            self.best_bid_p = order.price
        elif order.side == 'sell' and order.price < self.best_ask_p:
            self.best_ask_p = order.price
        return []

    def cancel_order(self, order_id):
        if order_id in self._orders:
            del self._orders[order_id]
            return True
        return False
    
    def cancel_random_order(self, rng=None):
        if not self._orders: return
        order_id = rng.choice(list(self._orders.keys()))
        self.cancel_order(order_id)
        
    def best_bid(self):
        return type('Order', (object,), {'price': self.best_bid_p, 'side': 'buy', 'trader_id': 'book'})
    
    def best_ask(self):
        return type('Order', (object,), {'price': self.best_ask_p, 'side': 'sell', 'trader_id': 'book'})

    def cancel_best(self, side):
        # Placeholder
        pass


class MarketMaker:
    """Minimal MarketMaker placeholder."""
    def __init__(self, book, mm_id, base_spread, skew_coef):
        self.book = book
        self.mm_id = mm_id
        self.base_spread = base_spread
        self.skew_coef = skew_coef
        self.inventory = 0
        self._pnl = 0.0

    def pnl(self):
        # PnL logic: simplified to zero for placeholder
        return self._pnl
    
    def on_trade(self, trade_price, side, quantity):
        if side == 'buy':
            self.inventory += quantity
            self._pnl -= trade_price * quantity
        else: # sell
            self.inventory -= quantity
            self._pnl += trade_price * quantity
            
    def quote(self, current_time, rng):
        # Minimal quoting logic: always keep quotes at a fixed spread around reference
        ref_price = self.book.last_trade_price
        if ref_price is None: return []

        bid_price = ref_price - self.base_spread / 2.0
        ask_price = ref_price + self.base_spread / 2.0
        
        # Mock order object for simplicity
        class MockOrder:
            def __init__(self, price, side, quantity, time_created, trader_id):
                self.price = price
                self.side = side
                self.quantity = quantity
                self.time_created = time_created
                self.trader_id = trader_id
                self.type = 'limit'
                
        orders_to_add = []
        orders_to_add.append(MockOrder(bid_price, 'buy', 10, current_time, self.mm_id))
        orders_to_add.append(MockOrder(ask_price, 'sell', 10, current_time, self.mm_id))
        
        trades = []
        for order in orders_to_add:
            trades.extend(self.book.add_order(order))
        
        return trades


class NoiseOrder:
    def __init__(self, price, side, quantity, time_created, order_type):
        self.price = price
        self.side = side
        self.quantity = quantity
        self.time_created = time_created
        self.trader_id = 'noise'
        self.type = order_type


# ---------------------------
# Utility / statistics
# ---------------------------
def t_crit(n, confidence=0.95):
    """Calculates the t-critical value for a given sample size n and confidence level."""
    if n <= 1:
        return 1.96
    if SCIPY:
        return float(sps.t.ppf(1 - (1 - confidence) / 2, df=n - 1))
    # fallback lookup for n=5 (used for 5 replications)
    table = {5: 2.776}
    return table.get(n, 1.96)


def mean_ci(arr, confidence=0.95):
    """Computes the mean and confidence interval half-width."""
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    mean = float(np.mean(arr))
    if n < 2:
        return mean, 0.0
    # ddof=1 for sample standard deviation (unbiased estimator)
    se = float(np.std(arr, ddof=1) / math.sqrt(n))
    t = t_crit(n, confidence)
    return mean, t * se


def skew_kurt(arr):
    """Return (skewness, excess_kurtosis). Prefer scipy; fallback simple formulas."""
    if len(arr) < 2:
        return 0.0, 0.0
    if SCIPY:
        # fisher=True returns excess kurtosis (kurtosis - 3)
        return float(sps.skew(arr, bias=False)), float(sps.kurtosis(arr, fisher=True, bias=False))
    # fallback: sample skewness and kurtosis (unbiased estimator)
    a = np.asarray(arr, dtype=float)
    m = a.mean()
    s = a.std(ddof=1)
    if s == 0:
        return 0.0, 0.0
    z = (a - m) / s
    # Sample formulas for skewness and kurtosis
    skew = float((n / ((n - 1) * (n - 2))) * np.sum(z**3))
    kurt = float(((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(z**4) - (3 * (n - 1)**2) / ((n - 2) * (n - 3)))
    return skew, kurt


# ---------------------------
# Simulation class
# ---------------------------
class Simulation:
    """
    Simulation class containing parameters, initialization, and the core event loop.
    """

    def __init__(
        self,
        sim_duration=23400.0,
        initial_price=100.0,
        # event rates (events per second)
        lam_limit_buy=0.5,
        lam_limit_sell=0.5,
        lam_mkt_buy=0.2,
        lam_mkt_sell=0.2,
        lam_cancel=0.1,
        # market structure params (NEW: limit_price_decay_rate)
        limit_price_decay_rate=0.5, # Rate for exponential decay of price distance (ticks^-1)
        # market maker params
        mm_base_spread=1.0,
        mm_skew_coef=0.05,
        mm_quote_interval=10.0, # Quote every 10 seconds
        # order sizing
        order_size_mean=1.0,
        # RNG seed
        seed=None,
    ):
        # semantics: all lam_* are events per second
        self.sim_duration = float(sim_duration)
        self.initial_price = float(initial_price)

        self.lam_limit_buy = float(lam_limit_buy)
        self.lam_limit_sell = float(lam_limit_sell)
        self.lam_mkt_buy = float(lam_mkt_buy)
        self.lam_mkt_sell = float(lam_mkt_sell)
        self.lam_cancel = float(lam_cancel)

        self.limit_price_decay_rate = float(limit_price_decay_rate)

        self.mm_base_spread = float(mm_base_spread)
        self.mm_skew_coef = float(mm_skew_coef)
        self.mm_quote_interval = float(mm_quote_interval)

        self.order_size_mean = float(order_size_mean)

        self.rng = np.random.default_rng(seed)

        # core objects
        self.book = OrderBook(initial_price=self.initial_price)
        self.mm = MarketMaker(book=self.book, mm_id="MM", base_spread=self.mm_base_spread, skew_coef=self.mm_skew_coef)

        # time
        self.time = 0.0
        self._last_mm_quote_time = -1e9

        # metrics
        self.spread_samples = []   # (time, spread)
        self.mm_times = []
        self.mm_pnls = []
        self.mm_invs = []

        # execution stats
        self.resting_exec_times = []  # seconds
        self.market_exec_count = 0
        self.resting_exec_count = 0
        self.trade_count = 0

        # initialize next-event times for each Poisson process
        self.next_limit_buy = self._sample_next(self.lam_limit_buy)
        self.next_limit_sell = self._sample_next(self.lam_limit_sell)
        self.next_mkt_buy = self._sample_next(self.lam_mkt_buy)
        self.next_mkt_sell = self._sample_next(self.lam_mkt_sell)
        self.next_cancel = self._sample_next(self.lam_cancel)
        self.next_mm_quote = self._sample_next(1.0 / self.mm_quote_interval) if self.mm_quote_interval > 0 else float('inf')


    # --------------------------
    def _sample_next(self, rate):
        """Return next event time offset (relative to current time). rate is events/sec."""
        if rate <= 0.0:
            return float('inf')
        # Exponential with mean 1/rate
        return self.time + self.rng.exponential(1.0 / rate)

    def _reference_price(self):
        return getattr(self.book, "last_trade_price", None) or self.initial_price
    
    def _sample_limit_price(self, side):
        """Samples a limit price using exponential decay from the best price on the opposite side."""
        ref = self._reference_price()
        tick = 0.5
        
        # Determine the reference price (BBO if available, otherwise mid)
        best_bid = self.book.best_bid()
        best_ask = self.book.best_ask()
        
        if side == 'buy':
            ref_price = best_bid.price if best_bid else ref
        else: # side == 'sell'
            ref_price = best_ask.price if best_ask else ref

        # Determine distance from the reference price in ticks using exponential decay
        if self.limit_price_decay_rate <= 0:
            distance_in_ticks = 1 # At least one tick away
        else:
            # Use exponential distribution to model distance. The inverse of the decay rate
            # is the expected distance in ticks.
            mean_ticks = 1.0 / self.limit_price_decay_rate
            distance = self.rng.exponential(mean_ticks)
            distance_in_ticks = max(1, round(distance))
        
        offset = distance_in_ticks * tick

        if side == "buy":
            price = ref_price - offset
        else: # sell
            price = ref_price + offset

        # Ensure price is above 0 and is on a tick increment
        return max(tick, round(price / tick) * tick)

    def _sample_order_size(self):
        if self.order_size_mean <= 1.0:
            return 1
        # Use a Poisson distribution for order size
        size = self.rng.poisson(lam=self.order_size_mean)
        return max(1, int(size))

    def _record_spread(self):
        b = self.book.best_bid()
        a = self.book.best_ask()
        if b is not None and a is not None:
            self.spread_samples.append((self.time, a.price - b.price))

    def _record_mm_state(self):
        self.mm_times.append(self.time)
        self.mm_pnls.append(self.mm.pnl())
        self.mm_invs.append(self.mm.inventory)

    def _cancel_random_order(self):
        """Attempt to cancel a random resting order."""
        self.book.cancel_random_order(rng=self.rng)

    def _process_trade(self, trade):
        # trade is expected to be dict with keys: price, quantity, time, buy_order, sell_order
        t = trade["time"]
        qty = trade["quantity"]
        self.trade_count += qty
        
        # Iterate over both sides to classify executions
        for role in ("buy_order", "sell_order"):
            order = trade.get(role)
            if order is None:
                continue
            
            trader = getattr(order, "trader_id", None)
            created = getattr(order, "time_created", t)
            duration = t - created

            if trader == "noise":
                # classify the order that *was executed*
                if order.type == 'limit' and duration > 1e-12:
                    # Resting limit order filled
                    self.resting_exec_times.append(duration)
                    self.resting_exec_count += 1
                elif order.type == 'market' and duration < 1e-12:
                    # Market order filled instantly
                    self.market_exec_count += 1
        
        # MM on_trade update
        for role, side in (("buy_order", "buy"), ("sell_order", "sell")):
            order = trade.get(role)
            if order is not None and getattr(order, "trader_id", None) == self.mm.mm_id:
                self.mm.on_trade(trade_price=trade["price"], side=side, quantity=trade["quantity"])

    def _generate_limit_order(self, side):
        price = self._sample_limit_price(side)
        size = self._sample_order_size()
        order = NoiseOrder(price=price, side=side, quantity=size, time_created=self.time, order_type='limit')
        trades = self.book.add_order(order)
        for trade in trades:
            self._process_trade(trade)

    def _generate_market_order(self, side):
        # For simplicity, market orders always take the best price with a size
        size = self._sample_order_size()
        # Price is irrelevant for market order generation, only for execution.
        order = NoiseOrder(price=0.0, side=side, quantity=size, time_created=self.time, order_type='market')
        trades = self.book.add_order(order)
        for trade in trades:
            self._process_trade(trade)

    def run(self, verbose=False):
        """
        CORE SIMULATION LOOP: Discrete Event Simulation
        """
        if verbose:
            print(f"Starting simulation. Seed: {self.rng._bit_generator.state['state']['keys'][0]}")

        # The loop runs until the simulation duration is reached
        while self.time < self.sim_duration:
            
            # 1. Determine the next event time
            next_times = {
                'limit_buy': self.next_limit_buy,
                'limit_sell': self.next_limit_sell,
                'mkt_buy': self.next_mkt_buy,
                'mkt_sell': self.next_mkt_sell,
                'cancel': self.next_cancel,
                'mm_quote': self.next_mm_quote,
            }
            
            event_type, next_time = min(next_times.items(), key=lambda x: x[1])

            if next_time == float('inf') or next_time > self.sim_duration:
                # No more events or end of simulation
                self.time = self.sim_duration
                break
            
            # 2. Advance time
            self.time = next_time
            
            # 3. Process the event
            if event_type == 'limit_buy':
                self._generate_limit_order('buy')
                self.next_limit_buy = self._sample_next(self.lam_limit_buy)
            
            elif event_type == 'limit_sell':
                self._generate_limit_order('sell')
                self.next_limit_sell = self._sample_next(self.lam_limit_sell)
            
            elif event_type == 'mkt_buy':
                self._generate_market_order('buy')
                self.next_mkt_buy = self._sample_next(self.lam_mkt_buy)
            
            elif event_type == 'mkt_sell':
                self._generate_market_order('sell')
                self.next_mkt_sell = self._sample_next(self.lam_mkt_sell)

            elif event_type == 'cancel':
                self._cancel_random_order()
                self.next_cancel = self._sample_next(self.lam_cancel)

            elif event_type == 'mm_quote':
                self.mm.quote(self.time, self.rng)
                self._record_mm_state()
                self.next_mm_quote = self._sample_next(1.0 / self.mm_quote_interval)
                
            # Record spread every time an event changes the book (or periodically)
            self._record_spread()


        # post-run summaries (metrics calculation template)
        try:
            avg_spread = float(np.mean([s for (_, s) in self.spread_samples])) if self.spread_samples else 0.0
            resting_exec_times_arr = np.array(self.resting_exec_times)
            
            median_exec = float(np.median(resting_exec_times_arr)) if resting_exec_times_arr.size > 0 else 0.0
            mean_exec = float(np.mean(resting_exec_times_arr)) if resting_exec_times_arr.size > 0 else 0.0
            p99_exec = float(np.percentile(resting_exec_times_arr, 99)) if resting_exec_times_arr.size > 0 else 0.0
            total_exec = self.resting_exec_count + self.market_exec_count

            # normalized metrics: per-hour and per-10k-trades
            per_hour_factor = 3600.0 / self.sim_duration if self.sim_duration > 0 else 0.0
            trades_per_hour = self.trade_count * per_hour_factor
            execs_per_hour = total_exec * per_hour_factor

            # PnL normalized
            mm_pnl = float(self.mm.pnl())
            mm_pnl_per_hour = mm_pnl * per_hour_factor
            mm_pnl_per_1k_trades = (mm_pnl / self.trade_count * 1000.0) if self.trade_count > 0 else 0.0

            # distribution diagnostics for resting exec times
            skew, kurt = skew_kurt(self.resting_exec_times) if resting_exec_times_arr.size > 0 else (0.0, 0.0)
        except ZeroDivisionError:
             print("Error: Simulation did not run (trade_count=0). Cannot calculate normalized metrics.")
             return {}
        except Exception as e:
            print(f"Error during post-run summary: {e}")
            return {}


        return {
            "avg_spread": avg_spread,
            "median_exec_time": median_exec,
            "mean_exec_time": mean_exec,
            "p99_exec_time": p99_exec,
            "resting_exec_count": self.resting_exec_count,
            "market_exec_count": self.market_exec_count,
            "total_executions": total_exec,
            "trade_count": self.trade_count,
            "trades_per_hour": trades_per_hour,
            "execs_per_hour": execs_per_hour,
            "mm_final_pnl": mm_pnl,
            "mm_final_pnl_per_hour": mm_pnl_per_hour,
            "mm_final_pnl_per_1k_trades": mm_pnl_per_1k_trades,
            "mm_final_inventory": self.mm.inventory,
            "avg_mm_inventory": float(np.mean(self.mm_invs)) if self.mm_invs else 0.0,
            "rest_exec_skew": skew,
            "rest_exec_kurtosis": kurt,
            # RAW data for time-series / distributions (not included in aggregation)
            # "spread_samples": self.spread_samples, 
            # "resting_exec_times": list(self.resting_exec_times),
        }


# ---------------------------
# Experiment driver
# ---------------------------
def run_full_experiment_suite(sim_duration, replications, scenarios, output_file, verbose=False):
    """
    Runs a full factorial suite using Common Random Numbers (CRN) and aggregates results.

    CRN is achieved by looping over replications first, and ensuring all scenarios
    within a replication use the same generated seed.
    """
    all_raw_rows = []
    all_agg_rows = []
    start = pytime.time()
    
    # 1. Loop over Replications (for CRN)
    for rep_id in range(1, replications + 1):
        # Generate a consistent seed for all scenarios in this replication
        # This is the implementation of Common Random Numbers (CRN)
        replication_seed = 1000000 + abs(hash((rep_id, "CRN_SEED_BASE"))) % 2_000_000_000
        
        print(f"\n--- Starting Replication {rep_id}/{replications} (CRN Seed: {replication_seed}) ---")
        
        scenario_results = []
        
        # 2. Loop over Scenarios
        for scenario_name, params in scenarios.items():
            
            print(f"  > Running Scenario {scenario_name}...", end="", flush=True)

            # --- Run Simulation with CRN Seed ---
            run_params = params.copy()
            run_params["sim_duration"] = sim_duration
            run_params["seed"] = replication_seed # Pass the consistent CRN seed
            
            try:
                sim = Simulation(**run_params)
                res = sim.run(verbose=verbose)
                
                # Append scenario identifier and replication ID to the raw results
                raw_row = {
                    "Type": "RAW",
                    "Scenario_Name": scenario_name,
                    "Replication_ID": rep_id,
                    **res,
                    **params
                }
                all_raw_rows.append(raw_row)
                scenario_results.append(res)
                print(f" (Trades: {res.get('trade_count', 0)})", end="", flush=True)

            except Exception as e:
                print(f" [ERROR: {e}]")
                # Append a placeholder error row
                all_raw_rows.append({
                    "Type": "RAW_ERROR",
                    "Scenario_Name": scenario_name,
                    "Replication_ID": rep_id,
                    "Error_Message": str(e),
                    **params
                })
        
        print(f"\n--- Replication {rep_id} Complete ---")

    # 3. Aggregate Results (Mean and CI)
    print("\n--- Aggregating Results (Mean & CI) ---")
    
    # Group raw data by scenario
    raw_df = pd.DataFrame(all_raw_rows)
    for scenario_name in scenarios.keys():
        scenario_df = raw_df[raw_df["Scenario_Name"] == scenario_name]
        if scenario_df.empty or scenario_df[scenario_df["Type"] == "RAW"].empty:
            print(f"Warning: No valid raw data for {scenario_name}. Skipping aggregation.")
            continue

        agg_row = {
            "Type": "AGGREGATE",
            "Scenario_Name": scenario_name,
            "Replication_ID": "MEAN/CI",
            **scenarios[scenario_name]
        }
        
        # Compute Mean and CI for every metric reported by run()
        metrics = [col for col in raw_df.columns if col not in ("Type", "Scenario_Name", "Replication_ID", "Error_Message") and col not in scenarios[scenario_name]]
        
        for metric in metrics:
            # Gather valid data points for the metric
            data = scenario_df[scenario_df["Type"] == "RAW"][metric].dropna().tolist()
            if not data: continue

            mean_val, ci_val = mean_ci(data)
            
            agg_row[f"{metric}_Mean"] = mean_val
            agg_row[f"{metric}_CI_95"] = ci_val
            
            # Optional: Report single values that don't need aggregation (like input params)
            if metric.endswith("_Mean") or metric.endswith("_CI_95"):
                continue

        all_agg_rows.append(agg_row)

    # 4. Save to CSV
    # Combine raw and aggregated data for a single, comprehensive output
    final_df = pd.concat([raw_df, pd.DataFrame(all_agg_rows)], ignore_index=True)
    
    # Reorder columns to put identifiers and experiment variables first
    cols = ["Type", "Scenario_Name", "Replication_ID"] + list(scenarios[list(scenarios.keys())[0]].keys()) + [col for col in final_df.columns if col not in ("Type", "Scenario_Name", "Replication_ID") and col not in scenarios[list(scenarios.keys())[0]].keys()]
    final_df = final_df[cols]
    
    elapsed = pytime.time() - start
    print(f"\nCompleted suite in {elapsed:.1f} s")
    final_df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")
    
    return final_df


# ---------------------------
# Main / experiments config
# ---------------------------
if __name__ == "__main__":
    
    # --- Experiment Configuration ---
    # Define the default parameters for the simulation
    DEFAULT_PARAMS = {
        "sim_duration": 23400.0, # 6.5 hours in seconds (typical trading day)
        "initial_price": 100.0,
        "lam_limit_buy": 0.5,
        "lam_limit_sell": 0.5,
        "lam_mkt_buy": 0.2,
        "lam_mkt_sell": 0.2,
        "lam_cancel": 0.1,
        "mm_skew_coef": 0.05,
        "mm_quote_interval": 10.0,
    }

    # Define the 3 factors and their 2 levels (Low, High)
    FACTORS = {
        # X1: Order strategy
        "order_size_mean": [1.0, 5.0],  # Low (1 share), High (5 shares)
        # X2: Market Structure
        "limit_price_decay_rate": [0.5, 0.1], # High decay (0.5, tight book), Low decay (0.1, wide book)
        # X3: MM Strat
        "mm_base_spread": [0.5, 2.0], # Low spread (0.5, aggressive MM), High spread (2.0, passive MM)
    }

    # Generate the 8 scenarios (Full Factorial Design)
    SCENARIOS = {}
    
    # itertools.product generates all 2^3 combinations
    for i, combination in enumerate(itertools.product(*FACTORS.values())):
        scenario_params = DEFAULT_PARAMS.copy()
        scenario_name = f"S{i+1}"
        
        # Map combination values back to factor names
        scenario_params["order_size_mean"] = combination[0]
        scenario_params["limit_price_decay_rate"] = combination[1]
        scenario_params["mm_base_spread"] = combination[2]
        
        # Create a cleaner set of input parameters for the output report
        input_params = {
            "order_size_mean": combination[0],
            "limit_price_decay_rate": combination[1],
            "mm_base_spread": combination[2],
        }
        
        SCENARIOS[scenario_name] = {**input_params} # Use input_params to preserve clean input values
        
    print(f"Defined {len(SCENARIOS)} scenarios for the full factorial experiment.")
    
    # --- Execute the Suite ---
    
    # Use 5 replications as required
    REPLICATIONS = 5 
    
    run_full_experiment_suite(
        sim_duration=DEFAULT_PARAMS["sim_duration"],
        replications=REPLICATIONS,
        scenarios=SCENARIOS,
        output_file="simulation_results.csv",
        verbose=False
    )
    
    print("\nSimulation complete. Check 'simulation_results.csv' for raw and aggregated data.")
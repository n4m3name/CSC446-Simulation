#!/usr/bin/env python3
"""
LOB simulation (scaling-fixed, per-process Poisson sampling)

STRIPPED CODE: ALL CORE SIMULATION LOOP LOGIC AND EXPERIMENT CONFIGURATION REMOVED.
This file now contains only the necessary imports, statistical utility functions, and
the structure of the Simulation class and run_suite for reuse in new experiments.

Dependencies:
- numpy, pandas
- (optional) scipy for t critical and distribution stats

Local dependencies:
- order_book.py : must implement OrderBook with methods used below
- market_maker.py : must implement MarketMaker used below

Author: Gemini (based on original file)
"""
import argparse
import math
import sys
from collections import defaultdict
import time as pytime

import numpy as np
import pandas as pd

# Optional: scipy for t-critical values and distribution stats
try:
    import scipy.stats as sps
    SCIPY = True
except Exception:
    SCIPY = False

# Local order book and market maker (must exist)
from order_book import OrderBook
from market_maker import MarketMaker


# ---------------------------
# Utility / statistics
# ---------------------------
def t_crit(n, confidence=0.95):
    if n <= 1:
        return 1.96
    if SCIPY:
        return float(sps.t.ppf(1 - (1 - confidence) / 2, df=n - 1))
    # fallback lookup
    table = {5: 2.776, 10: 2.262, 30: 2.045}
    return table.get(n, 1.96)


def mean_ci(arr, confidence=0.95):
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    mean = float(np.mean(arr))
    if n < 2:
        return mean, 0.0
    se = float(np.std(arr, ddof=1) / math.sqrt(n))
    t = t_crit(n, confidence)
    return mean, t * se


def skew_kurt(arr):
    """Return (skewness, excess_kurtosis). Prefer scipy; fallback simple formulas."""
    if len(arr) < 2:
        return 0.0, 0.0
    if SCIPY:
        return float(sps.skew(arr, bias=False)), float(sps.kurtosis(arr, fisher=True, bias=False))
    # fallback: sample skewness and kurtosis (biased/approx)
    a = np.asarray(arr, dtype=float)
    m = a.mean()
    s = a.std(ddof=1)
    if s == 0:
        return 0.0, 0.0
    z = (a - m) / s
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4)) - 3.0
    return skew, kurt


# ---------------------------
# Simulation class
# ---------------------------
class Simulation:
    """
    Simulation class containing parameters, initialization, and helper functions.
    NOTE: The core simulation loop and event logic have been removed from the run() method.
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
        # market maker params
        mm_base_spread=1.0,
        mm_skew_coef=0.05,
        mm_quote_interval=0.0,
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

        self.mm_base_spread = float(mm_base_spread)
        self.mm_skew_coef = float(mm_skew_coef)
        self.mm_quote_interval = float(mm_quote_interval)

        self.order_size_mean = float(order_size_mean)

        self.rng = np.random.default_rng(seed)

        # core objects
        self.book = OrderBook()
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
        # If a rate is 0, set next time to +inf so it never fires.
        self.next_limit_buy = self._sample_next(self.lam_limit_buy)
        self.next_limit_sell = self._sample_next(self.lam_limit_sell)
        self.next_mkt_buy = self._sample_next(self.lam_mkt_buy)
        self.next_mkt_sell = self._sample_next(self.lam_mkt_sell)
        self.next_cancel = self._sample_next(self.lam_cancel)

    # --------------------------
    def _sample_next(self, rate):
        """Return next event time offset (relative to current time). rate is events/sec."""
        if rate <= 0.0:
            return float('inf')
        # Exponential with mean 1/rate
        return self.time + self.rng.exponential(1.0 / rate)

    def _reference_price(self):
        return getattr(self.book, "last_trade_price", None) or self.initial_price

    def _sample_limit_price(self):
        ref = self._reference_price()
        tick = 0.5
        offset = int(self.rng.integers(-5, 6))
        price = ref + offset * tick
        return max(tick, price)

    def _sample_order_size(self):
        if self.order_size_mean <= 1.0:
            return 1
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
        """Attempt to cancel a random resting order; try book-level methods first."""
        if hasattr(self.book, "cancel_random_order"):
            try:
                self.book.cancel_random_order(rng=self.rng)
                return
            except TypeError:
                try:
                    self.book.cancel_random_order()
                    return
                except Exception:
                    pass
        # fallback: cancel best on a random side
        sides = []
        if self.book.best_bid() is not None:
            sides.append("buy")
        if self.book.best_ask() is not None:
            sides.append("sell")
        if not sides:
            return
        side = self.rng.choice(sides)
        self.book.cancel_best(side)

    def _process_trade(self, trade):
        # trade is expected to be dict with keys: price, quantity, time, buy_order, sell_order
        t = trade["time"]
        qty = trade["quantity"]
        self.trade_count += qty
        for role in ("buy_order", "sell_order"):
            order = trade.get(role)
            if order is None:
                continue
            trader = getattr(order, "trader_id", None)
            created = getattr(order, "time_created", t)
            duration = t - created
            if trader == "noise":
                # classify
                if duration > 1e-12:
                    self.resting_exec_times.append(duration)
                    self.resting_exec_count += 1
                else:
                    self.market_exec_count += 1
        # MM on_trade update
        for role, side in (("buy_order", "buy"), ("sell_order", "sell")):
            order = trade.get(role)
            if order is not None and getattr(order, "trader_id", None) == self.mm.mm_id:
                self.mm.on_trade(trade_price=trade["price"], side=side, quantity=trade["quantity"])

    def run(self, verbose=False):
        """
        ### CORE SIMULATION LOOP REMOVED ###
        Please implement the event-driven simulation loop (while self.time < self.sim_duration: ...) here.
        The simulation logic should consume events based on self.next_* times, update self.time,
        call self.book methods, and call self._process_trade on results.
        """

        # --- Placeholder for the removed loop: Implement your simulation logic here ---

        # --------------------------------------------------------------------------

        # post-run summaries (metrics calculation template)
        try:
            avg_spread = float(np.mean([s for (_, s) in self.spread_samples])) if self.spread_samples else 0.0
            median_exec = float(np.median(self.resting_exec_times)) if self.resting_exec_times else 0.0
            mean_exec = float(np.mean(self.resting_exec_times)) if self.resting_exec_times else 0.0
            p99_exec = float(np.percentile(self.resting_exec_times, 99)) if self.resting_exec_times else 0.0
            total_exec = self.resting_exec_count + self.market_exec_count

            # normalized metrics: per-hour and per-10k-trades
            per_hour_factor = 3600.0 / self.sim_duration
            trades_per_hour = self.trade_count * per_hour_factor
            execs_per_hour = total_exec * per_hour_factor

            # PnL normalized
            mm_pnl = float(self.mm.pnl())
            mm_pnl_per_hour = mm_pnl * per_hour_factor
            mm_pnl_per_1k_trades = (mm_pnl / self.trade_count * 1000.0) if self.trade_count > 0 else 0.0

            # distribution diagnostics for resting exec times
            skew, kurt = skew_kurt(self.resting_exec_times) if self.resting_exec_times else (0.0, 0.0)
        except ZeroDivisionError:
             print("Error: Simulation did not run (trade_count=0). Cannot calculate normalized metrics.")
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
            "spread_samples": self.spread_samples,
            "resting_exec_times": list(self.resting_exec_times),
        }


# ---------------------------
# Experiment driver
# ---------------------------
def run_suite(sim_duration, replications, experiments, output_file, verbose=False):
    """
    Function to run a suite of experiments, handle replications, and aggregate statistics
    with confidence intervals.

    NOTE: This function is intact to handle the required statistical aggregation.
    """
    rows = []
    start = pytime.time()
    for exp_name, exp in experiments.items():
        var = exp.get("variable_name", "N/A")
        values = exp.get("values", [None])
        defaults = exp.get("defaults", {})
        
        print(f"\n=== Running {exp_name} (sweep {var}) ===")
        
        # This section needs adjustment for multi-variable sweeps in the calling script.
        # It currently supports single-variable sweeps as per the original code.
        for val in values:
            print(f"-- {var} = {val} ", end="", flush=True)
            res_list = []
            for rep in range(replications):
                seed = 1000000 + abs(hash((exp_name, var, val, rep))) % 2_000_000_000
                params = defaults.copy()
                params["sim_duration"] = sim_duration
                if var == "lam_limit":
                    params["lam_limit_buy"] = val
                    params["lam_limit_sell"] = val
                else:
                    if var and var != "N/A":
                        params[var] = val
                sim = Simulation(seed=seed, **params)
                res = sim.run(verbose=verbose)
                res_list.append(res)
            
            # aggregation logic remains intact
            def gather(k):
                return [r[k] for r in res_list if k in r]

            def mean_ci_from_reps(k):
                arr = gather(k)
                return mean_ci(arr)

            # NOTE: All metric calculations remain intact for reuse.
            avg_spread_mean, avg_spread_ci = mean_ci_from_reps("avg_spread")
            median_exec_mean, median_exec_ci = mean_ci_from_reps("median_exec_time")
            mean_exec_mean, mean_exec_ci = mean_ci_from_reps("mean_exec_time")
            p99_mean, p99_ci = mean_ci_from_reps("p99_exec_time")
            mm_pnl_mean, mm_pnl_ci = mean_ci_from_reps("mm_final_pnl")
            mm_pnl_hour_mean, mm_pnl_hour_ci = mean_ci_from_reps("mm_final_pnl_per_hour")
            mm_inv_mean, mm_inv_ci = mean_ci_from_reps("mm_final_inventory")
            trades_per_hour_mean, trades_per_hour_ci = mean_ci_from_reps("trades_per_hour")
            execs_per_hour_mean, execs_per_hour_ci = mean_ci_from_reps("execs_per_hour")
            skew_mean, skew_ci = mean_ci_from_reps("rest_exec_skew")
            kurt_mean, kurt_ci = mean_ci_from_reps("rest_exec_kurtosis")

            print("->", end=" ")
            print(
                f"Spread={avg_spread_mean:.3f}±{avg_spread_ci:.3f}, "
                f"MM_PnL/hr={mm_pnl_hour_mean:.2f}±{mm_pnl_hour_ci:.2f}"
            )

            # store a CSV-friendly row
            row = {
                "Experiment": exp_name,
                "Independent_Var": var,
                "Value": val,
                "Spread_Mean": avg_spread_mean,
                "Spread_CI": avg_spread_ci,
                "Median_Exec_Mean_s": median_exec_mean,
                "Median_Exec_CI_s": median_exec_ci,
                "Mean_Exec_Mean_s": mean_exec_mean,
                "Mean_Exec_CI_s": mean_exec_ci,
                "P99_Exec_Mean_s": p99_mean,
                "P99_Exec_CI_s": p99_ci,
                "Trades_per_hour_Mean": trades_per_hour_mean,
                "Trades_per_hour_CI": trades_per_hour_ci,
                "Execs_per_hour_Mean": execs_per_hour_mean,
                "Execs_per_hour_CI": execs_per_hour_ci,
                "MM_PnL_Mean": mm_pnl_mean,
                "MM_PnL_CI": mm_pnl_ci,
                "MM_PnL_per_hour_Mean": mm_pnl_hour_mean,
                "MM_PnL_per_hour_CI": mm_pnl_hour_ci,
                "MM_Inventory_Mean": mm_inv_mean,
                "MM_Inventory_CI": mm_inv_ci,
                "RestExec_Skew_Mean": skew_mean,
                "RestExec_Skew_CI": skew_ci,
                "RestExec_Kurt_Mean": kurt_mean,
                "RestExec_Kurt_CI": kurt_ci,
            }
            rows.append(row)

    elapsed = pytime.time() - start
    print(f"\nCompleted suite in {elapsed:.1f} s")
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")
    return df


# ---------------------------
# Main / experiments config
# ---------------------------
if __name__ == "__main__":
    # --- ALL EXPERIMENT CONFIGURATION AND ARGPARSING REMOVED ---
    # Please define your new experiments and call run_suite(..) here.
    
    print("Code stripped for new simulation design. Please configure your experiments here and call run_suite.")
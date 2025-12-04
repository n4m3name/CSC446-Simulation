# simulation.py
"""
Main simulation loop for the limit order book.
... (rest of the docstring remains the same)
"""

import numpy as np
import argparse
import pandas as pd

from order_book import OrderBook
from market_maker import MarketMaker


class Simulation:
    def __init__(
        self,
        sim_duration=60.0,          # total simulation time (arbitrary units)
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
        Returns a dict of summary stats and time series.
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
        avg_spread = float(np.mean(self.spreads)) if self.spreads else None
        
        # Median of RESTING orders (Limit orders)
        median_exec = float(np.median(self.exec_times)) if self.exec_times else None

        # Calculate Total Executions (Resting Limit + Instant Market)
        total_executions = len(self.exec_times) + self.mkt_exec_count

        return {
            "avg_spread": avg_spread,
            "median_exec_time": median_exec,
            "total_executions": total_executions,  # Includes Market & Limit
            "mm_final_pnl": self.mm.pnl(),
            "mm_final_inventory": self.mm.inventory,
            "spread_times": self.spread_times,
            "spreads": self.spreads,
            "exec_times": self.exec_times,
            "mm_times": self.mm_times,
            "mm_pnls": self.mm_pnls,
            "mm_inventory": self.mm_inventory,
        }

if __name__ == "__main__":

    # --- CONSTANTS: Define Scenarios ONCE ---
    SIM_DURATION = 600.0
    INITIAL_PRICE = 100.0
    MM_SKEW_COEF = 0.05

    BASELINE_PARAMS = dict(
        sim_duration=SIM_DURATION,
        initial_price=INITIAL_PRICE,
        lam_limit_buy=0.6,
        lam_limit_sell=0.6,
        lam_mkt_buy=0.2,
        lam_mkt_sell=0.2,
        lam_cancel=0.1,
        mm_base_spread=2.0,
        mm_skew_coef=MM_SKEW_COEF,
        seed=123,
    )

    ACTIVITY_SCENARIOS = {
        "Low_Activity": {
            "lam_limit_buy": 0.6,
            "lam_limit_sell": 0.6,
            "lam_mkt_buy": 0.2,
            "lam_mkt_sell": 0.2,
            "lam_cancel": 0.1,
            "mm_base_spread": 2.0,
            "seed": 100,
        },
        "Mid_Activity": {
            "lam_limit_buy": 2.0,
            "lam_limit_sell": 2.0,
            "lam_mkt_buy": 0.8,
            "lam_mkt_sell": 0.8,
            "lam_cancel": 3.0,
            "mm_base_spread": 1.0,
            "seed": 200,
        },
        "High_Activity": {
            "lam_limit_buy": 5.0,
            "lam_limit_sell": 5.0,
            "lam_mkt_buy": 2.0,
            "lam_mkt_sell": 2.0,
            "lam_cancel": 7.0,
            "mm_base_spread": 0.5,
            "seed": 300,
        },
    }

    STRESS_BASE = dict(
        sim_duration=SIM_DURATION,
        initial_price=INITIAL_PRICE,
        lam_limit_buy=0.6,
        lam_limit_sell=0.6,
        lam_mkt_buy=0.2,
        lam_mkt_sell=0.2,
        lam_cancel=0.1,
        mm_base_spread=2.0,
        mm_skew_coef=MM_SKEW_COEF,
    )
    
    STRESS_SCENARIOS = {
        "High order flow": {
            "seed": 124,
            "lam_limit_buy": 1.0,
            "lam_limit_sell": 1.0,
            "lam_mkt_buy": 0.4,
            "lam_mkt_sell": 0.4
        },
        "Low order flow": {
            "seed": 125,
            "lam_limit_buy": 0.3,
            "lam_limit_sell": 0.3,
            "lam_mkt_buy": 0.1,
            "lam_mkt_sell": 0.1
        },
        "High cancellations": {"seed": 126, "lam_cancel": 0.3},
        "Low cancellations": {"seed": 127, "lam_cancel": 0.02},
        "Aggressive MM": {"seed": 128, "mm_base_spread": 1.0, "mm_skew_coef": 0.02},
        "Passive MM": {"seed": 129, "mm_base_spread": 3.0, "mm_skew_coef": 0.1},
        "Buy pressure": {"seed": 130, "lam_limit_buy": 0.9, "lam_mkt_buy": 0.4},
        "Sell pressure": {"seed": 131, "lam_limit_sell": 0.9, "lam_mkt_sell": 0.4},
    }


    # ------------------------------------------------------------------
    # Helper: run a scenario and return results dict
    # ------------------------------------------------------------------
    def run_scenario(label, **params):
        """Runs the simulation and prints key results."""
        sim = Simulation(**params)
        results = sim.run()
        print("\n=== Scenario:", label, "===")
        print("Average spread:       ", results["avg_spread"])
        print("Median exec time:     ", results["median_exec_time"])
        print("MM final P&L:         ", results["mm_final_pnl"])
        print("MM final inventory:   ", results["mm_final_inventory"])
        print("Total executions:     ", results["total_executions"])
        print("-" * 40)
        return results

    # ------------------------------------------------------------------
    # Helper: Run the Activity Scenario Suite
    # ------------------------------------------------------------------
    def run_activity_suite():
        print("Running Activity Scenarios (Low/Mid/High)...")
        for name, params in ACTIVITY_SCENARIOS.items():
            # Merge base parameters with scenario-specific overrides
            full_params = {
                "sim_duration": SIM_DURATION,
                "initial_price": INITIAL_PRICE,
                "mm_skew_coef": MM_SKEW_COEF,
                **params
            }
            run_scenario(name, **full_params)

    # ------------------------------------------------------------------
    # Helper: Run the Stress-Test Scenario Suite
    # ------------------------------------------------------------------
    def run_stress_suite():
        print("Running Stress-Test Scenarios...")
        # Start with Baseline as the first test in the stress suite for context
        run_scenario("Baseline", **BASELINE_PARAMS) 

        for name, params in STRESS_SCENARIOS.items():
            # Merge the STRESS_BASE with the scenario-specific overrides
            full_params = {**STRESS_BASE, **params}
            run_scenario(name, **full_params)
            
    # ------------------------------------------------------------------
    # Command Line Argument Parsing
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="LOB Simulation Runner")
    parser.add_argument(
        "--activity",
        action="store_true",
        help="Run Low/Mid/High Activity scenarios (VFV-style rates)"
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Run Extended Stress-Test scenario suite (arrival/cancel/MM variations)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run baseline, activity, and stress suites sequentially"
    )
    args = parser.parse_args()

    # When --all is set, reuse existing branches by enabling both flags for execution flow
    if args.all:
        args.activity = True
        args.stress = True
    
    # ------------------------------------------------------------------
    # MODE SELECTION (Now only using the execution helper functions)
    # ------------------------------------------------------------------

    is_baseline_only = not args.activity and not args.stress

    if args.all:
        print("Running full suite: Baseline, Activity, Stress...\n")
        # Baseline is run as part of the stress suite now, but running it first for clarity
        run_scenario("Baseline", **BASELINE_PARAMS)
        run_activity_suite()
        run_stress_suite()
        
    elif args.activity:
        run_activity_suite()

    elif args.stress:
        run_stress_suite()

    elif is_baseline_only:
        print("Running single baseline simulation...\n")
        run_scenario("Baseline", **BASELINE_PARAMS)
        
    # Exit is no longer needed on every branch since the logic is consolidated
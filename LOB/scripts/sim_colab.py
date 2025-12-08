#!/usr/bin/env python3
"""
Simulation driver for LOB experiments (full factorial, CRN, detailed metrics).

- Relies on local files: order.py, order_book.py, market_maker.py, bid.py, ask.py
- Produces:
    - results/simulation_results_full.csv  (all raw runs + aggregate rows)
    - results/<scenario>/run_<r>.csv       (optional per-run raw output)
Author: Assistant (rewritten to match user's experimental spec)
"""
import os
import math
import time as pytime
import random
import itertools
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np
import pandas as pd

# local modules
from order import Order
from order_book import OrderBook
from market_maker import MarketMaker

# ---------------------------
# Utilities
# ---------------------------
def t_crit(n, confidence=0.95):
    """Return t critical value for two-sided CI. Fallback to z if small."""
    if n <= 1:
        return 1.96
    # common small-sample values: df=4 -> 2.776 (we use this for 5 reps)
    if n == 5:
        return 2.776
    # approximate
    return 1.96

def safe_mean_ci(arr: List[float], confidence=0.95):
    """Return (mean, half_width_CI). Uses t-critical with n-1 df."""
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    mean = float(np.mean(arr))
    if n < 2:
        return mean, 0.0
    std = float(np.std(arr, ddof=1))
    se = std / math.sqrt(n)
    t = t_crit(n, confidence)
    return mean, t * se

# ---------------------------
# Depth helpers (inspect heaps)
# ---------------------------
def extract_orders_from_heap(heap_list):
    """Given a heap as stored in BidHeap/AskHeap (tuples), return order objects present."""
    # Each heap entry is like: (key1, key2, order_id, order_obj) - order is last element
    return [entry[3] for entry in heap_list if len(entry) >= 4]

def compute_depth_profile(book: OrderBook, midprice: float, tick: float = 0.01, levels=(1,2,3,4,5)):
    """
    Compute depth at discrete tick levels from midprice.
    Returns dict of level->volume (sum of quantities) on both sides and total within +/-5 ticks.
    """
    bids = extract_orders_from_heap(book.bids.heap)
    asks = extract_orders_from_heap(book.asks.heap)

    depth = {"bid_levels": {}, "ask_levels": {}, "depth_within_5_ticks": 0.0, "mm_depth_within_5_ticks": 0.0}
    # initialize levels
    for L in levels:
        depth["bid_levels"][L] = 0
        depth["ask_levels"][L] = 0

    # accumulate bids
    for order in bids:
        if not getattr(order, "active", True):
            continue
        price = order.price
        qty = getattr(order, "quantity", 0)
        # compute ticks difference (mid - bid)
        ticks = int(round((midprice - price) / tick)) if midprice is not None else None
        if ticks is not None and ticks >= 0:
            for L in levels:
                if ticks == L - 1:  # level 1 = best price (0 ticks away)
                    depth["bid_levels"][L] += qty
        # within +/-5 ticks:
        if midprice is not None and abs(price - midprice) <= 5 * tick:
            depth["depth_within_5_ticks"] += qty
            if getattr(order, "trader_id", None) == "MM":
                depth["mm_depth_within_5_ticks"] += qty

    # accumulate asks
    for order in asks:
        if not getattr(order, "active", True):
            continue
        price = order.price
        qty = getattr(order, "quantity", 0)
        ticks = int(round((price - midprice) / tick)) if midprice is not None else None
        if ticks is not None and ticks >= 0:
            for L in levels:
                if ticks == L - 1:
                    depth["ask_levels"][L] += qty
        if midprice is not None and abs(price - midprice) <= 5 * tick:
            depth["depth_within_5_ticks"] += qty
            if getattr(order, "trader_id", None) == "MM":
                depth["mm_depth_within_5_ticks"] += qty

    return depth

# ---------------------------
# Single-run simulation
# ---------------------------
def run_single_simulation(params: Dict[str, Any], seed: int, verbose: bool=False) -> Dict[str, Any]:
    """
    Run one replication using the provided params and seed.
    Returns a dictionary of metrics (Y-variables) and some raw traces.
    """
    # set python RNG for CRN-controlled randomness
    random.seed(seed)
    # create book & mm
    book = OrderBook()
    mm = MarketMaker(book=book, mm_id="MM", base_spread=params["mm_base_spread"], skew_coef=params.get("mm_skew_coef", 0.0))

    # initialize book with a small touching inside to allow MM quoting to compute midprice
    init_price = float(params.get("initial_price", 100.0))
    book.submit_limit("init", "buy", init_price - 0.01, 10, -1.0)
    book.submit_limit("init", "sell", init_price + 0.01, 10, -1.0)

    # simulation state
    sim_time = 0.0
    sim_duration = float(params.get("sim_duration", 23400.0))
    lam_limit_buy = params.get("lam_limit_buy", 0.5)
    lam_limit_sell = params.get("lam_limit_sell", 0.5)
    lam_mkt_buy = params.get("lam_mkt_buy", 0.2)
    lam_mkt_sell = params.get("lam_mkt_sell", 0.2)
    lam_cancel = params.get("lam_cancel", 0.1)

    # We'll manage independent Poisson processes for each event type (next-event sampling)
    # helper to sample next exponential jump given rate (events/sec)
    def sample_next(current, rate):
        if rate <= 0.0:
            return float("inf")
        return current + random.expovariate(rate)

    # initialize next-event times
    next_limit_buy = sample_next(sim_time, lam_limit_buy)
    next_limit_sell = sample_next(sim_time, lam_limit_sell)
    next_mkt_buy = sample_next(sim_time, lam_mkt_buy)
    next_mkt_sell = sample_next(sim_time, lam_mkt_sell)
    next_cancel = sample_next(sim_time, lam_cancel)
    next_mm_quote = 0.0  # start immediate quoting if mm_quote_interval small
    mm_quote_interval = float(params.get("mm_quote_interval", 10.0))

    # bookkeeping & metrics
    spread_samples = []            # (time, spread)
    mm_pnl_times = []              # store MM pnl over time
    mm_inv_times = []              # store mm inventory over time
    midprice_series = []           # time-ordered midprices
    trade_records = []             # raw trades
    resting_exec_times = []        # durations for resting limit orders from create->fill

    trade_count = 0
    resting_exec_count = 0
    market_exec_count = 0

    # For realized spread approx we can store when MM's resting orders are executed the difference vs mid
    mm_realized_spreads = []

    # runtime timer
    t0 = pytime.time()

    # helper: sample limit price (exponential distance in ticks)
    def sample_limit_price(side):
        # reference price is mid if available else last trade price else initial price
        ref = book.last_trade_price if book.last_trade_price is not None else init_price
        tick = params.get("tick", 0.01)
        decay = params.get("limit_price_decay_rate", 1.0)
        if decay <= 0:
            dist = 1.0
        else:
            dist = random.expovariate(decay)
        # round distance to ticks
        dist_ticks = max(1, int(math.ceil(dist / tick)))
        offset = dist_ticks * tick
        price = ref - offset if side == "buy" else ref + offset
        # align to tick
        price = round(price / tick) * tick
        return float(max(tick, price))

    # main event loop (discrete-event)
    while sim_time < sim_duration:
        next_time = min(next_limit_buy, next_limit_sell, next_mkt_buy, next_mkt_sell, next_cancel, next_mm_quote)
        if next_time == float("inf") or next_time > sim_duration:
            sim_time = sim_duration
            break
        sim_time = next_time

        # which event fired (tie order deterministic by if/elif)
        if next_limit_buy == next_time:
            # submit limit buy
            price = sample_limit_price("buy")
            size = int(max(1, round(params.get("order_size_mean", 1))))
            book.submit_limit("noise", "buy", price, size, sim_time)
            next_limit_buy = sample_next(sim_time, lam_limit_buy)

        elif next_limit_sell == next_time:
            price = sample_limit_price("sell")
            size = int(max(1, round(params.get("order_size_mean", 1))))
            book.submit_limit("noise", "sell", price, size, sim_time)
            next_limit_sell = sample_next(sim_time, lam_limit_sell)

        elif next_mkt_buy == next_time:
            size = int(max(1, round(params.get("order_size_mean", 1))))
            trades = book.submit_market("noise", "buy", size, sim_time)
            for tr in trades:
                trade_records.append(tr)
                trade_count += tr.get("quantity", 0)
                # classify resting vs market execution (market taker has buy_order None or sell_order None)
                # For market taker buy, filled at resting ask (sell_order exists)
                if tr.get("sell_order") is not None:
                    # resting order filled
                    created = getattr(tr["sell_order"], "time_created", sim_time)
                    if created >= 0:
                        resting_exec_times.append(tr["time"] - created)
                        resting_exec_count += 1
                        if getattr(tr["sell_order"], "trader_id", None) == "MM":
                            mm.on_trade(tr["price"], "sell", tr["quantity"])
                            # approximate realized spread: mid - quoted: mid_at_trade - ask_price
                            mid = mm.midprice() or book.last_trade_price or init_price
                            mm_realized_spreads.append((mid - tr["price"]))
                    else:
                        market_exec_count += 1
                else:
                    # no resting sell => maybe filled against mm's bid or no liquidity
                    market_exec_count += 1
            next_mkt_buy = sample_next(sim_time, lam_mkt_buy)

        elif next_mkt_sell == next_time:
            size = int(max(1, round(params.get("order_size_mean", 1))))
            trades = book.submit_market("noise", "sell", size, sim_time)
            for tr in trades:
                trade_records.append(tr)
                trade_count += tr.get("quantity", 0)
                if tr.get("buy_order") is not None:
                    created = getattr(tr["buy_order"], "time_created", sim_time)
                    if created >= 0:
                        resting_exec_times.append(tr["time"] - created)
                        resting_exec_count += 1
                        if getattr(tr["buy_order"], "trader_id", None) == "MM":
                            mm.on_trade(tr["price"], "buy", tr["quantity"])
                            mid = mm.midprice() or book.last_trade_price or init_price
                            mm_realized_spreads.append((tr["price"] - mid))
                    else:
                        market_exec_count += 1
                else:
                    market_exec_count += 1
            next_mkt_sell = sample_next(sim_time, lam_mkt_sell)

        elif next_cancel == next_time:
            # simple cancel: randomly cancel best on random side if exists
            side = random.choice(["buy", "sell"])
            book.cancel_best(side)
            next_cancel = sample_next(sim_time, lam_cancel)

        elif next_mm_quote == next_time:
            # MM posts quotes
            mm.quote(sim_time)
            # record mm state
            mm_pnl_times.append((sim_time, float(mm.pnl())))
            mm_inv_times.append((sim_time, float(mm.inventory)))
            # schedule next quote
            if mm_quote_interval > 0:
                next_mm_quote = sim_time + mm_quote_interval
            else:
                next_mm_quote = float("inf")

        # after each event, try matching limit crossing until none
        while True:
            trade = book.try_match_limit_crossing(sim_time)
            if trade is None:
                break
            trade_records.append(trade)
            trade_count += trade.get("quantity", 0)
            # classify and update MM if involved
            buy_ord = trade.get("buy_order")
            sell_ord = trade.get("sell_order")
            # check resting order sides
            if buy_ord is not None and getattr(buy_ord, "time_created", -1) >= 0:
                resting_exec_times.append(trade["time"] - buy_ord.time_created)
                resting_exec_count += 1
                if getattr(buy_ord, "trader_id", None) == "MM":
                    mm.on_trade(trade["price"], "buy", trade["quantity"])
                    mid = mm.midprice() or book.last_trade_price or init_price
                    mm_realized_spreads.append((mid - trade["price"]))
            if sell_ord is not None and getattr(sell_ord, "time_created", -1) >= 0:
                resting_exec_times.append(trade["time"] - sell_ord.time_created)
                resting_exec_count += 1
                if getattr(sell_ord, "trader_id", None) == "MM":
                    mm.on_trade(trade["price"], "sell", trade["quantity"])
                    mid = mm.midprice() or book.last_trade_price or init_price
                    mm_realized_spreads.append((trade["price"] - mid))

        # record spread and midprice samples
        bb = book.best_bid()
        ba = book.best_ask()
        if bb is not None and ba is not None:
            spread_samples.append((sim_time, float(ba.price - bb.price)))
            mid = 0.5 * (bb.price + ba.price)
            midprice_series.append(mid)

    # end simulation
    runtime = pytime.time() - t0

    # post-run metrics
    avg_spread = float(np.mean([s for (_, s) in spread_samples])) if spread_samples else 0.0
    median_exec = float(np.median(resting_exec_times)) if resting_exec_times else 0.0
    mean_exec = float(np.mean(resting_exec_times)) if resting_exec_times else 0.0
    p99_exec = float(np.percentile(resting_exec_times, 99)) if resting_exec_times else 0.0
    total_exec = resting_exec_count + market_exec_count

    trades_per_hour = trade_count * (3600.0 / sim_duration) if sim_duration > 0 else 0.0
    execs_per_hour = total_exec * (3600.0 / sim_duration) if sim_duration > 0 else 0.0

    mm_final_pnl = float(mm.pnl())
    mm_final_pnl_per_hour = mm_final_pnl * (3600.0 / sim_duration) if sim_duration > 0 else 0.0
    mm_final_pnl_per_1k_trades = (mm_final_pnl / trade_count * 1000.0) if trade_count > 0 else 0.0

    avg_mm_inv = float(np.mean([v for (_, v) in mm_inv_times])) if mm_inv_times else float(mm.inventory)

    # price volatility: variance of midprice returns (log returns)
    vol = 0.0
    if len(midprice_series) >= 2:
        mp = np.array(midprice_series)
        # use simple returns
        rets = np.diff(mp) / mp[:-1]
        vol = float(np.var(rets, ddof=1))

    # depth and MM contribution
    mid_for_depth = (book.best_bid().price + book.best_ask().price)/2.0 if (book.best_bid() and book.best_ask()) else init_price
    depth = compute_depth_profile(book, mid_for_depth, tick=params.get("tick", 0.01), levels=(1,2,3,4,5))
    mm_depth_frac = (depth["mm_depth_within_5_ticks"] / depth["depth_within_5_ticks"]) if depth["depth_within_5_ticks"] > 0 else 0.0

    # realized spread average (approx)
    mm_realized = float(np.mean(mm_realized_spreads)) if mm_realized_spreads else 0.0

    # Compile results dictionary
    results = {
        "sim_duration": sim_duration,
        "seed": seed,
        "avg_spread": avg_spread,
        "median_exec_time": median_exec,
        "mean_exec_time": mean_exec,
        "p99_exec_time": p99_exec,
        "resting_exec_count": resting_exec_count,
        "market_exec_count": market_exec_count,
        "total_executions": total_exec,
        "trade_count": trade_count,
        "trades_per_hour": trades_per_hour,
        "execs_per_hour": execs_per_hour,
        "mm_final_pnl": mm_final_pnl,
        "mm_final_pnl_per_hour": mm_final_pnl_per_hour,
        "mm_final_pnl_per_1k_trades": mm_final_pnl_per_1k_trades,
        "mm_final_inventory": float(mm.inventory),
        "avg_mm_inventory": avg_mm_inv,
        "rest_exec_skew": 0.0,
        "rest_exec_kurtosis": 0.0,
        "price_volatility": vol,
        "depth_within_5_ticks": depth["depth_within_5_ticks"],
        "mm_depth_within_5_ticks": depth["mm_depth_within_5_ticks"],
        "mm_depth_fraction": mm_depth_frac,
        "mm_realized_spread": mm_realized,
        "runtime_sec": runtime
    }

    # also include per-level depths
    for L, v in depth["bid_levels"].items():
        results[f"bid_level_{L}"] = v
    for L, v in depth["ask_levels"].items():
        results[f"ask_level_{L}"] = v

    # raw traces optionally returned (not written to main CSV to keep compact)
    results["_raw_trades"] = trade_records
    results["_spread_samples"] = spread_samples
    results["_mm_pnl_times"] = mm_pnl_times
    results["_mm_inv_times"] = mm_inv_times
    results["_resting_exec_times"] = resting_exec_times

    return results

# ---------------------------
# Experiment runner: full factorial, CRN
# ---------------------------
def run_experiment_suite(sim_duration=23400.0, replications=5, output_file="simulation_results_full.csv", verbose=False):
    """
    Build scenarios (3 factors -> 8 scenarios) and run experiments with CRN.
    """
    # factors and levels (user-specified design)
    FACTORS = {
        "order_size_mean": [1.0, 3.0, 5.0, 7.0, 10.0],
        "limit_price_decay_rate": [0.05, 0.1, 0.2, 0.3, 0.5],
        "mm_base_spread": [0.1, 0.25, 0.5, 1.0, 2.0],
    }

    # build scenarios dictionary
    scenarios = {}
    for i, combo in enumerate(itertools.product(*FACTORS.values())):
        sid = f"S{i+1}"
        scenarios[sid] = {
            "order_size_mean": combo[0],
            "limit_price_decay_rate": combo[1],
            "mm_base_spread": combo[2],
            # default network of other params
            "lam_limit_buy": 0.6,
            "lam_limit_sell": 0.6,
            "lam_mkt_buy": 0.2,
            "lam_mkt_sell": 0.2,
            "lam_cancel": 0.1,
            "mm_quote_interval": 10.0,
            "mm_skew_coef": 0.05,
            "initial_price": 100.0,
            "tick": 0.01,
            "sim_duration": sim_duration,
            "order_size_mean": combo[0],
            "limit_price_decay_rate": combo[1],
            "mm_base_spread": combo[2]
        }

    # prepare output folders
    os.makedirs("results", exist_ok=True)
    all_raw_rows = []

    base_crn_seed = 1000000  # stable base

    for rep in range(1, replications + 1):
        # replication seed (CRN base)
        replication_seed = base_crn_seed + rep
        print(f"\n--- Replication {rep}/{replications} (CRN seed {replication_seed}) ---")
        for sid, sparams in scenarios.items():
            print(f" Running {sid} ...", end="", flush=True)
            # assemble run params and pass same replication_seed to ensure CRN across scenarios
            run_params = sparams.copy()
            run_params["sim_duration"] = sim_duration
            run_seed = replication_seed  # CRN: same seed across scenarios for this replication
            try:
                res = run_single_simulation(run_params, seed=run_seed, verbose=verbose)
                # attach scenario metadata
                row = {
                    "Scenario": sid,
                    "Replication": rep,
                    "seed": run_seed,
                    "order_size_mean": sparams["order_size_mean"],
                    "limit_price_decay_rate": sparams["limit_price_decay_rate"],
                    "mm_base_spread": sparams["mm_base_spread"],
                }
                # add all measured metrics
                for k, v in res.items():
                    if k.startswith("_"):
                        continue
                    row[k] = v
                all_raw_rows.append(row)

                # optional: save per-run file
                scenario_dir = os.path.join("results", sid)
                os.makedirs(scenario_dir, exist_ok=True)
                perrun_df = pd.DataFrame([row])
                perrun_df.to_csv(os.path.join(scenario_dir, f"run_{rep}.csv"), index=False)

                print(" done.")
            except Exception as e:
                print(f" ERROR: {e}")
                # append an error row
                all_raw_rows.append({
                    "Scenario": sid,
                    "Replication": rep,
                    "seed": run_seed,
                    "error": str(e)
                })

    # aggregate: compute mean + CI per scenario
    raw_df = pd.DataFrame(all_raw_rows)
    agg_rows = []
    for sid in sorted(raw_df["Scenario"].unique()):
        scen_df = raw_df[(raw_df["Scenario"] == sid) & (raw_df.get("error").isna() if "error" in raw_df.columns else True)]
        if scen_df.empty:
            print(f"Warning: no valid data for {sid}")
            continue
        agg = {
            "Scenario": sid,
            "Replications": replications,
            "order_size_mean": scen_df["order_size_mean"].iloc[0],
            "limit_price_decay_rate": scen_df["limit_price_decay_rate"].iloc[0],
            "mm_base_spread": scen_df["mm_base_spread"].iloc[0],
        }
        metric_cols = [c for c in scen_df.columns if c not in ("Scenario","Replication","seed","order_size_mean","limit_price_decay_rate","mm_base_spread","error")]
        for col in metric_cols:
            data = scen_df[col].dropna().astype(float).tolist()
            mean, ci = safe_mean_ci(data)
            agg[f"{col}_Mean"] = mean
            agg[f"{col}_CI_95"] = ci
        agg_rows.append(agg)

    agg_df = pd.DataFrame(agg_rows)

    # final combined output
    out_df = pd.concat([raw_df.reset_index(drop=True), agg_df.reset_index(drop=True)], ignore_index=True, sort=False)
    out_df.to_csv(output_file, index=False)
    print(f"\nSaved full results to {output_file}")
    return raw_df, agg_df

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    RAW, AGG = run_experiment_suite(sim_duration=23400.0, replications=5, output_file="results/simulation_results_full.csv", verbose=False)

# REPORT OUTLINE
------------------------------------------------------------
1. Introduction
------------------------------------------------------------
- This project implements a discrete‑event simulation (DES) of a 
  single‑asset limit order book (LOB).
- Orders arrive via independent Poisson processes:
    • limit buy/sell
    • market buy/sell
    • cancellations
- A market maker provides liquidity using an inventory‑based quoting rule.
- Matching uses price–time priority implemented with heaps.

Objectives:
- Evaluate market quality (spread, execution time, volatility).
- Analyze market‑maker inventory risk and P&L.
- Study system behavior under stress (high order volume, high cancellations).

------------------------------------------------------------
2. Model Overview
------------------------------------------------------------
- Continuous‑time DES with exponential inter‑event times.
- Two order types: limit and market.
- Noise traders generate all non‑MM order flow.
- Market maker quotes after every event.
- Matching occurs when best_bid >= best_ask.

------------------------------------------------------------
3. Order Representation
------------------------------------------------------------
Order object contains:
- order_id
- side (“buy” / “sell”)
- price
- quantity
- time_created
- trader_id (“MM” or “noise”)
- active flag

------------------------------------------------------------
4. Queueing Structures (LOB)
------------------------------------------------------------
- Bid side: max‑heap using key (-price, time, order)
- Ask side: min‑heap using key (price, time, order)
- Implements price–time priority.
- Lazy deletion: inactive orders remain but are skipped.

------------------------------------------------------------
5. Matching Logic (OrderBook)
------------------------------------------------------------
- submit_limit(): inserts limit order into correct heap.
- submit_market(): executes 1‑unit trades vs best opposite quote.
- try_match_limit_crossing(): executes trades while book is crossed.
- cancel_best(): removes top order on chosen side.

Trade record includes:
- price
- quantity
- time
- buy_order
- sell_order

------------------------------------------------------------
6. Market Maker Logic
------------------------------------------------------------
- Computes midprice = (best_bid + best_ask)/2
- Computes skew = -skew_coef * inventory
- Quotes 1‑unit:
    bid = mid - base_spread + skew
    ask = mid + base_spread + skew
- Tracks:
    inventory
    cash
    last_trade_price
    P&L = cash + inventory * last_trade_price
- Updated whenever its order appears in a trade.

------------------------------------------------------------
7. Simulation Structure (simulation.py)
------------------------------------------------------------
Event rates:
- lam_limit_buy
- lam_limit_sell
- lam_mkt_buy
- lam_mkt_sell
- lam_cancel

Main loop:
1. Sample Δt ~ Exponential(1/total_rate)
2. Choose event type with probability proportional to its rate.
3. Execute one event:
    - limit order arrival
    - market order arrival
    - cancellation
4. Market maker quotes new bid/ask
5. While book is crossed: match trades
6. Record spread, MM P&L, inventory, execution times

------------------------------------------------------------
8. Metrics Collected
------------------------------------------------------------
Market Quality:
- Bid‑ask spreads over time
- Average spread

Execution Efficiency:
- Execution times for filled noise limit orders
- Median execution time

Market Maker Performance:
- Inventory path
- P&L path
- Final P&L and inventory

Optional:
- Midprice time series → volatility
- Market depth (active queue sizes)

------------------------------------------------------------
9. Experimental Design
------------------------------------------------------------
Baseline parameters:
- initial_price
- arrival rates
- cancellation rate
- market maker parameters (base spread, skew coefficient, bandwidth)

Stress scenarios:
- Increase arrival rates
- Increase cancellation rate
- Modify MM quoting behavior (spread, skew, bandwidth)

Replications:
- Multiple independent runs with different RNG seeds.

------------------------------------------------------------
10. Results (Template)
------------------------------------------------------------
- Plot average spreads under different parameter sets.
- Histogram/boxplot of execution times.
- MM inventory and P&L time series.
- Compare performance under stress vs baseline.
- Discuss effects of MM parameters on risk and profitability.

------------------------------------------------------------
11. Conclusion
------------------------------------------------------------
- Summary of market quality findings.
- Findings on execution delays.
- MM profitability vs. inventory risk.
- Impact of stress conditions.
- Limitations and possible extensions.
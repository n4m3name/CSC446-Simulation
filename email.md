Subject: Project Proposal — Simulating a Limit Order Book and Market Dynamics

Dear [Professor’s Name],

We would like to propose a project titled “Simulating a Limit Order Book (LOB) and Market Dynamics.”

The objective is to develop a discrete-event simulation of a financial market’s limit order book—similar to those used in the TSX or NASDAQ—to study how random order arrivals, cancellations, and executions affect liquidity, volatility, and market-maker performance.

The system will model buy/sell limit and market orders as stochastic events, using SimPy in Python. Each order type (limit, market, cancellation) will trigger queueing and matching processes governed by price–time priority. We will track market metrics such as bid-ask spreads, execution time distributions, and market-maker profit/loss across multiple replications.

Our analysis will focus on:

The effects of arrival rates, cancellation rates, and market-maker strategies on execution efficiency and liquidity.

Trade-offs between profit, inventory risk, and volatility.

Sensitivity of the system to market stress (e.g., high-order-volume conditions).

If time allows, we plan to extend the simulation to include learning-based market makers, self-exciting order flows, and price impact modeling.

Please let us know if this project aligns with the course scope or if you’d like us to adjust the focus.

Best regards,
[Your Full Name(s)]
[Course Code / Section, e.g., SENG 404]
University of Victoria
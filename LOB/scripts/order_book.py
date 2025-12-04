# order_book.py
"""
Order book for a simple limit order book simulation.

- Two priority queues:
    * Bid side: max-heap (highest price, then earliest time)
    * Ask side: min-heap (lowest price, then earliest time)
- Orders are stored as Order objects (see order.py).
- Matching rule:
    Trade when best_bid.price >= best_ask.price (price–time priority).
"""

from bid import BidHeap
from ask import AskHeap
from order import Order


class OrderBook:
    def __init__(self):
        self.bids = BidHeap()
        self.asks = AskHeap()
        self.next_order_id = 0
        self.last_trade_price = None

    # ----------------- basic helpers -----------------

    def _new_id(self) -> int:
        oid = self.next_order_id
        self.next_order_id += 1
        return oid

    def best_bid(self):
        """Return the best bid Order (or None if no bids)."""
        return self.bids.best()

    def best_ask(self):
        """Return the best ask Order (or None if no asks)."""
        return self.asks.best()

    # ----------------- order submission -----------------

    def submit_limit(self, trader_id, side, price, quantity, time_created):
        """
        Insert a limit order into the book.

        trader_id   : identifier for who placed the order ("MM", "noise", etc.)
        side        : "buy" or "sell"
        price       : limit price
        quantity    : initial quantity (integer)
        time_created: simulation time

        Returns the created Order object.
        """
        order = Order(
            order_id=self._new_id(),
            side=side,
            price=price,
            quantity=quantity,
            time_created=time_created,
            trader_id=trader_id,
        )

        if side == "buy":
            self.bids.add(order)
        else:
            self.asks.add(order)

        return order

    def submit_market(self, trader_id, side, quantity, time_created):
        """
        Execute a market order as a sequence of 1-unit trades against
        the opposite side of the book.

        Returns a list of trade dicts:
            {
                "price": float,
                "quantity": int,
                "time": float,
                "buy_order": Order or None,
                "sell_order": Order or None,
            }
        """
        trades = []
        remaining = quantity

        while remaining > 0:
            trade = self._execute_one_market_unit(
                trader_id=trader_id,
                side=side,
                time_created=time_created,
            )
            if trade is None:
                break
            trades.append(trade)
            remaining -= 1

        return trades

    def _execute_one_market_unit(self, trader_id, side, time_created):
        """
        Execute a single unit of a market order.
        Returns a single trade dict or None if no liquidity.
        """
        if side == "buy":
            best_ask = self.best_ask()
            if not best_ask:
                return None

            trade_price = best_ask.price
            best_ask.fill(1)
            self.last_trade_price = trade_price

            return {
                "price": trade_price,
                "quantity": 1,
                "time": time_created,
                "buy_order": None,         # market taker not stored as limit order
                "sell_order": best_ask,
            }

        else:  # side == "sell"
            best_bid = self.best_bid()
            if not best_bid:
                return None

            trade_price = best_bid.price
            best_bid.fill(1)
            self.last_trade_price = trade_price

            return {
                "price": trade_price,
                "quantity": 1,
                "time": time_created,
                "buy_order": best_bid,
                "sell_order": None,        # market taker
            }

    # ----------------- limit-order matching -----------------

    def try_match_limit_crossing(self, current_time):
        """
        If the book is crossed (best_bid.price >= best_ask.price),
        execute ONE unit trade between best bid and best ask.

        Returns:
            trade dict as above, or None if no trade.
        """
        best_bid = self.best_bid()
        best_ask = self.best_ask()

        if not best_bid or not best_ask:
            return None

        # price compatibility condition
        if best_bid.price < best_ask.price:
            return None

        trade_price = best_ask.price  # common convention: trade at resting ask
        quantity = 1

        best_bid.fill(quantity)
        best_ask.fill(quantity)
        self.last_trade_price = trade_price

        trade_time = current_time

        return {
            "price": trade_price,
            "quantity": quantity,
            "time": trade_time,
            "buy_order": best_bid,
            "sell_order": best_ask,
        }

    # ----------------- simple cancellation -----------------

    def cancel_best(self, side: str):
        """
        Simple cancellation rule: cancel the current best order on the given side.
        This is enough to model cancellations as 'abandonment' in the queues.
        """
        if side == "buy":
            order = self.best_bid()
        else:
            order = self.best_ask()

        if order is not None:
            order.cancel()

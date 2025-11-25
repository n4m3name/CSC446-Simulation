# Market Maker Logic

class MarketMaker:
    def __init__(self, book, mm_id="MM", base_spread=1.0, skew_coef=0.1):
        """
        book: reference to the order book
        base_spread: half-spread around midprice (in ticks)
        skew_coef: how strongly inventory affects quoting
        """
        self.book = book
        self.mm_id = mm_id
        self.base_spread = base_spread
        self.skew_coef = skew_coef

        self.inventory = 0
        self.cash = 0
        self.last_trade_price = None

    # ------------------------------------------------------
    # 1. Compute midprice
    # ------------------------------------------------------
    def midprice(self):
        best_bid = self.book.best_bid()
        best_ask = self.book.best_ask()
        if not best_bid or not best_ask:
            return None
        return 0.5 * (best_bid.price + best_ask.price)

    # ------------------------------------------------------
    # 2. Compute skew based on inventory
    # ------------------------------------------------------
    def inventory_skew(self):
        return -self.skew_coef * self.inventory
        # Positive inventory → shift prices DOWN (you want to sell)
        # Negative inventory → shift prices UP (you want to buy)

    # ------------------------------------------------------
    # 3. Post bid/ask quotes
    # ------------------------------------------------------
    def quote(self, current_time):
        mid = self.midprice()
        if mid is None:
            return  # not enough liquidity yet

        skew = self.inventory_skew()

        bid_price = mid - self.base_spread + skew
        ask_price = mid + self.base_spread + skew

        # Post 1‑unit bid & ask orders
        self.book.submit_limit(
            trader_id=self.mm_id,
            side="buy",
            price=bid_price,
            quantity=1,
            time_created=current_time
        )

        self.book.submit_limit(
            trader_id=self.mm_id,
            side="sell",
            price=ask_price,
            quantity=1,
            time_created=current_time
        )

    # ------------------------------------------------------
    # 4. Update inventory & cash when trades occur
    # ------------------------------------------------------
    def on_trade(self, trade_price, side, quantity):
        """
        side: 'buy' means the MM bought (inventory increases)
              'sell' means the MM sold (inventory decreases)
        """
        self.last_trade_price = trade_price

        if side == "buy":
            self.inventory += quantity
            self.cash -= trade_price * quantity
        else:
            self.inventory -= quantity
            self.cash += trade_price * quantity

    # ------------------------------------------------------
    # 5. Compute profit & loss
    # ------------------------------------------------------
    def pnl(self):
        if self.last_trade_price is None:
            return self.cash
        return self.cash + self.inventory * self.last_trade_price

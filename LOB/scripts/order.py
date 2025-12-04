# order.py

class Order:
    """
    Simple order object for a limit order book simulation.

    Attributes:
        order_id     : unique integer ID
        side         : "buy" or "sell"
        price        : limit price (float). For market orders, you typically
                       don't create an Order; you just hit the book directly.
        quantity     : remaining quantity (int)
        time_created : simulation timestamp (float)
        trader_id    : identifier for who placed the order (e.g. "MM", "noise")
        active       : whether order is still valid in the book
    """

    def __init__(self, order_id, side, price, quantity, time_created, trader_id="noise"):
        self.order_id = order_id
        self.side = side                  # "buy" or "sell"
        self.price = price                # float
        self.quantity = quantity          # remaining quantity
        self.time_created = time_created  # arrival time in the sim
        self.trader_id = trader_id        # "MM" for market maker, "noise" for others
        self.active = True                # False when fully filled or cancelled

    def fill(self, qty):
        """
        Reduce order quantity by qty. Mark inactive if fully filled.
        """
        if not self.active:
            return

        self.quantity -= qty
        if self.quantity <= 0:
            self.quantity = 0
            self.active = False

    def cancel(self):
        """
        Mark this order as cancelled (inactive) and zero out quantity.
        """
        self.active = False
        self.quantity = 0

    def __repr__(self):
        return (
            f"Order(id={self.order_id}, side={self.side}, "
            f"price={self.price}, qty={self.quantity}, "
            f"time={self.time_created}, trader_id={self.trader_id}, "
            f"active={self.active})"
        )

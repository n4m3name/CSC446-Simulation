# Order Object
class Order:
    """
    Simple order object for a limit order book simulation.

    Attributes:
        order_id      : unique integer ID
        side          : "buy" or "sell"
        price         : limit price (None for market orders)
        quantity      : number of units available
        time_created  : simulation timestamp (float)
        active        : whether order is still valid in the book
    """

    def __init__(self, order_id, side, price, quantity, time_created):
        self.order_id = order_id
        self.side = side            # "buy" or "sell"
        self.price = price          # float, None for market orders
        self.quantity = quantity    # remaining quantity
        self.time_created = time_created
        self.active = True          # becomes False when fully filled/cancelled

    def fill(self, qty):
        """
        Reduce order quantity by qty. Mark inactive if fully filled.
        """
        self.quantity -= qty
        if self.quantity <= 0:
            self.quantity = 0
            self.active = False

    def cancel(self):
        """
        Mark this order as cancelled.
        """
        self.active = False
        self.quantity = 0

    def __repr__(self):
        return (f"Order(id={self.order_id}, side={self.side}, "
                f"price={self.price}, qty={self.quantity}, "
                f"time={self.time_created}, active={self.active})")

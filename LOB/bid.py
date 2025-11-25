# Max heap (-price, time, order)
import heapq

class BidHeap:
    def __init__(self):
        self.heap = []   # items: (-price, time_created, order_obj)

    def add(self, order):
        # negate the price to turn min-heap → max-heap
        key = (-order.price, order.time_created, order)
        heapq.heappush(self.heap, key)

    def best(self):
        # Skip inactive/cancelled orders
        while self.heap and not self.heap[0][2].active:
            heapq.heappop(self.heap)
        return self.heap[0][2] if self.heap else None

    def pop_best(self):
        # Remove and return the best active order
        while self.heap:
            _, _, order = heapq.heappop(self.heap)
            if order.active:
                return order
        return None

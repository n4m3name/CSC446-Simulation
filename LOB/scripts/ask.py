# Min heap with key (price, time, order)

import heapq

class AskHeap:
    def __init__(self):
        self.heap = []   # items: (price, time_created, order_id, order_obj)

    def add(self, order):
        # Include order_id as a deterministic tiebreaker to avoid comparing Order objects
        key = (order.price, order.time_created, order.order_id, order)
        heapq.heappush(self.heap, key)

    def best(self):
        # Skip inactive/cancelled orders
        while self.heap and not self.heap[0][3].active:
            heapq.heappop(self.heap)
        return self.heap[0][3] if self.heap else None

    def pop_best(self):
        # Remove and return the best active order
        while self.heap:
            _, _, _, order = heapq.heappop(self.heap)
            if order.active:
                return order
        return None
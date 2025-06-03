from queue import PriorityQueue
import heapq

class ComparableItem:
    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        return self.value < other.value

class UncomparableItem:
    def __init__(self, value):
        self.value = value

class UncomparableTask:
    def __init__(self, priority, task_name):
        self.priority = priority
        self.task_name = task_name

class ComparableTask:
    def __init__(self, priority, task_name):
        self.priority = priority
        self.task_name = task_name

    def __lt__(self, other):
        return self.priority < other.priority


# ===================== OK =====================

heap = []
items = [ComparableItem(3)]
for item in items:
    heapq.heappush(heap, item)


# ===================== OK =====================
priority_queue = PriorityQueue()
tasks = [ComparableTask(3, 'Task 1'), ComparableTask(1, 'Task 2'), ComparableTask(2, 'Task 3')]
for task in tasks:
    priority_queue.put(task)


# ===================== VIOLATION =====================
heap = []
items = [UncomparableItem('dog')]
for item in items:
    try:
        heapq.heappush(heap, item) # DyLin warn
    except TypeError:
        pass  # Expecting TypeError due to lack of '<' implementation


# ===================== VIOLATION =====================
priority_queue = PriorityQueue()
tasks = [UncomparableTask(3, 'Task 1')]
for task in tasks:
    try:
        priority_queue.put(task) # DyLin warn
    except TypeError:
        pass  # Expecting TypeError due to lack of '<' implementations
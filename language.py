#!/usr/bin/env python3
"""
Python Syntax Cheatsheet for LeetCode
Quick reference for native structures and useful libraries
"""

import heapq
import bisect
import math
import sys
from collections import defaultdict, deque, Counter, OrderedDict, namedtuple, ChainMap
from functools import lru_cache, reduce, partial, wraps
from itertools import permutations, combinations, product, chain, cycle, groupby, accumulate
from typing import List, Dict, Set, Tuple, Optional, Union, Any
import re
import string

# == BASIC SYNTAX ==
# Multiple assignment
a, b, c = 1, 2, 3
x = y = z = 0
a, b = b, a  # swap

# Conditional assignment
result = "positive" if x > 0 else "non-positive"
value = data.get("key") or "default"

# Chained comparisons
if 0 <= x <= 100:
    pass

# Boolean operations
is_valid = all([True, True, False])  # False
has_any = any([False, False, True])  # True

# == STRING OPERATIONS ==
s = "Hello World"
s[0]                    # 'H'
s[-1]                   # 'd'
s[1:5]                  # 'ello'
s[:5]                   # 'Hello'
s[6:]                   # 'World'
s[::-1]                 # 'dlroW olleH' (reverse)
s[::2]                  # 'HloWrd' (every 2nd char)

# String methods
s.lower()               # 'hello world'
s.upper()               # 'HELLO WORLD'
s.strip()               # removes whitespace
s.split()               # ['Hello', 'World']
s.split('l')            # ['He', '', 'o Wor', 'd']
' '.join(['a', 'b'])    # 'a b'
s.replace('World', 'Python')  # 'Hello Python'
s.find('World')         # 6 (-1 if not found)
s.count('l')            # 3
s.startswith('Hello')   # True
s.endswith('World')     # True
s.isdigit()             # False
s.isalpha()             # False
s.isalnum()             # False

# String formatting
f"Hello {name}"
"Hello {}".format(name)
"Hello {name}".format(name="World")

# == LIST OPERATIONS ==
arr = [1, 2, 3, 4, 5]
arr[0]                  # 1
arr[-1]                 # 5
arr[1:4]                # [2, 3, 4]
arr[:3]                 # [1, 2, 3]
arr[2:]                 # [3, 4, 5]
arr[::-1]               # [5, 4, 3, 2, 1] (reverse)
arr[::2]                # [1, 3, 5] (every 2nd element)

# List methods
arr.append(6)           # [1, 2, 3, 4, 5, 6]
arr.insert(2, 99)       # [1, 2, 99, 3, 4, 5, 6]
arr.remove(99)          # [1, 2, 3, 4, 5, 6] (removes first occurrence)
popped = arr.pop()      # 6, arr = [1, 2, 3, 4, 5]
popped = arr.pop(0)     # 1, arr = [2, 3, 4, 5]
arr.extend([6, 7])      # [2, 3, 4, 5, 6, 7]
arr.sort()              # sorts in place
arr.sort(reverse=True)  # descending sort
arr.reverse()           # reverses in place
new_arr = sorted(arr)   # returns new sorted list
idx = arr.index(3)      # 1 (first occurrence)
count = arr.count(3)    # 1
arr.clear()             # []

# List comprehensions
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
matrix = [[i*j for j in range(3)] for i in range(3)]
flattened = [item for sublist in matrix for item in sublist]

# == DICT OPERATIONS ==
d = {"a": 1, "b": 2, "c": 3}
d["a"]                  # 1
d.get("d", 0)           # 0 (default value)
d["d"] = 4              # {"a": 1, "b": 2, "c": 3, "d": 4}
del d["a"]              # {"b": 2, "c": 3, "d": 4}
popped = d.pop("b", None)  # 2, d = {"c": 3, "d": 4}
key, val = d.popitem()  # removes and returns arbitrary item
d.update({"x": 10})     # {"c": 3, "d": 4, "x": 10}
d.clear()               # {}

# Dict methods
keys = list(d.keys())
values = list(d.values())
items = list(d.items())
"a" in d                # membership test

# Dict comprehensions
squared = {x: x**2 for x in range(5)}
filtered = {k: v for k, v in d.items() if v > 2}

# == SET OPERATIONS ==
s1 = {1, 2, 3}
s2 = {3, 4, 5}
s1.add(4)               # {1, 2, 3, 4}
s1.remove(2)            # {1, 3, 4} (KeyError if not found)
s1.discard(5)           # {1, 3, 4} (no error if not found)
s1.update([6, 7])       # {1, 3, 4, 6, 7}
s1.clear()              # set()

# Set operations
union = s1 | s2         # {1, 3, 4, 5, 6, 7}
intersection = s1 & s2  # {3, 4}
difference = s1 - s2    # {1, 6, 7}
sym_diff = s1 ^ s2      # {1, 5, 6, 7}
s1.issubset(s2)         # False
s1.issuperset(s2)       # False

# Set comprehensions
evens = {x for x in range(10) if x % 2 == 0}

# == TUPLE OPERATIONS ==
t = (1, 2, 3)
t[0]                    # 1
t.count(1)              # 1
t.index(2)              # 1
a, b, c = t             # unpacking

# Named tuples
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
p.x, p.y                # 1, 2

# == COLLECTIONS MODULE ==
from collections import defaultdict, deque, Counter, OrderedDict, ChainMap

# defaultdict
dd = defaultdict(int)   # default value is 0
dd = defaultdict(list)  # default value is []
dd = defaultdict(set)   # default value is set()
dd["key"] += 1          # no KeyError, starts at 0

# Counter
counter = Counter("hello")          # Counter({'l': 2, 'h': 1, 'e': 1, 'o': 1})
counter = Counter([1, 2, 2, 3, 3, 3])  # Counter({3: 3, 2: 2, 1: 1})
counter.most_common(2)              # [(3, 3), (2, 2)]
counter.update([1, 1, 1])           # add counts
counter.subtract([1, 2])            # subtract counts
counter["missing"]                  # 0 (doesn't raise KeyError)

# deque (double-ended queue)
dq = deque([1, 2, 3])
dq.appendleft(0)        # deque([0, 1, 2, 3])
dq.append(4)            # deque([0, 1, 2, 3, 4])
dq.popleft()            # 0
dq.pop()                # 4
dq.rotate(1)            # deque([3, 1, 2])
dq.extend([4, 5])       # deque([3, 1, 2, 4, 5])
dq.extendleft([0, -1])  # deque([-1, 0, 3, 1, 2, 4, 5])

# OrderedDict (maintains insertion order - less needed in Python 3.7+)
od = OrderedDict([('a', 1), ('b', 2)])
od.move_to_end('a')     # moves 'a' to end
od.popitem(last=False)  # pops from beginning

# ChainMap (combines multiple dicts)
dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 3, 'c': 4}
chain_map = ChainMap(dict1, dict2)
chain_map['b']          # 2 (from first dict)

# == HEAPQ (PRIORITY QUEUE) ==
import heapq
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 4)
min_val = heapq.heappop(heap)    # 1
min_val = heap[0]                # peek (3)
heapq.heapify([3, 1, 4])        # convert list to heap in-place
largest = heapq.nlargest(2, [1, 3, 4, 2])     # [4, 3]
smallest = heapq.nsmallest(2, [1, 3, 4, 2])   # [1, 2]

# Max heap (negate values)
max_heap = [-x for x in [3, 1, 4]]
heapq.heapify(max_heap)
max_val = -heapq.heappop(max_heap)  # 4

# == BISECT (BINARY SEARCH) ==
import bisect
arr = [1, 3, 4, 7, 9]
idx = bisect.bisect_left(arr, 4)     # 2 (leftmost position)
idx = bisect.bisect_right(arr, 4)    # 3 (rightmost position)
bisect.insort(arr, 5)                # [1, 3, 4, 5, 7, 9]

# == ITERTOOLS ==
from itertools import permutations, combinations, product, chain, cycle, groupby, accumulate

# Permutations and combinations
list(permutations([1, 2, 3]))       # all permutations
list(permutations([1, 2, 3], 2))    # 2-element permutations
list(combinations([1, 2, 3], 2))    # 2-element combinations
list(combinations_with_replacement([1, 2], 2))  # [(1, 1), (1, 2), (2, 2)]

# Cartesian product
list(product([1, 2], ['a', 'b']))   # [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]
list(product([1, 2], repeat=2))     # [(1, 1), (1, 2), (2, 1), (2, 2)]

# Chain and cycle
list(chain([1, 2], [3, 4]))         # [1, 2, 3, 4]
list(chain.from_iterable([[1, 2], [3, 4]]))  # [1, 2, 3, 4]
# cycle creates infinite iterator: cycle([1, 2, 3]) -> 1, 2, 3, 1, 2, 3, ...

# Groupby (requires sorted input)
data = [('a', 1), ('a', 2), ('b', 3), ('b', 4)]
for key, group in groupby(data, key=lambda x: x[0]):
    print(key, list(group))          # a [('a', 1), ('a', 2)], b [('b', 3), ('b', 4)]

# Accumulate (cumulative operations)
list(accumulate([1, 2, 3, 4]))      # [1, 3, 6, 10] (cumsum)
list(accumulate([1, 2, 3, 4], max)) # [1, 2, 3, 4] (running max)

# == FUNCTOOLS ==
from functools import lru_cache, reduce, partial, wraps

# LRU Cache (memoization)
@lru_cache(maxsize=None)
def fibonacci(n):
    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)

# Reduce
result = reduce(lambda x, y: x + y, [1, 2, 3, 4])  # 10
result = reduce(max, [1, 5, 2, 9, 3])              # 9

# Partial application
def multiply(x, y):
    return x * y
double = partial(multiply, 2)
double(5)                            # 10

# == MATH MODULE ==
import math
math.ceil(4.2)          # 5
math.floor(4.8)         # 4
math.sqrt(16)           # 4.0
math.pow(2, 3)          # 8.0
math.log(8, 2)          # 3.0 (log base 2)
math.log10(100)         # 2.0
math.factorial(5)       # 120
math.gcd(12, 18)        # 6
math.lcm(12, 18)        # 36 (Python 3.9+)
math.isqrt(15)          # 3 (integer square root, Python 3.8+)

# Constants
math.pi                 # 3.14159...
math.e                  # 2.71828...
math.inf                # infinity
math.nan                # not a number

# == REGEX (RE MODULE) ==
import re
pattern = r'\d+'                    # one or more digits
text = "I have 123 apples and 45 oranges"
matches = re.findall(pattern, text) # ['123', '45']
match = re.search(pattern, text)    # first match object
if match:
    print(match.group())            # '123'

# Common patterns
re.findall(r'\w+', text)           # words
re.findall(r'\d+', text)           # digits
re.findall(r'[A-Z]', text)         # uppercase letters
re.sub(r'\d+', 'X', text)          # replace digits with 'X'

# == STRING MODULE ==
import string
string.ascii_lowercase   # 'abcdefghijklmnopqrstuvwxyz'
string.ascii_uppercase   # 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
string.ascii_letters     # lowercase + uppercase
string.digits           # '0123456789'
string.punctuation      # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

# == USEFUL BUILT-IN FUNCTIONS ==
# map, filter, zip
list(map(str, [1, 2, 3]))          # ['1', '2', '3']
list(filter(lambda x: x > 2, [1, 2, 3, 4]))  # [3, 4]
list(zip([1, 2, 3], ['a', 'b', 'c']))        # [(1, 'a'), (2, 'b'), (3, 'c')]

# enumerate
for i, val in enumerate(['a', 'b', 'c']):
    print(i, val)                   # 0 a, 1 b, 2 c

# reversed
list(reversed([1, 2, 3]))          # [3, 2, 1]

# sum with start value
sum([1, 2, 3], 10)                 # 16

# min/max with key
min(['apple', 'pie'], key=len)     # 'pie'
max([1, -5, 3], key=abs)           # -5

# sorted with key
sorted(['apple', 'pie', 'cherry'], key=len)  # ['pie', 'apple', 'cherry']
sorted([1, 2, 3], reverse=True)    # [3, 2, 1]

# == COMMON PATTERNS ==
# Infinity
pos_inf = float('inf')
neg_inf = float('-inf')

# Check if number
try:
    num = int(s)
except ValueError:
    pass

# Safe division
result = a / b if b != 0 else 0

# Default dict pattern without defaultdict
d = {}
d.setdefault(key, []).append(value)

# Frequency counter without Counter
freq = {}
for item in items:
    freq[item] = freq.get(item, 0) + 1

# Min/max with default
min_val = min(arr) if arr else float('inf')
max_val = max(arr) if arr else float('-inf')

# Multiple variable swap
a, b, c = c, a, b

# Check if all/any elements satisfy condition
all(x > 0 for x in [1, 2, 3])     # True
any(x > 5 for x in [1, 2, 3])     # False

# Flatten 2D list
flattened = [item for row in matrix for item in row]

# Count occurrences in nested structure
count = sum(row.count(target) for row in matrix)

# Create 2D array
matrix = [[0] * cols for _ in range(rows)]  # CORRECT
# matrix = [[0] * cols] * rows              # WRONG (same reference)

# == COMMON TRICKS ==
# Remove duplicates while preserving order
def remove_duplicates(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

# Get unique elements
unique = list(dict.fromkeys(lst))   # preserves order
unique = list(set(lst))             # doesn't preserve order

# Rotate list
def rotate_left(arr, k):
    k %= len(arr)
    return arr[k:] + arr[:k]

def rotate_right(arr, k):
    k %= len(arr)
    return arr[-k:] + arr[:-k]

# Safe indexing
def safe_get(arr, idx, default=None):
    return arr[idx] if 0 <= idx < len(arr) else default
    
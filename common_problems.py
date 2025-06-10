#!/usr/bin/env python3
"""
Python Algorithm Patterns for LeetCode
Fundamental algorithms with comments explaining patterns
"""

from collections import deque, defaultdict, Counter
from functools import lru_cache
import heapq

# == TABLE OF CONTENTS ==
# Search: DFS, BFS, Binary Search
# Sorting: Merge Sort, Quick Sort
# Recursion: Basic, Divide & Conquer
# Backtracking: Permutations, Combinations, N-Queens
# DP: Memoization, Tabulation, Classic Problems
# Two Pointers: Sorted Arrays, In-place Modification
# Sliding Window: Fixed/Variable Size
# Graph: Cycle Detection, Topological Sort, Dijkstra
# Tree: LCA, BST Validation
# Union-Find: Connected Components, Cycle Detection
# Greedy: Activity Selection, Intervals
# Trie: Prefix Tree, Word Search
# Heap: Top-K, Priority Queue
# Math: Prefix Sum, Kadane's Algorithm

# == DFS (DEPTH-FIRST SEARCH) ==

def dfs_recursive(graph, node, visited=None):
    """
    DFS Recursive Pattern - explores as far as possible before backtracking
    Time: O(V + E), Space: O(V) for recursion stack
    """
    if visited is None:
        visited = set()
    
    if node in visited:
        return
    
    visited.add(node)
    print(node)  # process node
    
    for neighbor in graph[node]:
        dfs_recursive(graph, neighbor, visited)

def dfs_iterative(graph, start):
    """
    DFS Iterative Pattern - uses stack to simulate recursion
    Better for deep graphs to avoid stack overflow
    """
    visited = set()
    stack = [start]
    
    while stack:
        node = stack.pop()  # LIFO - last in, first out
        
        if node not in visited:
            visited.add(node)
            print(node)  # process node
            
            # Add neighbors to stack (reverse order to maintain left-to-right)
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return visited

def dfs_tree_traversal(root):
    """
    Binary Tree DFS Pattern - inorder, preorder, postorder
    """
    if not root:
        return
    
    # Preorder: root -> left -> right
    print(root.val)
    dfs_tree_traversal(root.left)
    dfs_tree_traversal(root.right)

def dfs_path_finding(graph, start, target, path=None):
    """
    DFS Path Finding Pattern - find path between two nodes
    """
    if path is None:
        path = []
    
    path = path + [start]
    
    if start == target:
        return path
    
    for neighbor in graph[start]:
        if neighbor not in path:  # avoid cycles
            new_path = dfs_path_finding(graph, neighbor, target, path)
            if new_path:
                return new_path
    
    return None

# == BFS (BREADTH-FIRST SEARCH) ==

def bfs_iterative(graph, start):
    """
    BFS Pattern - explores all neighbors before going deeper
    Time: O(V + E), Space: O(V)
    Guarantees shortest path in unweighted graphs
    """
    visited = set([start])
    queue = deque([start])
    
    while queue:
        node = queue.popleft()  # FIFO - first in, first out
        print(node)  # process node
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return visited

def bfs_shortest_path(graph, start, target):
    """
    BFS Shortest Path Pattern - finds shortest unweighted path
    """
    if start == target:
        return [start]
    
    visited = set([start])
    queue = deque([(start, [start])])  # (node, path_to_node)
    
    while queue:
        node, path = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor == target:
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return []  # no path found

def bfs_level_order_tree(root):
    """
    Binary Tree Level Order Pattern - process tree level by level
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result

# == SORTING ALGORITHMS ==

def merge_sort(arr):
    """
    Merge Sort Pattern - divide and conquer, stable sort
    Time: O(n log n), Space: O(n)
    Key insight: merge two sorted arrays efficiently
    """
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer (merge)
    return merge(left, right)

def merge(left, right):
    """
    Merge Pattern - combine two sorted arrays
    Two pointers technique
    """
    result = []
    i = j = 0
    
    # Compare elements from both arrays
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

def quick_sort(arr):
    """
    Quick Sort Pattern - divide and conquer, in-place possible
    Time: O(n log n) average, O(n²) worst, Space: O(log n)
    Key insight: partition around pivot
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    
    # Partition: elements < pivot, = pivot, > pivot
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

def partition(arr, low, high):
    """
    Partition Pattern - rearrange array around pivot
    Used in quick sort and quick select
    """
    pivot = arr[high]
    i = low - 1  # index of smaller element
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# == BINARY SEARCH ==

def binary_search(arr, target):
    """
    Binary Search Pattern - search in sorted array
    Time: O(log n), Space: O(1)
    Key insight: eliminate half the search space each iteration
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # avoid overflow
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1  # search right half
        else:
            right = mid - 1  # search left half
    
    return -1  # not found

def binary_search_leftmost(arr, target):
    """
    Find leftmost occurrence of target
    """
    left, right = 0, len(arr)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left

def binary_search_rightmost(arr, target):
    """
    Find rightmost occurrence of target
    """
    left, right = 0, len(arr)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    
    return left - 1

# == RECURSION PATTERNS ==

def fibonacci_recursive(n):
    """
    Basic Recursion Pattern - solve smaller subproblems
    Time: O(2^n), Space: O(n) for call stack
    """
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

def factorial(n):
    """
    Tail Recursion Pattern
    """
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def power(base, exp):
    """
    Divide and Conquer Recursion Pattern
    Time: O(log exp)
    """
    if exp == 0:
        return 1
    if exp == 1:
        return base
    
    if exp % 2 == 0:
        half = power(base, exp // 2)
        return half * half
    else:
        return base * power(base, exp - 1)

# == BACKTRACKING ==

def permutations(nums):
    """
    Permutation Backtracking Pattern - generate all arrangements
    Time: O(n! * n), Space: O(n)
    Key insight: try each possibility, backtrack if needed
    """
    result = []
    
    def backtrack(path):
        # Base case: complete permutation
        if len(path) == len(nums):
            result.append(path[:])  # copy current path
            return
        
        # Try each unused number
        for num in nums:
            if num not in path:
                path.append(num)      # choose
                backtrack(path)       # explore
                path.pop()            # backtrack (unchoose)
    
    backtrack([])
    return result

def combinations(n, k):
    """
    Combination Backtracking Pattern - choose k from n
    Time: O(C(n,k) * k), Space: O(k)
    """
    result = []
    
    def backtrack(start, path):
        # Base case: have k elements
        if len(path) == k:
            result.append(path[:])
            return
        
        # Try each number from start to n
        for i in range(start, n + 1):
            path.append(i)           # choose
            backtrack(i + 1, path)   # explore (i+1 to avoid duplicates)
            path.pop()               # backtrack
    
    backtrack(1, [])
    return result

def subsets(nums):
    """
    Subset Backtracking Pattern - generate all subsets
    Time: O(2^n * n), Space: O(n)
    """
    result = []
    
    def backtrack(start, path):
        # Every path is a valid subset
        result.append(path[:])
        
        # Try adding each remaining element
        for i in range(start, len(nums)):
            path.append(nums[i])      # choose
            backtrack(i + 1, path)    # explore
            path.pop()                # backtrack
    
    backtrack(0, [])
    return result

def solve_n_queens(n):
    """
    N-Queens Backtracking Pattern - constraint satisfaction
    Key insight: check constraints before exploring further
    """
    result = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    
    def is_safe(row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        
        # Check diagonal (top-left to bottom-right)
        for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
            if board[i][j] == 'Q':
                return False
        
        # Check anti-diagonal (top-right to bottom-left)
        for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
            if board[i][j] == 'Q':
                return False
        
        return True
    
    def backtrack(row):
        if row == n:
            result.append([''.join(row) for row in board])
            return
        
        for col in range(n):
            if is_safe(row, col):
                board[row][col] = 'Q'    # choose
                backtrack(row + 1)       # explore
                board[row][col] = '.'    # backtrack
    
    backtrack(0)
    return result

# == DYNAMIC PROGRAMMING ==

def fibonacci_memo(n, memo=None):
    """
    Memoization DP Pattern - top-down approach
    Time: O(n), Space: O(n)
    Key insight: store results to avoid recomputation
    """
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

@lru_cache(maxsize=None)
def fibonacci_lru(n):
    """
    LRU Cache DP Pattern - automatic memoization
    """
    if n <= 1:
        return n
    return fibonacci_lru(n-1) + fibonacci_lru(n-2)

def fibonacci_tabulation(n):
    """
    Tabulation DP Pattern - bottom-up approach
    Time: O(n), Space: O(n) or O(1) optimized
    Key insight: build solution from smallest subproblems
    """
    if n <= 1:
        return n
    
    # dp[i] = fibonacci number at index i
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

def coin_change(coins, amount):
    """
    Classic DP Pattern - minimum coins to make amount
    Time: O(amount * len(coins)), Space: O(amount)
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # base case: 0 coins needed for amount 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

def longest_increasing_subsequence(arr):
    """
    LIS DP Pattern - longest increasing subsequence
    Time: O(n²), Space: O(n)
    """
    if not arr:
        return 0
    
    # dp[i] = length of LIS ending at index i
    dp = [1] * len(arr)
    
    for i in range(1, len(arr)):
        for j in range(i):
            if arr[j] < arr[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

def edit_distance(word1, word2):
    """
    2D DP Pattern - minimum edit distance
    Time: O(m * n), Space: O(m * n)
    """
    m, n = len(word1), len(word2)
    
    # dp[i][j] = edit distance between word1[:i] and word2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i  # delete all characters
    for j in range(n + 1):
        dp[0][j] = j  # insert all characters
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # no operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],     # delete
                    dp[i][j-1],     # insert
                    dp[i-1][j-1]    # replace
                )
    
    return dp[m][n]

# == TWO POINTERS ==

def two_sum_sorted(arr, target):
    """
    Two Pointers Pattern - find pair that sums to target
    Time: O(n), Space: O(1)
    Works on sorted arrays
    """
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1   # need larger sum
        else:
            right -= 1  # need smaller sum
    
    return []

def remove_duplicates(arr):
    """
    Two Pointers Pattern - modify array in-place
    Time: O(n), Space: O(1)
    """
    if not arr:
        return 0
    
    write_idx = 1  # where to write next unique element
    
    for read_idx in range(1, len(arr)):
        if arr[read_idx] != arr[read_idx - 1]:
            arr[write_idx] = arr[read_idx]
            write_idx += 1
    
    return write_idx

def reverse_string(s):
    """
    Two Pointers Pattern - reverse in-place
    """
    left, right = 0, len(s) - 1
    
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1

# == SLIDING WINDOW ==

def max_sum_subarray_k(arr, k):
    """
    Fixed Size Sliding Window Pattern
    Time: O(n), Space: O(1)
    """
    if len(arr) < k:
        return 0
    
    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide window: remove leftmost, add rightmost
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

def longest_substring_without_repeating(s):
    """
    Variable Size Sliding Window Pattern
    Time: O(n), Space: O(min(m, n)) where m is charset size
    """
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # Shrink window until no duplicates
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length

# == GRAPH ALGORITHMS ==

def has_cycle_directed(graph):
    """
    Cycle Detection in Directed Graph - DFS with colors
    White (0): unvisited, Gray (1): visiting, Black (2): visited
    """
    color = {node: 0 for node in graph}
    
    def dfs(node):
        if color[node] == 1:  # back edge found
            return True
        if color[node] == 2:  # already processed
            return False
        
        color[node] = 1  # mark as visiting
        
        for neighbor in graph[node]:
            if dfs(neighbor):
                return True
        
        color[node] = 2  # mark as visited
        return False
    
    for node in graph:
        if color[node] == 0:
            if dfs(node):
                return True
    
    return False

def topological_sort(graph, in_degree):
    """
    Topological Sort Pattern - Kahn's algorithm
    Time: O(V + E), Space: O(V)
    """
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == len(in_degree) else []

def dijkstra(graph, start):
    """
    Dijkstra's Algorithm Pattern - shortest path with weights
    Time: O((V + E) log V), Space: O(V)
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    heap = [(0, start)]  # (distance, node)
    visited = set()
    
    while heap:
        curr_dist, node = heapq.heappop(heap)
        
        if node in visited:
            continue
        
        visited.add(node)
        
        for neighbor, weight in graph[node]:
            distance = curr_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))
    
    return distances

# == TREE ALGORITHMS ==

def lowest_common_ancestor(root, p, q):
    """
    LCA Pattern - find lowest common ancestor
    Time: O(n), Space: O(h) where h is height
    """
    if not root or root == p or root == q:
        return root
    
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    
    if left and right:
        return root  # p and q in different subtrees
    
    return left or right  # both in same subtree

def validate_bst(root, min_val=float('-inf'), max_val=float('inf')):
    """
    BST Validation Pattern - recursive bounds checking
    """
    if not root:
        return True
    
    if root.val <= min_val or root.val >= max_val:
        return False
    
    return (validate_bst(root.left, min_val, root.val) and
            validate_bst(root.right, root.val, max_val))

# == UNION-FIND (DISJOINT SET UNION) ==

class UnionFind:
    """
    Union-Find Pattern - track connected components
    Time: O(α(n)) amortized for union/find operations
    Key insight: path compression + union by rank for efficiency
    """
    def __init__(self, size):
        self.parent = list(range(size))  # each node is its own parent initially
        self.rank = [0] * size           # rank for union by rank optimization
        self.components = size           # number of connected components
    
    def find(self, x):
        """Find with path compression - makes tree flatter"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]
    
    def union(self, x, y):
        """Union by rank - attach smaller tree to larger tree"""
        root_x, root_y = self.find(x), self.find(y)
        
        if root_x == root_y:
            return False  # already connected
        
        # Union by rank: attach smaller tree to larger tree
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.components -= 1
        return True
    
    def connected(self, x, y):
        """Check if two nodes are in same component"""
        return self.find(x) == self.find(y)

def number_of_islands_unionfind(grid):
    """
    Union-Find application - count connected components
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    uf = UnionFind(rows * cols)
    
    def get_id(r, c):
        return r * cols + c
    
    water_count = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '0':
                water_count += 1
                continue
            
            # Connect to adjacent land cells
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols and 
                    grid[nr][nc] == '1'):
                    uf.union(get_id(r, c), get_id(nr, nc))
    
    # Count unique components minus water cells
    land_components = len(set(uf.find(get_id(r, c)) 
                             for r in range(rows) 
                             for c in range(cols) 
                             if grid[r][c] == '1'))
    
    return land_components

# == GREEDY ALGORITHMS ==

def activity_selection(intervals):
    """
    Activity Selection Pattern - classic greedy problem
    Time: O(n log n), Space: O(1)
    Key insight: always choose activity that ends earliest
    """
    if not intervals:
        return 0
    
    # Sort by end time
    intervals.sort(key=lambda x: x[1])
    
    count = 1
    last_end = intervals[0][1]
    
    for start, end in intervals[1:]:
        if start >= last_end:  # no overlap
            count += 1
            last_end = end
    
    return count

def merge_intervals(intervals):
    """
    Merge Intervals Pattern - greedy merging
    Time: O(n log n), Space: O(1)
    """
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:  # overlap
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    
    return merged

def fractional_knapsack(items, capacity):
    """
    Fractional Knapsack Pattern - greedy by value/weight ratio
    items: [(weight, value), ...]
    """
    # Sort by value/weight ratio (descending)
    items.sort(key=lambda x: x[1] / x[0], reverse=True)
    
    total_value = 0
    for weight, value in items:
        if capacity >= weight:
            total_value += value
            capacity -= weight
        else:
            # Take fraction of remaining item
            total_value += value * (capacity / weight)
            break
    
    return total_value

# == PREFIX SUM ==

def build_prefix_sum(arr):
    """
    Prefix Sum Pattern - precompute cumulative sums
    Time: O(n) build, O(1) query, Space: O(n)
    Key insight: prefix[i] = sum of elements from 0 to i-1
    """
    prefix = [0]
    for num in arr:
        prefix.append(prefix[-1] + num)
    return prefix

def range_sum_query(prefix, left, right):
    """Query sum from index left to right (inclusive)"""
    return prefix[right + 1] - prefix[left]

def subarray_sum_equals_k(nums, k):
    """
    Subarray Sum Pattern - use prefix sum + hashmap
    Time: O(n), Space: O(n)
    Key insight: if prefix[j] - prefix[i] = k, then subarray i+1 to j sums to k
    """
    count = 0
    prefix_sum = 0
    sum_count = {0: 1}  # prefix sum 0 occurs once (empty subarray)
    
    for num in nums:
        prefix_sum += num
        
        # Check if (prefix_sum - k) exists
        if prefix_sum - k in sum_count:
            count += sum_count[prefix_sum - k]
        
        # Add current prefix sum to map
        sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1
    
    return count

# == KADANE'S ALGORITHM ==

def max_subarray_kadane(nums):
    """
    Kadane's Algorithm Pattern - maximum subarray sum
    Time: O(n), Space: O(1)
    Key insight: at each position, either extend previous subarray or start new
    """
    if not nums:
        return 0
    
    max_sum = curr_sum = nums[0]
    
    for num in nums[1:]:
        curr_sum = max(num, curr_sum + num)  # extend or restart
        max_sum = max(max_sum, curr_sum)
    
    return max_sum

def max_subarray_indices(nums):
    """
    Kadane's with indices - return (max_sum, start, end)
    """
    max_sum = curr_sum = nums[0]
    start = end = 0
    temp_start = 0
    
    for i in range(1, len(nums)):
        if curr_sum < 0:
            curr_sum = nums[i]
            temp_start = i
        else:
            curr_sum += nums[i]
        
        if curr_sum > max_sum:
            max_sum = curr_sum
            start = temp_start
            end = i
    
    return max_sum, start, end

def max_product_subarray(nums):
    """
    Modified Kadane's Pattern - maximum product subarray
    Key insight: track both max and min (negative * negative = positive)
    """
    if not nums:
        return 0
    
    max_prod = min_prod = result = nums[0]
    
    for num in nums[1:]:
        # If num is negative, swap max and min
        if num < 0:
            max_prod, min_prod = min_prod, max_prod
        
        max_prod = max(num, max_prod * num)
        min_prod = min(num, min_prod * num)
        
        result = max(result, max_prod)
    
    return result

# == TRIE (PREFIX TREE) ==

class TrieNode:
    """
    Trie Node - each node represents a character
    """
    def __init__(self):
        self.children = {}  # char -> TrieNode
        self.is_end = False  # marks end of word

class Trie:
    """
    Trie Pattern - prefix tree for string operations
    Time: O(m) for insert/search where m is word length
    Space: O(total chars in all words)
    Key insight: shared prefixes save space, enables fast prefix queries
    """
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        """Insert word into trie"""
        node = self.root
        for char in word:
            node = node.children.setdefault(char, TrieNode())
        node.is_end = True
    
    def search(self, word):
        """Search for exact word"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def starts_with(self, prefix):
        """Check if any word starts with prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    
    def find_words_with_prefix(self, prefix):
        """Find all words that start with prefix"""
        node = self.root
        # Navigate to prefix
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # DFS to find all words from this node
        result = []
        self._dfs(node, prefix, result)
        return result
    
    def _dfs(self, node, current_word, result):
        """Helper for finding words with prefix"""
        if node.is_end:
            result.append(current_word)
        
        for char, child in node.children.items():
            self._dfs(child, current_word + char, result)

def word_search_trie(board, words):
    """
    Word Search with Trie Pattern - find words in 2D board
    Time: O(m*n*4^k) where k is max word length
    """
    trie = Trie()
    for word in words:
        trie.insert(word)
    
    result = set()
    rows, cols = len(board), len(board[0])
    
    def dfs(r, c, node, path):
        if node.is_end:
            result.add(path)
        
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            board[r][c] not in node.children):
            return
        
        char = board[r][c]
        board[r][c] = '#'  # mark as visited
        
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            dfs(r + dr, c + dc, node.children[char], path + char)
        
        board[r][c] = char  # backtrack
    
    for r in range(rows):
        for c in range(cols):
            dfs(r, c, trie.root, "")
    
    return list(result)

# == HEAP PATTERNS ==

def k_largest_elements(nums, k):
    """
    Top-K Pattern using heap
    Time: O(n log k), Space: O(k)
    """
    return heapq.nlargest(k, nums)

def k_smallest_elements(nums, k):
    """
    Bottom-K Pattern using heap
    """
    return heapq.nsmallest(k, nums)

def top_k_frequent(nums, k):
    """
    Top-K Frequent Pattern - Counter + heap
    Time: O(n log k), Space: O(n)
    """
    count = Counter(nums)
    # Get k most frequent elements
    return [item for item, freq in heapq.nlargest(k, count.items(), key=lambda x: x[1])]

def kth_largest_in_stream():
    """
    Kth Largest in Stream Pattern - maintain min heap of size k
    """
    class KthLargest:
        def __init__(self, k, nums):
            self.k = k
            self.heap = nums
            heapq.heapify(self.heap)
            # Keep only k largest elements
            while len(self.heap) > k:
                heapq.heappop(self.heap)
        
        def add(self, val):
            heapq.heappush(self.heap, val)
            if len(self.heap) > self.k:
                heapq.heappop(self.heap)
            return self.heap[0]  # kth largest
    
    return KthLargest

def merge_k_sorted_lists(lists):
    """
    Merge K Sorted Lists Pattern - heap with (value, index, list_index)
    Time: O(n log k), Space: O(k)
    """
    heap = []
    
    # Initialize heap with first element from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], 0, i))
    
    result = []
    
    while heap:
        val, idx, list_idx = heapq.heappop(heap)
        result.append(val)
        
        # Add next element from same list
        if idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][idx + 1]
            heapq.heappush(heap, (next_val, idx + 1, list_idx))
    
    return result

def max_heap_simulation():
    """
    Max Heap Pattern - negate values for min heap
    """
    max_heap = []
    
    def push(val):
        heapq.heappush(max_heap, -val)
    
    def pop():
        return -heapq.heappop(max_heap)
    
    def peek():
        return -max_heap[0] if max_heap else None
    
    return push, pop, peek
"""
1. DFS: Use recursion or stack, good for path finding, tree traversal
2. BFS: Use queue, good for shortest path, level-order traversal
3. Two Pointers: Sorted arrays, palindromes, removing duplicates
4. Sliding Window: Subarray/substring problems with size constraints
5. Binary Search: Sorted arrays, search space reduction
6. DP: Optimal substructure, overlapping subproblems
7. Backtracking: Generate all possibilities, constraint satisfaction
8. Greedy: Local optimal choices, activity selection
9. Union-Find: Connected components, cycle detection
10. Topological Sort: Dependency resolution, course scheduling
"""



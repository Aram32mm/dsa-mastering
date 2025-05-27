# Reverse a Doubly Linked List

## Problem Description
Given the head of a doubly linked list, reverse the direction of the list. The doubly linked list has nodes with both next and previous pointers.

For example:
- Input: 1 ⟷ 2 ⟷ 3 ⟷ 4 ⟷ 5
- Output: 5 ⟷ 4 ⟷ 3 ⟷ 2 ⟷ 1

## Approach
1. Handle edge cases: If the list is empty or has only one node, return the head as is.
2. Initialize current pointer to head and prev pointer to None.
3. Iterate through the list:
   - Store the next node before changing any pointers
   - Swap the next and prev pointers of the current node
   - Move prev to current, and current to the next node
4. Return prev as the new head of the reversed list.

## Complexity
- Time: O(n) - We visit each node exactly once, where n is the number of nodes in the list.
- Space: O(1) - We only use a constant amount of extra space regardless of input size.
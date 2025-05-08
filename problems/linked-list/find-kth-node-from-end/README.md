# Find Kth Node From End

## Problem Description
Given a singly linked list and an integer k, find and return the kth node from the end of the list. If the list has fewer than k nodes, return None.

## Approach
Two-pointer technique. The fast pointer is k nodes ahead of the slow pointer, so when fast reaches the end, slow is k nodes from the end.

## Complexity
- Time: O(n)
- Space: O(1)

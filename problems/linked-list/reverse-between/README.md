# Reverse Between

## Problem Description
Given a linked list and two positions 'left' and 'right', reverse the nodes of the list from position 'left' to position 'right', and return the modified list.

The positions are 1-indexed, meaning the first node has position 1, the second node has position 2, and so on.

given 1->2->3->4->5 left=2, right=4, then 1->4->3->2->5.

## Approach
1. Use a dummy head to simplify edge cases (especially when left=1)
2. Find the node at position (left-1), which is just before the reversal begins
3. Perform the reversal for the nodes from position left to right:
   - Use iterative reversal technique with three pointers (prev, curr, next)
   - Keep track of the connections to the rest of the list
4. Reconnect the reversed portion to the original list
5. Return the head of the modified list

## Complexity
- Time: O(n)
- Space: O(1)
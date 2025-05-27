# Swap Nodes in Pairs in a Doubly Linked List

## Problem Description
Given the head of a doubly linked list, swap every two adjacent nodes and return the head of the modified list.

You should solve the problem without modifying the values in the list's nodes (i.e., only the nodes' pointers should be changed).

For example:
- Input: 1 ⟷ 2 ⟷ 3 ⟷ 4
- Output: 2 ⟷ 1 ⟷ 4 ⟷ 3

## Approach
1. Handle edge cases:
   - If the list is empty or has only one node, return the head as is.
2. Create a dummy node to handle edge cases and link it to the head.
3. Initialize a pointer called 'current' to the dummy node.
4. Iterate through the list while there are at least two nodes ahead:
   - Identify the two nodes to swap (first and second)
   - Keep track of the node after the pair (next_pair)
   - Update the connections to perform the swap:
     - Connect current to second
     - Connect second to first
     - Connect first to next_pair
   - Update the prev pointers to maintain the doubly linked structure
   - Move current two nodes ahead for the next iteration
5. Return the new head (dummy.next).

## Complexity
- Time: O(n) - We iterate through the list once, where n is the number of nodes.
- Space: O(1) - We only use a constant amount of extra space for pointers, regardless of input size.
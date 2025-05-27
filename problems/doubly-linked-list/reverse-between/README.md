# Reverse Between in a Doubly Linked List

## Problem Description
Given the head of a doubly linked list and two positions 'left' and 'right', reverse the nodes of the list from position 'left' to position 'right', and return the head of the modified list.

The doubly linked list has nodes with both next and previous pointers. The position of the first node is 1 (not 0-indexed).

For example:
- Input: 1 ⟷ 2 ⟷ 3 ⟷ 4 ⟷ 5, left = 2, right = 4
- Output: 1 ⟷ 4 ⟷ 3 ⟷ 2 ⟷ 5

## Approach
1. Handle edge cases:
   - If the list is empty or left equals right, return the head as is.
2. Create a dummy node to handle edge cases and link it to the head.
3. Find the nodes at positions:
   - prev_left: The node at position (left-1)
   - start: The node at position left
   - end: The node at position right
   - after_end: The node at position (right+1)
4. Temporarily disconnect the sublist (from start to end) from the main list.
5. Reverse the sublist using a helper function.
6. Reconnect the reversed sublist back to the original list:
   - Connect prev_left to the new head of the reversed sublist
   - Connect the new tail of the reversed sublist (original start) to after_end
   - Update prev pointers to maintain the doubly linked structure
7. Return the head of the modified list.

## Complexity
- Time: O(n) - We might need to traverse up to n nodes to find the right position, where n is the number of nodes in the list.
- Space: O(1) - We only use a constant amount of extra space for the pointers, regardless of input size.
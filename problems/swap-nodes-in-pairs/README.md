# swap-Nodes-In-Pairs

## Problem Description
Given a linked list, swap every two adjacent nodes and return the head of the modified list. You should swap the nodes themselves, not just their values.

For example, given the linked list 1->2->3->4, the result should be 2->1->4->3.

If the list has an odd number of nodes, the last node should remain in its original position. For example, given 1->2->3, the result should be 2->1->3.

## Approach
Iterative approach with a dummy head:

1. Create a dummy node that points to the head of the list
2. Initialize a current pointer to the dummy node
3. Iterate through the list while current has at least two nodes after it:
   - Save pointers to the first and second nodes to be swapped
   - Modify the next pointers to perform the swap
   - Update current to move two nodes ahead for the next iteration
4. Return the dummy node's next pointer, which points to the head of the modified list


## Complexity
- Time: O(n)
- Space: O(1)
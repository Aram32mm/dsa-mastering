# Partition List for a Doubly Linked List

## Problem Description
Given the head of a doubly linked list and a value x, partition the list such that all nodes with values less than x come before nodes with values greater than or equal to x.

You should preserve the original relative order of the nodes in each of the two partitions.

For example:
- Input: 3 ⟷ 5 ⟷ 8 ⟷ 5 ⟷ 10 ⟷ 2 ⟷ 1, x = 5
- Output: 3 ⟷ 2 ⟷ 1 ⟷ 5 ⟷ 8 ⟷ 5 ⟷ 10

## Approach
1. Handle edge case: If the list is empty, return null.

2. Create two dummy nodes to act as the heads of two separate lists:
   - less_dummy: For nodes with values less than x
   - greater_dummy: For nodes with values greater than or equal to x

3. Initialize two pointers to track the tails of these lists:
   - less_tail: Initially points to less_dummy
   - greater_tail: Initially points to greater_dummy

4. Traverse the original list:
   - For each node, disconnect it from the original list
   - If its value is less than x, append it to the less list and update less_tail
   - Otherwise, append it to the greater list and update greater_tail

5. Connect the two lists by setting less_tail.next to greater_dummy.next

6. Update the prev pointer of the first node in the greater list (if it exists) to point to less_tail

7. Set the prev pointer of the head of the combined list to null

8. Return the head of the combined list (less_dummy.next)

## Complexity
- Time: O(n) - We traverse the list exactly once, where n is the number of nodes.
- Space: O(1) - We only use a constant amount of extra space regardless of input size.
# Partition List

## Problem Description
Given a linked list and a value x, partition the linked list such that all nodes with values less than x come before nodes with values greater than or equal to x. The original relative order of the nodes should be preserved.

For example, given the linked list 3->5->8->5->10->2->1 and x = 5, the result should be 3->2->1->5->8->5->10.

## Approach
We'll create two separate linked lists:
1. One for nodes with values less than x (the "less" list)
2. One for nodes with values greater than or equal to x (the "greater" list)

Then we'll traverse the original list, placing each node in the appropriate list while maintaining the original order. Finally, we'll connect the "less" list to the "greater" list to form the result.

This approach ensures that we preserve the original order within each partition, as required.

## Complexity
- Time: O(n)
- Space: O(1)

# Has-Loop

## Problem Description
Given a linked list, determine if it contains a cycle (a loop). A cycle in a linked list occurs when a node's next pointer points to a previously visited node, causing a loop in the traversal.

## Approach
Floyd's Cycle-Finding Algorithm ("tortoise and hare" algorithm), use of slow and fast pointers, if there is a cycle, the fast pointer will eventually lap the slow pointer 

## Complexity
- Time: O(n) 
- Space: O(1)

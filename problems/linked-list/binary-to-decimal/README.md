# Binary To Decimal

## Problem Description
Given a linked list where each node contains a single digit, convert the binary number represented by this linked list to a decimal integer.

## Approach
1. Initialize a decimal value to 0
2. Traverse the linked list from head to tail
3. For each node:
   - Multiply the current decimal value by 2 (shifting left)
   - Add the current node's value (0 or 1)
4. Return the final decimal value

we process each bit from most significant (leftmost) to least significant (rightmost).

## Complexity
- Time: O(n)
- Space: O(1)
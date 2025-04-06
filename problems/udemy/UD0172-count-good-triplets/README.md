# UD0172. Count Good Triplets

Platform: Udemy
Difficulty: Easy

## Problem Description

Problem consists in finding triplets in an array with two conditions:

	1. They must be in a GP, geometric progression of factor r. In other words they should be strictly increasing with a factor r (eg. 2, 4, 8 -> r = 2)

	2. Given (i, j, k) being the indices of the triplet. They must be i < j < k.

## Approach
Sliding Hashing. Keep two hash maps, left and right map, each maps stores the frequency of elements in each side at a 
given index. As you iterate through the array, you switch the frequencies from one map to the other so you ensure you always have the right count, finally you just look for the elements in the GP.

## Complexity Analysis
- Time Complexity: O(n)
- Space Complexity: O(n)

## Notes
Sliding hashing blew my mind.

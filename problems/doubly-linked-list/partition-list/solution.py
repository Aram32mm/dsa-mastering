class Node:
    def __init__(self, value=0, next=None, prev=None):
        self.value = value
        self.next = next
        self.prev = prev

class Solution:
    def partition(self, head, x):
        """
        Partitions a doubly linked list around a value x, such that all nodes with values
        less than x come before all nodes with values greater than or equal to x.
        
        Args:
            head: The head of the doubly linked list
            x: The value to partition around
            
        Returns:
            The head of the modified list
        """
        # Handle edge case: empty list
        if not head:
            return None
            
        # Create dummy nodes for the two partitions
        less_dummy = Node(0)
        greater_dummy = Node(0)
        
        # Initialize pointers for the tails of each partition
        less_tail = less_dummy
        greater_tail = greater_dummy
        
        # Traverse the original list
        current = head
        while current:
            # Save next node before modifying the current node
            next_node = current.next
            
            # Disconnect the current node from the list
            current.next = None
            current.prev = None
            
            if current.value < x:
                # Append to the less partition
                less_tail.next = current
                current.prev = less_tail
                less_tail = current
            else:
                # Append to the greater or equal partition
                greater_tail.next = current
                current.prev = greater_tail
                greater_tail = current
            
            current = next_node
        
        # Connect the two partitions: less + greater/equal
        less_tail.next = greater_dummy.next
        if greater_dummy.next:
            greater_dummy.next.prev = less_tail
        
        # Update the head of the combined list
        result = less_dummy.next
        
        # Set the prev pointer of the new head to None
        if result:
            result.prev = None
        
        return result

def create_doubly_linked_list(values):
    """Creates a doubly linked list from the given values."""
    if not values:
        return None
        
    dummy = Node(0)
    current = dummy
    
    for val in values:
        new_node = Node(val)
        new_node.prev = current
        current.next = new_node
        current = new_node
        
    # Set dummy.next.prev to None
    if dummy.next:
        dummy.next.prev = None
        
    return dummy.next

def doubly_linked_list_to_list(head):
    """Converts a doubly linked list to a list."""
    result = []
    current = head
    while current:
        result.append(current.value)
        current = current.next
    return result

def check_prev_pointers(head):
    """Checks if prev pointers are correctly set by traversing backward."""
    if not head:
        return True
        
    # Find the end of the list
    current = head
    while current.next:
        current = current.next
    
    # Traverse backward and collect values
    backward_values = []
    while current:
        backward_values.append(current.value)
        current = current.prev
    
    # Get forward values for comparison
    forward_values = doubly_linked_list_to_list(head)
    
    # Compare backward traversal with reversed forward traversal
    return backward_values == forward_values[::-1]

def test_solution():
    solution = Solution()
    
    # Test case 1: Regular case
    head1 = create_doubly_linked_list([3, 5, 8, 5, 10, 2, 1])
    result1 = solution.partition(head1, 5)
    assert doubly_linked_list_to_list(result1) == [3, 2, 1, 5, 8, 5, 10]
    assert check_prev_pointers(result1) == True
    
    # Test case 2: All elements less than x
    head2 = create_doubly_linked_list([1, 2, 3])
    result2 = solution.partition(head2, 4)
    assert doubly_linked_list_to_list(result2) == [1, 2, 3]
    assert check_prev_pointers(result2) == True
    
    # Test case 3: All elements greater than or equal to x
    head3 = create_doubly_linked_list([5, 6, 7])
    result3 = solution.partition(head3, 5)
    assert doubly_linked_list_to_list(result3) == [5, 6, 7]
    assert check_prev_pointers(result3) == True
    
    # Test case 4: Empty list
    result4 = solution.partition(None, 5)
    assert result4 is None
    
    # Test case 5: Single node
    head5 = create_doubly_linked_list([3])
    result5 = solution.partition(head5, 5)
    assert doubly_linked_list_to_list(result5) == [3]
    assert check_prev_pointers(result5) == True
    
    # Test case 6: Already partitioned
    head6 = create_doubly_linked_list([1, 2, 3, 5, 6, 7])
    result6 = solution.partition(head6, 4)
    assert doubly_linked_list_to_list(result6) == [1, 2, 3, 5, 6, 7]
    assert check_prev_pointers(result6) == True
    
    # Test case 7: All equal elements, some less than x
    head7 = create_doubly_linked_list([3, 3, 3, 3])
    result7 = solution.partition(head7, 5)
    assert doubly_linked_list_to_list(result7) == [3, 3, 3, 3]
    assert check_prev_pointers(result7) == True
    
    print("All test cases passed!")

if __name__ == "__main__":
    test_solution()
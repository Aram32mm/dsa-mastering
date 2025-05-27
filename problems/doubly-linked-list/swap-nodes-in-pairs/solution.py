class Node:
    def __init__(self, value=0, next=None, prev=None):
        self.value = value
        self.next = next
        self.prev = prev

class Solution:
    def swap_pairs(self, head):
        """
        Swaps every two adjacent nodes in a doubly linked list and returns the new head.
        
        Args:
            head: The head of the doubly linked list
            
        Returns:
            The new head of the modified list
        """
        # Handle edge cases: empty list or single node
        if not head or not head.next:
            return head
            
        # Create a dummy node to handle edge cases
        dummy = Node(0)
        dummy.next = head
        head.prev = dummy
        
        # Initialize a pointer to the node before each pair
        current = dummy
        
        # Iterate while there are at least two more nodes to swap
        while current.next and current.next.next:
            # Identify nodes to swap
            first = current.next
            second = current.next.next
            
            # Get the node after second (might be None)
            next_pair = second.next
            
            # Update connections to perform the swap
            # 1. Connect current to second
            current.next = second
            second.prev = current
            
            # 2. Connect second to first
            second.next = first
            first.prev = second
            
            # 3. Connect first to next_pair
            first.next = next_pair
            if next_pair:
                next_pair.prev = first
            
            # Move current two nodes ahead for next iteration
            current = first
        
        return dummy.next

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
    
    # Test case 1: Regular case - even number of nodes
    head1 = create_doubly_linked_list([1, 2, 3, 4])
    result1 = solution.swap_pairs(head1)
    assert doubly_linked_list_to_list(result1) == [2, 1, 4, 3]
    assert check_prev_pointers(result1) == True
    
    # Test case 2: Odd number of nodes
    head2 = create_doubly_linked_list([1, 2, 3])
    result2 = solution.swap_pairs(head2)
    assert doubly_linked_list_to_list(result2) == [2, 1, 3]
    assert check_prev_pointers(result2) == True
    
    # Test case 3: Single node
    head3 = create_doubly_linked_list([1])
    result3 = solution.swap_pairs(head3)
    assert doubly_linked_list_to_list(result3) == [1]
    assert check_prev_pointers(result3) == True
    
    # Test case 4: Empty list
    result4 = solution.swap_pairs(None)
    assert result4 is None
    
    # Test case 5: Two nodes
    head5 = create_doubly_linked_list([1, 2])
    result5 = solution.swap_pairs(head5)
    assert doubly_linked_list_to_list(result5) == [2, 1]
    assert check_prev_pointers(result5) == True
    
    # Test case 6: Longer list
    head6 = create_doubly_linked_list([1, 2, 3, 4, 5, 6])
    result6 = solution.swap_pairs(head6)
    assert doubly_linked_list_to_list(result6) == [2, 1, 4, 3, 6, 5]
    assert check_prev_pointers(result6) == True
    
    print("All test cases passed!")

if __name__ == "__main__":
    test_solution()
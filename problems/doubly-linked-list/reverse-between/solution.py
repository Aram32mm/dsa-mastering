class Node:
    def __init__(self, value=0, next=None, prev=None):
        self.value = value
        self.next = next
        self.prev = prev

class Solution:
    def reverse_between(self, head, left, right):
        """
        Reverses the nodes of a doubly linked list from position left to position right.
        The first node is at position 1.
        
        Args:
            head: The head of the doubly linked list
            left: The position of the first node to be reversed
            right: The position of the last node to be reversed
            
        Returns:
            The head of the modified list
        """
        # Handle edge cases
        if not head or left == right:
            return head
            
        # Create a dummy node to handle edge cases
        dummy = Node(0)
        dummy.next = head
        head.prev = dummy
        
        # Find the node at position (left-1)
        prev_left = dummy
        for _ in range(left - 1):
            prev_left = prev_left.next
            if not prev_left:
                return head  # If left is out of bounds
        
        # Find the node at position left
        start = prev_left.next
        if not start:
            return head  # If left is out of bounds
        
        # Find the node at position right
        end = start
        for _ in range(right - left):
            end = end.next
            if not end:
                break  # If right is out of bounds
        
        # Find the node at position (right+1)
        after_end = end.next
        
        # Disconnect the sublist to be reversed
        prev_left.next = None
        if start.prev:
            start.prev = None
        if end.next:
            end.next = None
        
        # Reverse the sublist
        reversed_head = self._reverse(start)
        
        # Connect the reversed sublist back to the original list
        prev_left.next = reversed_head
        reversed_head.prev = prev_left
        start.next = after_end
        if after_end:
            after_end.prev = start
        
        return dummy.next
    
    def _reverse(self, head):
        """Helper function to reverse a doubly linked list."""
        if not head or not head.next:
            return head
            
        current = head
        prev = None
        
        while current:
            next_node = current.next
            current.next = prev
            current.prev = next_node
            prev = current
            current = next_node
            
        return prev

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
    head1 = create_doubly_linked_list([1, 2, 3, 4, 5])
    result1 = solution.reverse_between(head1, 2, 4)
    assert doubly_linked_list_to_list(result1) == [1, 4, 3, 2, 5]
    assert check_prev_pointers(result1) == True
    
    # Test case 2: Reverse entire list
    head2 = create_doubly_linked_list([1, 2, 3, 4, 5])
    result2 = solution.reverse_between(head2, 1, 5)
    assert doubly_linked_list_to_list(result2) == [5, 4, 3, 2, 1]
    assert check_prev_pointers(result2) == True
    
    # Test case 3: No reversal (left == right)
    head3 = create_doubly_linked_list([1, 2, 3])
    result3 = solution.reverse_between(head3, 2, 2)
    assert doubly_linked_list_to_list(result3) == [1, 2, 3]
    assert check_prev_pointers(result3) == True
    
    # Test case 4: Empty list
    result4 = solution.reverse_between(None, 1, 2)
    assert result4 is None
    
    # Test case 5: Single node
    head5 = create_doubly_linked_list([1])
    result5 = solution.reverse_between(head5, 1, 1)
    assert doubly_linked_list_to_list(result5) == [1]
    assert check_prev_pointers(result5) == True
    
    # Test case 6: Reverse first two nodes
    head6 = create_doubly_linked_list([1, 2, 3, 4])
    result6 = solution.reverse_between(head6, 1, 2)
    assert doubly_linked_list_to_list(result6) == [2, 1, 3, 4]
    assert check_prev_pointers(result6) == True
    
    # Test case 7: Reverse last two nodes
    head7 = create_doubly_linked_list([1, 2, 3, 4])
    result7 = solution.reverse_between(head7, 3, 4)
    assert doubly_linked_list_to_list(result7) == [1, 2, 4, 3]
    assert check_prev_pointers(result7) == True
    
    print("All test cases passed!")

if __name__ == "__main__":
    test_solution()
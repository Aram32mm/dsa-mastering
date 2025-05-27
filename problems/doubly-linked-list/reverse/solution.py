class Node:
    def __init__(self, value=0, next=None, prev=None):
        self.value = value
        self.next = next
        self.prev = prev

class Solution:
    def reverse(self, head):
        """
        Reverses a doubly linked list.
        
        Args:
            head: The head of the doubly linked list
            
        Returns:
            The new head of the reversed list
        """
        # Handle edge cases: empty list or single node
        if not head or not head.next:
            return head
            
        # Initialize current pointer and prev pointer
        current = head
        prev = None
        
        # Iterate through the list
        while current:
            # Store next node before we change the pointers
            next_node = current.next
            
            # Reverse the pointers
            current.next = prev
            current.prev = next_node
            
            # Move prev and current one step forward
            prev = current
            current = next_node
            
        # The new head is the last node we visited (prev)
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

def test_solution():
    solution = Solution()
    
    # Test case 1: Regular list
    head1 = create_doubly_linked_list([1, 2, 3, 4, 5])
    result1 = solution.reverse(head1)
    assert doubly_linked_list_to_list(result1) == [5, 4, 3, 2, 1]
    
    # Test case 2: Single node
    head2 = create_doubly_linked_list([1])
    result2 = solution.reverse(head2)
    assert doubly_linked_list_to_list(result2) == [1]
    
    # Test case 3: Empty list
    result3 = solution.reverse(None)
    assert result3 is None
    
    # Test case 4: Two nodes
    head4 = create_doubly_linked_list([1, 2])
    result4 = solution.reverse(head4)
    assert doubly_linked_list_to_list(result4) == [2, 1]
    
    # Test case 5: Check proper prev links after reversal
    head5 = create_doubly_linked_list([1, 2, 3])
    reversed_head = solution.reverse(head5)
    # Traverse forward
    forward_results = []
    current = reversed_head
    while current:
        forward_results.append(current.value)
        last_node = current
        current = current.next
    # Traverse backward
    backward_results = []
    current = last_node
    while current:
        backward_results.append(current.value)
        current = current.prev
    assert forward_results == [3, 2, 1]
    assert backward_results == [1, 2, 3]
    
    print("All test cases passed!")

if __name__ == "__main__":
    test_solution()
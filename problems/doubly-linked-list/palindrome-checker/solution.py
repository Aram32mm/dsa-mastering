class Node:
    def __init__(self, value=0, next=None, prev=None):
        self.value = value
        self.next = next
        self.prev = prev

class Solution:
    def is_palindrome(self, head):
        if not head or not head.next:
            return True
            
        tail = head
        while tail.next:
            tail = tail.next
            
        # Initialize pointers from both ends
        left = head
        right = tail
        
        # Compare nodes moving from both ends toward the middle
        while left and right and left != right and left.prev != right:
            if left.value != right.value:
                return False
            left = left.next
            right = right.prev
            
        return True
        
        # Approach 2: Alternative method for singly linked lists (not using prev pointers)
        # This is an alternative implementation if we couldn't use prev pointers
        """
        # Find the middle of the list using slow and fast pointers
        slow = head
        fast = head
        
        # Move slow to the middle
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        # Reverse the second half of the list
        second_half = self._reverse(slow)
        second_half_head = second_half
        
        # Compare the first half with the reversed second half
        result = True
        first_half = head
        
        while second_half:
            if first_half.value != second_half.value:
                result = False
                break
            first_half = first_half.next
            second_half = second_half.next
            
        # Restore the list (optional)
        self._reverse(second_half_head)
        
        return result
        """
    
    def _reverse(self, head):
        if not head:
            return None
            
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

def test_solution():
    solution = Solution()
    
    # Test case 1: Palindrome (even length)
    head1 = create_doubly_linked_list([1, 2, 2, 1])
    assert solution.is_palindrome(head1) == True
    
    # Test case 2: Palindrome (odd length)
    head2 = create_doubly_linked_list([1, 2, 3, 2, 1])
    assert solution.is_palindrome(head2) == True
    
    # Test case 3: Not a palindrome
    head3 = create_doubly_linked_list([1, 2, 3, 4])
    assert solution.is_palindrome(head3) == False
    
    # Test case 4: Single node (trivial palindrome)
    head4 = create_doubly_linked_list([5])
    assert solution.is_palindrome(head4) == True
    
    # Test case 5: Empty list (trivial palindrome)
    assert solution.is_palindrome(None) == True
    
    # Test case 6: Two nodes (palindrome)
    head6 = create_doubly_linked_list([7, 7])
    assert solution.is_palindrome(head6) == True
    
    # Test case 7: Two nodes (not a palindrome)
    head7 = create_doubly_linked_list([7, 8])
    assert solution.is_palindrome(head7) == False
    
    # Test case 8: Longer palindrome
    head8 = create_doubly_linked_list([1, 2, 3, 4, 5, 4, 3, 2, 1])
    assert solution.is_palindrome(head8) == True
    
    print("All test cases passed!")

if __name__ == "__main__":
    test_solution()
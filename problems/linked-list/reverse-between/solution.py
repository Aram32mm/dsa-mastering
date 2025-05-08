class Node:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

class Solution:
    def reverse_between(self, head, left, right):
        if not head or left == right:
            return head
            
        dummy = Node(0)
        dummy.next = head
        
        # Find the node at position (left-1)
        prev_left = dummy
        for _ in range(left - 1):
            prev_left = prev_left.next
        
        # start node
        start = prev_left.next
        
        curr = start
        prev = None
        for _ in range(right - left + 1):
            next_node = curr.next  
            curr.next = prev
            prev = curr 
            curr = next_node
            
        # Connect the reversed portion back to the list
        start.next = curr         # Connect the last node of reversed portion to the rest
        prev_left.next = prev     # Connect the node before 'left' to the new start
        
        return dummy.next

def create_linked_list(values):
    dummy = Node()
    current = dummy
    for val in values:
        current.next = Node(val)
        current = current.next
    return dummy.next

def linked_list_to_list(head):
    result = []
    current = head
    while current:
        result.append(current.value)
        current = current.next
    return result

def test_solution():
    head1 = create_linked_list([1, 2, 3, 4, 5])
    result1 = Solution().reverse_between(head1, 2, 4)
    assert linked_list_to_list(result1) == [1, 4, 3, 2, 5]
    
    head2 = create_linked_list([1, 2, 3, 4, 5])
    result2 = Solution().reverse_between(head2, 1, 5)
    assert linked_list_to_list(result2) == [5, 4, 3, 2, 1]
    
    head3 = create_linked_list([1, 2, 3])
    result3 = Solution().reverse_between(head3, 2, 2)
    assert linked_list_to_list(result3) == [1, 2, 3]
    
    result4 = Solution().reverse_between(None, 1, 2)
    assert result4 is None
    
    print("All test cases passed!")

if __name__ == "__main__":
    test_solution()

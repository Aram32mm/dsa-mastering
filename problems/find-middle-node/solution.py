class Node:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

class Solution:
    def find_middle_node(self, head):
        if not head:
            return None
        
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
        return slow

def create_linked_list(values):
    dummy = Node()
    current = dummy
    for val in values:
        current.next = Node(val)
        current = current.next
    return dummy.next

def test_solution():
    head1 = create_linked_list([1, 2, 3, 4, 5])
    result1 = Solution().find_middle_node(head1)
    assert result1.value == 3
    
    head2 = create_linked_list([1, 2, 3, 4, 5, 6])
    result2 = Solution().find_middle_node(head2)
    assert result2.value == 4
    
    head3 = create_linked_list([1])
    result3 = Solution().find_middle_node(head3)
    assert result3.value == 1
    
    # Test case 4: Empty list
    result4 = Solution().find_middle_node(None)
    assert result4 is None

    print("All test cases passed!")
    
if __name__ == "__main__":
    test_solution()

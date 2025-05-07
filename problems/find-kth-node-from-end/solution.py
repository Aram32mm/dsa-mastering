class Node:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

class Solution:
    def find_kth_from_end(self, head, k):
        if not head or k < 1:
            return None
            
        fast = slow = head
        
        for _ in range(k):
            if not fast:
                return None 
            fast = fast.next
            
        while fast:
            slow = slow.next
            fast = fast.next
            
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
    result1 = Solution().find_kth_from_end(head1, 2)
    assert result1.value == 4

    head2 = create_linked_list([1, 2, 3])
    result2 = Solution().find_kth_from_end(head2, 3)
    assert result2.value == 1
    
    head3 = create_linked_list([1, 2, 3])
    result3 = Solution().find_kth_from_end(head3, 4)
    assert result3 is None
    
    result4 = Solution().find_kth_from_end(None, 1)
    assert result4 is None

    print("All test cases passed!")

if __name__ == "__main__":
    test_solution()

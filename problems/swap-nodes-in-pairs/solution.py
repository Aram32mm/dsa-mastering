class Node:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

class Solution:
    def swap_pairs(self, head):
        
        if not head or not head.next:
            return head
            
        # Create dummy node to handle edge cases
        dummy = Node(0)
        dummy.next = head
        
        # Initialize current pointer to dummy node
        current = dummy
        
        # Iterate while there are at least two more nodes
        while current.next and current.next.next:
            # Identify nodes to swap
            first = current.next
            second = current.next.next
            
            # Perform the swap
            first.next = second.next
            second.next = first
            current.next = second
            
            # Move current two nodes ahead for next iteration
            current = first
            
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
    head1 = create_linked_list([1, 2, 3, 4])
    result1 = Solution().swap_pairs(head1)
    assert linked_list_to_list(result1) == [2, 1, 4, 3]
    
    head2 = create_linked_list([1, 2, 3])
    result2 = Solution().swap_pairs(head2)
    assert linked_list_to_list(result2) == [2, 1, 3]
    
    head3 = create_linked_list([1])
    result3 = Solution().swap_pairs(head3)
    assert linked_list_to_list(result3) == [1]

    result4 = Solution().swap_pairs(None)
    assert result4 is None
    
    print("All test cases passed!")

if __name__ == "__main__":
    test_solution()

class Node:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

class Solution:
    def partition(self, head, x):
        if not head:
            return None
            
        less_dummy = less_tail = Node(0)
        greater_dummy = greater_tail = Node(0)
        
        current = head
        while current:
            # Save next pointer before modifying the node
            next_node = current.next
            current.next = None

            if current.value < x:
                less_tail.next = current
                less_tail = less_tail.next
            else:
                greater_tail.next = current
                greater_tail = greater_tail.next

            current = next_node
            
        # Connect the lists
        less_tail.next = greater_dummy.next
        
        # Return the head of the merged list
        return less_dummy.next

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
    head1 = create_linked_list([3, 5, 8, 5, 10, 2, 1])
    result1 = Solution().partition(head1, 5)
    assert linked_list_to_list(result1) == [3, 2, 1, 5, 8, 5, 10]
    
    head2 = create_linked_list([1, 2, 3])
    result2 = Solution().partition(head2, 4)
    assert linked_list_to_list(result2) == [1, 2, 3]
    
    head3 = create_linked_list([5, 6, 7])
    result3 = Solution().partition(head3, 5)
    assert linked_list_to_list(result3) == [5, 6, 7]
    
    result4 = Solution().partition(None, 5)
    assert result4 is None

    print("All test cases passed!")

if __name__ == "__main__":
    test_solution()

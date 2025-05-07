class Node:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

class Solution:
    def remove_duplicates(self, head):
        if not head:
            return None

        seen = set()

        dummy = Node(0)
        dummy.next = head
        
        prev = dummy
        curr = head
        
        while curr:
            if curr.value in seen:
                prev.next = curr.next
            else:
                seen.add(curr.value)
                prev = curr
                
            curr = curr.next
            
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
    head1 = create_linked_list([1, 2, 1, 3, 2, 1, 4])
    result1 = Solution().remove_duplicates(head1)
    assert linked_list_to_list(result1) == [1, 2, 3, 4]
    
    head2 = create_linked_list([1, 2, 3, 4])
    result2 = Solution().remove_duplicates(head2)
    assert linked_list_to_list(result2) == [1, 2, 3, 4]
    
    head3 = create_linked_list([1, 1, 1, 1])
    result3 = Solution().remove_duplicates(head3)
    assert linked_list_to_list(result3) == [1]

    result4 = Solution().remove_duplicates(None)
    assert result4 is None

    print("All test cases passed!")

if __name__ == "__main__":
    test_solution()

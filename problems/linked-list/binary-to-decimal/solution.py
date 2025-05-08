class Node:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

class Solution:
    def binary_to_decimal(self, head):
        if not head:
            return 0
            
        decimal = 0
        current = head

        while current:
            # Left shift by 1 (multiply by 2) and add current bit
            decimal = decimal * 2 + current.value
            current = current.next

        return decimal

def create_linked_list(values):
    dummy = Node()
    current = dummy
    for val in values:
        current.next = Node(val)
        current = current.next
    return dummy.next

def test_solution():
    head1 = create_linked_list([1, 0, 1])
    assert Solution().binary_to_decimal(head1) == 5
    
    head2 = create_linked_list([1, 1, 0, 1])
    assert Solution().binary_to_decimal(head2) == 13
    
    head3 = create_linked_list([0])
    assert Solution().binary_to_decimal(head3) == 0
    
    assert Solution().binary_to_decimal(None) == 0

    print("All test cases passed!")

if __name__ == "__main__":
    test_solution()

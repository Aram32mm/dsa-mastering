class Node:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

class Solution:
    def has_loop(self, head):
        if not head or not head.next:
            return False
            
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                return True
                
        return False

def create_linked_list(values, pos=-1):
    if not values:
        return None
        
    nodes = []
    for val in values:
        nodes.append(Node(val))
    
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    
    # Create cycle if specified
    if pos >= 0 and pos < len(nodes):
        nodes[-1].next = nodes[pos]
    
    return nodes[0]


def test_solution():
    head1 = create_linked_list([1, 2, 3, 4], pos=1)
    assert Solution().has_loop(head1) == True
    
    head2 = create_linked_list([1, 2, 3, 4])
    assert Solution().has_loop(head2) == False
    
    head3 = create_linked_list([1], pos=0)
    assert Solution().has_loop(head3) == True
    
    assert Solution().has_loop(None) == False

    print("All test cases passed!")

if __name__ == "__main__":
    test_solution()

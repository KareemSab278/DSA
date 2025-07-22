
# Node: holds data and link to next node
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None  # points to next node

# LinkedList: just points to first node (head)
class LinkedList:
    def __init__(self):
        self.head = None  # empty list starts here

    # Add to front
    def add_front(self, data):
        new_node = Node(data)     # create new node
        new_node.next = self.head # new node points to current head
        self.head = new_node      # head is now new node

    # Print all elements
    def print_list(self):
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

# Usage:
ll = LinkedList()
ll.add_front(10)
ll.add_front(20)
ll.add_front(30)

ll.print_list()  # Output: 30 -> 20 -> 10 -> None

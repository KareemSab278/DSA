# class Node: #basically an object bruh
#     def __init__(self, data):
#         self.data = data      # Store the value of this node
#         self.next = None      # Pointer to the next node, starts as None

# class LinkedList:
#     def __init__(self):
#         self.head = None      # Start of the list, initially empty
#         self.tail = None      # End of the list, initially empty

#     def add(self, data):
#         new_node = Node(data)  # Create a new node with the given data
#         if not self.head:      # If list is empty (no head)
#             self.head = new_node  # New node becomes the head
#             self.tail = new_node  # New node is also the tail since list has one node
#         else:
#             self.tail.next = new_node  # Link current tail node to new node
#             self.tail = new_node       # Update tail pointer to new node (new end of list)

#     def show(self):
#         current = self.head
#         while current:  # Traverse the list
#             print(current.data, end=" -> ")  # Print current node's data
#             current = current.next  # Move to the next node
#         print("None")  # Indicate the end of the list

# linkedlist = LinkedList()
# linkedlist.add(10)
# linkedlist.add(20)
# linkedlist.add(30)
# linkedlist.show()  # Output: 10 -> 20 -> 30 -> None



































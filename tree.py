class Node:
    def __init__(self, value):
        self.value = value
        self.children = []
    
    def add_child(self, value):
        self.children.append(Node(value))

class Tree:
    def __init__(self, root_value): 
        self.root = Node(root_value)
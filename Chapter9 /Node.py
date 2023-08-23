class Node:
    def __init__(self, value=None, parents=[]):
        self.value = value
        self.parents = parents
        self.grad = 0

    def forward(self):
        raise NotImplemented

    def backward(self):
        raise NotImplemented


class AddNode(Node):
    def forward(self):
        self.value = sum([parent.value for parent in self.parents])
        return self.value

    def backward(self):
        for parent in self.parents:
            parent.grad += 1 * self.grad


class MulNode(Node):
    def forward(self):
        vals = [parent.value for parent in self.parents]
        self.value = vals[0] * vals[1]
        return self.value

    def backward(self):
        self.parents[0].grad += self.parents[1].value * self.grad
        self.parents[1].grad += self.parents[0].value * self.grad


# Define our computation
x = Node(value=2)
y = Node(value=3)
z = Node(value=4)

add_node = AddNode(parents=[x, y])
mul_node = MulNode(parents=[add_node, z])

# Forward pass
f_value = mul_node.forward()
print(f_value)  # This should print 20 (i.e., (2 + 3) * 4)

# Backward pass
mul_node.grad = 1  # derivative of a function with respect to itself is 1
mul_node.backward()
add_node.backward()

print(x.grad, y.grad, z.grad)  # gradients for x, y, and z

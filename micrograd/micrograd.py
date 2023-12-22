import math
import random
import graphviz


class Value:

    def __init__(self, data, label='', _inputs=(), _op=''):
        self.data = data
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None
        self._inputs = set(_inputs)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _inputs=(self, other), _op="+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _inputs=(self, other), _op="*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only ints & floats supported"

        out = Value(self.data**other, _inputs=(self,), _op=f"**{other}")

        def _backward():
            self.grad += other * self.data**(other - 1) * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, _inputs=(self,), _op="tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        x = self.data

        ex = math.exp(x)
        out = Value(ex, _inputs=(self,), _op="exp")

        def _backward():
            self.grad += ex * out.grad

        out._backward = _backward

        return out

    @property
    def uid(self):
        return str(id(self))

    def backward(self):

        topo = build_topo_sorted_graph(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class Neuron:

    def __init__(self, num_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum([v[0] * v[1] for v in zip(self.w, x)]) + self.b
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:

    def __init__(self, num_inputs, num_outputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_outputs)]

    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:

    def __init__(self, num_inputs, num_outputs):
        layer_sizes = [num_inputs] + num_outputs
        self.layers = [Layer(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(num_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]


def trace(root):
    nodes = set()
    edges = set()
    def build(node):
        if node not in nodes:
            nodes.add(node)
            for input_node in node._inputs:
                edges.add((input_node, node))
                build(input_node)
    build(root)
    return nodes, edges


def draw_dot(root):
    dot = graphviz.Digraph(format='svg', graph_attr={"rankdir": "LR"}) # LR == Left to right

    nodes, edges = trace(root)
    for node in nodes:

        # For any Value, draw a rectangular node
        dot.node(name=node.uid, label="{ %s | data %.4f | grad %.4f }" % (node.label, node.data, node.grad), shape="record")

        # If the Value is the output of an operation, draw a op node.
        if node._op:
            dot.node(name=node.uid + node._op, label=node._op)

            # Connect the operation to the output
            dot.edge(node.uid + node._op, node.uid)

    # Connect all the inputs to their operation nodes.
    for input_node, output_node in edges:
        dot.edge(input_node.uid, output_node.uid + output_node._op)

    return dot


def build_topo_sorted_graph(o):
    topo = []
    visited = set()

    def build_topo(node):
        if node in visited:
            return

        visited.add(node)

        for input_node in node._inputs:
            build_topo(input_node)

        topo.append(node)

    build_topo(o)

    return topo


def create_example_expression():
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")

    e = a * b
    e.label = "e"

    d = e + c
    d.label = "d"

    f = Value(-2.0, label="f")
    L = d * f
    L.label = "L"

    return L


def create_example_neuron_with_tanh():
    # Inputs
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")

    # Weights
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")

    # Bias of the neuron
    b = Value(6.8813735870195432, label="b")

    # Compose the variables into a neuron
    x1w1 = x1 * w1
    x1w1.label = "x1 * w1"

    x2w2 = x2 * w2
    x2w2.label = "x2 * w2"

    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1*w1 + x2*w2"

    n = x1w1x2w2 + b
    n.label = "n"

    o = n.tanh()
    o.label = "o"

    return o


def create_example_neuron_without_using_tanh_directly():
    # Inputs
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")

    # Weights
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")

    # Bias of the neuron
    b = Value(6.8813735870195432, label="b")

    # Compose the variables into a neuron
    x1w1 = x1 * w1
    x1w1.label = "x1 * w1"

    x2w2 = x2 * w2
    x2w2.label = "x2 * w2"

    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1*w1 + x2*w2"

    n = x1w1x2w2 + b
    n.label = "n"

    # This is equivalent to tanh
    e = (2 * n).exp()
    o = (e - 1) / (e + 1)

    o.label = "o"

    return o


def example_data_set():

    network = MLP(3, [4, 4, 1])

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    for x in range(100):
        ypred = [network(x) for x in xs]

        # Forward pass
        loss = sum([(output - target)**2 for target, output in zip(ys, ypred)])

        # Reset gradients
        for p in network.parameters():
            p.grad = 0.0

        # Backward pass to propagate gradients.
        loss.backward()

        # Nudge weights and biases in the opposite direction of the gradient to
        # minimize the loss.
        for p in network.parameters():
            p.data += -0.01 * p.grad

        print(loss, ypred)

    return loss


if __name__ == "__main__":

    o = example_data_set()
    dot = draw_dot(o)
    dot.view()

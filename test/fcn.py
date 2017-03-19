from faketf import Variable, Placeholder
from faketf.train import SGD
from faketf import get_graph
import numpy as np


if __name__ == '__main__':
    x = Placeholder(shape=[3], name="x")
    W = Variable(np.random.randn(3), name="Weight")
    b = Variable(0, name="bias")
    y = x * W + b
    x_value = np.array([1.0, -1.0, 2.0])
    graph = get_graph()
    trainable = graph.trainable
    with graph.compute_gradients():
        gradients = y.auto_derivate(trainable)
    for variable, gradient in gradients.items():
        print(variable, gradient.eval({x: x_value}))

from faketf import reduce_sum, reduce_mean, Variable, Placeholder
import numpy as np


x = Placeholder([2, 5], name="x")
w = Variable([[0],[1],[2],[3],[4]], name="weight")
y = x @ w
l = reduce_mean(y)
graph = y.graph
print(l.eval({x: [[1,2,3,4,5], [2,3,4,5,6]]}))
with graph.compute_gradients():
    gradients = l.auto_derivate({w})
print(gradients[w].eval({x: [[1,2,3,4,5], [2,3,4,5,6]]}))

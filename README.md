# fake ft
This is a self practice to imitate Google's tensorflow.
The fucntion of this model is mainly to realize automatic derivation, optimization, etc.

Of course, all the calculations are done on CPU without multiprocess.

## TODO list
- ~~cache for repeated calculation~~
- operations such as abs, pow
- cache for repeated constants and operations
- random initialization
- None in placeholder shape
- activation functions
- SGD
- higher end apis

## Usage
```python
from faketf import reduce_mean, Variable, Placeholder

x = Placeholder([2, 5], name="x")
y_real = Placeholder([2, 1], name="real")
w = Variable([[0.0], [0.1], [0.2], [-0.2], [-0.1]], name="weight")
b = Variable([0.0], name="bias")
y = x @ w + b
loss = reduce_mean(y - y_real)  # for now power is not realized yet
graph = y.graph

x_values = [[1, 2, 3, -4, 0], [2, 1, 0, 5, 6]]
y_values = [[1], [2]]
feed_dict = {x: x_values, y_real: y_values}
print("Loss:", loss.eval(feed_dict))
with graph.compute_gradients():
    gradients = loss.auto_derivate({w, b})
with graph.enable_cache():
    print("Gradient for weights:", gradients[w].eval(feed_dict))
    print("Gradient for bias:", gradients[b].eval(feed_dict))
```

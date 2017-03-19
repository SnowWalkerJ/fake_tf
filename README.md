# fake ft
This is a self practice to imitate Google's tensorflow.
The fucntion of this model is mainly to realize automatic derivation, optimization, etc.

Of course, all the calculations are done on CPU without multiprocess.

## TODO list
- ~~cache for repeated calculation~~
- operations such as ~~abs~~, pow
- cache for repeated constants and operations
- ~~random initialization~~
- None in placeholder shape
- ~~activation functions~~
- ~~SGD~~
- higher end apis

## Usage
```python
from faketf import reduce_mean, Placeholder
from faketf.highend.layers import fully_connected
from faketf.train import SGD
import numpy as np

x = Placeholder([100, 5], name="x")
y_real = Placeholder([100, 1], name="real")
y = fully_connected(x, 1)
loss = reduce_mean(abs(y - y_real))  # for now power is not realized yet
trainer = SGD(loss, learning_rate=0.01)

x_values = np.random.randn(100, 5)
true_weights = np.array([[1.0], [0.5], [-0.5], [2.0], [-0.3]])
y_values = x_values @ true_weights
feed_dict = {x: x_values, y_real: y_values}

for i in range(1000):
    trainer.train(feed_dict)
    if i % 100 == 0:
        trainer.learning_rate *= 0.8
        print("Loss: %0.4f" % np.asscalar(loss.eval(feed_dict)))
print(y.W.eval(), y.b.eval())
```

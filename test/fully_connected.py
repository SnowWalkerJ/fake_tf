from faketf import reduce_mean, Placeholder
from faketf.highend.layers import fully_connected
from faketf.train import SGD
import numpy as np

x = Placeholder([1000, 5], name="x")
y_real = Placeholder([1000, 1], name="real")
y = fully_connected(x, 1)
loss = reduce_mean((y - y_real)**2)
trainer = SGD(loss, learning_rate=0.01)

x_values = np.random.randn(1000, 5)
true_weights = np.array([[1.0], [0.5], [-0.5], [2.0], [-0.3]])
noise = np.random.randn(1000, 1) * 0.1
y_values = 2.0 + x_values @ true_weights + noise        # a linear regression problem with noise
feed_dict = {x: x_values, y_real: y_values}

for i in range(3000):
    trainer.train(feed_dict)
    if i % 100 == 0:
        trainer.learning_rate *= 0.9     # exponential decay for learning rate
        print("Loss: %0.4f" % np.asscalar(loss.eval(feed_dict)))
print(y.W.eval(), y.b.eval())

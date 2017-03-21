from faketf import reduce_mean, Placeholder
from faketf.highend.layers import fully_connected
from faketf.train import SGD
import numpy as np

x = Placeholder([None, 5], name="x")
y_real = Placeholder([None, 1], name="real")
y = fully_connected(x, 1)
loss = reduce_mean((y - y_real)**2)  # for now power is not realized yet
trainer = SGD(loss, learning_rate=0.01)


for i in range(3000):
    x_values = np.random.randn(1000, 5)
    true_weights = np.array([[1.0], [0.5], [-0.5], [2.0], [-0.3]])
    noise = np.random.randn(1000, 1) * 0.3
    y_values = x_values @ true_weights + noise
    feed_dict = {x: x_values, y_real: y_values}
    trainer.train(feed_dict)
    if i % 100 == 0:
        trainer.learning_rate *= 0.9
        print("Loss: %0.4f" % np.asscalar(loss.eval(feed_dict)))
print(y.W.eval(), y.b.eval())

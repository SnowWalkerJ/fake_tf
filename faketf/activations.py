from .tensor import Op, Tensor
import numpy as np


class Sigmoid(Op):
    def __init__(self, income: Tensor):
        self.income = income
        self.direct_dependencies = {income}
        self.gradients = {income: self * (1 - self)}
        super(Sigmoid, self).__init__(shape=income.shape, dtype=income.dtype, name="Sigmoid")

    def eval(self, feed_dict=None):
        income_value = self.income.eval(feed_dict)
        return 1 / 1 + np.exp(-income_value)

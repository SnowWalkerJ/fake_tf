import numpy as np
from operations import AddOp
from tensor import Tensor


class Variable(Tensor):
    def __init__(self, values, dtype, name=None):
        self.values = np.asarray(values, dtype=dtype)
        name = name or "Variable"
        super(Variable, self).__init__(shape=self.values.shape, dtype=dtype, dependency=None, name=name)

    def __add__(self, other):
        return AddOp(self, other)

from op import Op
from tensor import Tensor


class AddOp(Op):
    def __init__(self, t1: Tensor, t2: Tensor):
        if t1.shape != t2.shape:
            raise TypeError("shape of %s and %s incompatible: `%s` and `%s`" % (t1.name, t2.name, t1.shape, t2.shape))
        self.t1 = t1
        self.t2 = t2
        super(AddOp, self).__init__(dependency=Op.get_dependency([t1, t2]), shape=t1.shape, dtype=t1.dtype, name="Add")

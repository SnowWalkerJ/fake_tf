from collections import defaultdict
import numpy as np
from .graph import get_graph
from typing import Set
from .constant import *


class Tensor:
    tensor_type = None
    direct_dependencies = None

    def __init__(self, shape, dtype, dependency=None, name=None):
        self.dependency = dependency or set()
        self.direct_dependencies = self.direct_dependencies or set()
        self.shape = shape
        self.dtype = dtype
        self.graph = get_graph()
        self.name = self.graph.check_name(self, name)
        self.graph.add_tensor(self)

    def eval(self, feed_dict=None):
        raise NotImplementedError

    def auto_derivate(self, trainable_variables: set):
        raise NotImplementedError

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return "%s: <%s>" % (self.tensor_type, self.name)

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        if other == 0:
            return self
        if not isinstance(other, Tensor):
            other = Constant(other)
        return AddOp(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Constant(other)
        return SubOp(self, other)

    def __mul__(self, other):
        if other == 0:
            return Constant(np.zeros(self.shape, self.dtype))
        elif other == 1:
            return self
        elif not isinstance(other, Tensor):
            other = Constant(other)
        return MulOp(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)


class Variable(Tensor):
    def __init__(self, values, dtype=None, trainable: bool=True, name: str=None):
        self.trainable = trainable
        self.values = np.asarray(values, dtype=dtype)
        self.tensor_type = TYPE_VARIABLE
        name = name or "Variable"
        super(Variable, self).__init__(shape=self.values.shape, dtype=self.values.dtype, dependency=None, name=name)

    def eval(self, feed_dict=None):
        if feed_dict and self in feed_dict:
            return feed_dict[self]
        return self.values.copy()

    def auto_derivate(self, trainable_variables: set):
        if self in trainable_variables:
            return {self: 1.0}
        return {}


class Constant(Tensor):
    def __init__(self, values, dtype=None, name: str=None):
        self.values = np.asarray(values, dtype=dtype)
        self.tensor_type = TYPE_CONSTANT
        name = name or "Constant"
        super(Constant, self).__init__(shape=self.values.shape, dtype=self.values.dtype, dependency=None, name=name)

    def eval(self, feed_dict=None):
        if feed_dict and self in feed_dict:
            return feed_dict[self]
        return self.values.copy()

    def auto_derivate(self, trainable_variables: set):
        return {}


class Placeholder(Tensor):
    def __init__(self, shape, dtype=np.float32, name: str=None):
        self.tensor_type = TYPE_PLACEHOLDER
        name = name or "Placeholder"
        super(Placeholder, self).__init__(shape=shape, dtype=dtype, name=name)

    def eval(self, feed_dict=None):
        if not feed_dict or self not in feed_dict:
            raise RuntimeError("This operation requires values fed to placeholder %s" % self.name)
        fed_value = np.asarray(feed_dict[self], dtype=self.dtype)
        if fed_value.shape != self.shape:
            raise TypeError("Placeholder %s is of shape `%s`, but fed value is of shape `%s`" % (self.name, self.shape, fed_value.shape))
        return fed_value


class Op(Tensor):
    gradients = None

    def __init__(self, *args, **kwargs):
        self.tensor_type = TYPE_OPERATION
        super(Op, self).__init__(dependency=Op.get_dependency(self.direct_dependencies), *args, **kwargs)

    def auto_derivate(self, trainable_variables: set):
        to_be_trained = self.dependency & trainable_variables
        gradients = defaultdict(list)
        with self.graph.compute_gradients():
            for tensor in self.gradients:
                this_grad = self.gradients[tensor]
                grads = tensor.auto_derivate(to_be_trained)
                for target, grad in grads.items():
                    gradients[target].append(grad * this_grad)
            for k in gradients:
                gradients[k] = sum(gradients[k])
        return gradients

    @staticmethod
    def get_dependency(list_of_tensors: Set[Tensor]) -> set:
        dependency = set()
        for tensor in list_of_tensors:
            if tensor.tensor_type == TYPE_VARIABLE:
                dependency.add(tensor)
            else:
                dependency |= Op.get_dependency(tensor.dependency)
        return dependency


class AddOp(Op):
    def __init__(self, t1: Tensor, t2: Tensor):
        # if t1.shape != t2.shape:
        #     raise TypeError("shape of %s and %s incompatible: `%s` and `%s`" % (t1.name, t2.name, t1.shape, t2.shape))
        self.t1 = t1
        self.t2 = t2
        self.direct_dependencies = {t1, t2}
        self.gradients = {
            t1: 1.0,
            t2: 1.0,
        }
        super(AddOp, self).__init__(shape=t1.shape, dtype=t1.dtype, name="Add")

    def eval(self, feed_dict=None):
        return self.t1.eval(feed_dict) + self.t2.eval(feed_dict)


class SubOp(Op):
    def __init__(self, t1: Tensor, t2: Tensor):
        # if t1.shape != t2.shape:
        #     raise TypeError("shape of %s and %s incompatible: `%s` and `%s`" % (t1.name, t2.name, t1.shape, t2.shape))
        self.t1 = t1
        self.t2 = t2
        self.direct_dependencies = {t1, t2}
        self.gradients = {
            t1: 1.0,
            t2: -1.0,
        }
        super(SubOp, self).__init__(shape=t1.shape, dtype=t1.dtype, name="Sub")

    def eval(self, feed_dict=None):
        return self.t1.eval(feed_dict) - self.t2.eval(feed_dict)


class MulOp(Op):
    def __init__(self, t1: Tensor, t2: Tensor):
        # if t1.shape != t2.shape:
        #     raise TypeError("shape of %s and %s incompatible: `%s` and `%s`" % (t1.name, t2.name, t1.shape, t2.shape))
        self.t1 = t1
        self.t2 = t2
        self.direct_dependencies = {t1, t2}
        self.gradients = {
            t1: t2,
            t2: t1,
        }
        super(MulOp, self).__init__(shape=t1.shape, dtype=t1.dtype, name="Mul")

    def eval(self, feed_dict=None):
        return self.t1.eval(feed_dict) * self.t2.eval(feed_dict)


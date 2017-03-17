from tensor import Tensor
from variable import Variable


class Op(Tensor):
    @staticmethod
    def get_dependency(list_of_tensors):
        dependency = set()
        for tensor in list_of_tensors:
            if isinstance(tensor, Variable):
                dependency.add(tensor)
            else:
                dependency += Op.get_dependency(tensor.dependency)
        return dependency

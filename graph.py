from variable import Variable
from operations import Op

__graph = None


def get_graph():
    global __graph
    if not __graph:
        __graph = Graph()
    return __graph


class Graph:
    def __init__(self):
        self.tensors = {}
        self.ops = {}
        self.variables = {}

    def add_tensor(self, tensor, name):
        if name in self.tensors:
            raise KeyError("Tensor %s already exists in graph" % name)
        self.tensors[name] = tensor
        if isinstance(tensor, Op):
            self.ops[name] = tensor
        elif isinstance(tensor, Variable):
            self.variables[name] = tensor

    def check_name(self, tensor, name):
        if isinstance(tensor, Variable):
            prefix = "V"
        elif isinstance(tensor, Op):
            prefix = "Op"
        else:
            prefix = "T"
        i = 0
        while 1:
            postfix = str(i)
            valid_name = "_".join([prefix, name, postfix])
            if valid_name not in self.tensors:
                return valid_name
            i += 1


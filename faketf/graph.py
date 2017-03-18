from .constant import TYPE_VARIABLE, TYPE_OPERATION, TYPE_CONSTANT
from contextlib import contextmanager
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
        self.constants = {}
        self.trainable = set()
        self.cache = None
        self.mute = False

    @contextmanager
    def compute_gradients(self):
        self.mute = True
        yield
        self.mute = False

    def add_tensor(self, tensor):
        if tensor.name in self.tensors:
            raise KeyError("Tensor %s already exists in graph" % tensor.name)
        self.tensors[tensor.name] = tensor
        if tensor.tensor_type == TYPE_OPERATION:
            self.ops[tensor.name] = tensor
        elif tensor.tensor_type == TYPE_VARIABLE:
            self.variables[tensor.name] = tensor
            if tensor.trainable:
                self.trainable.add(tensor)
        elif tensor.tensor_type == TYPE_CONSTANT:
            self.constants[tensor.name] = tensor

    def check_name(self, tensor, name):
        if tensor.tensor_type == TYPE_VARIABLE:
            prefix = "V"
        elif tensor.tensor_type == TYPE_OPERATION:
            prefix = "Op"
        elif tensor.tensor_type == TYPE_CONSTANT:
            prefix = "C"
        else:
            prefix = "T"
        if self.mute:
            prefix = "G_" + prefix
        i = 0
        while 1:
            postfix = str(i)
            valid_name = "_".join([prefix, name, postfix])
            if valid_name not in self.tensors:
                return valid_name
            i += 1

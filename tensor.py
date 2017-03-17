from graph import get_graph


class Tensor:
    shape = None
    dtype = None

    def __init__(self, shape, dtype, dependency=None, name=None):
        self.dependency = dependency or set()
        self.shape = shape
        self.dtype = dtype
        self.graph = get_graph()
        self.name = self.graph.check_name(self, name)
        self.graph.add_tensor(self)



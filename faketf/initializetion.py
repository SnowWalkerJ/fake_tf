import numpy as np


class Initializer:
    def generate(self, *shape):
        raise NotImplementedError


class NormalInitializer(Initializer):
    def __init__(self, mu=0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

    def generate(self, *shape):
        return np.random.randn(*shape) * self.sigma + self.mu


class ZeroInitializer(Initializer):
    def generate(self, *shape):
        return np.zeros(shape, dtype=np.float32)

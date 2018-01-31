from .common import Layer

class Inputs(Layer):

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self):
        assert len(self.neurons) == self.size, 'Input layer of size %d but has %d inputs!' % (
            self.size, len(self.neurons))
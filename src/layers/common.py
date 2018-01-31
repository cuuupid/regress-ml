import src.functions.activations as activations

class Layer(object):
    def __init__(self):
        self.neurons = []
        self.weights = []
        self.inputs = []
        self.size = 0
        self.activation_fn = activations.linear
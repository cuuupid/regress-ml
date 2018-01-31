from .common import Layer
from random import random


class FeedForward(Layer):
    def __init__(self, inputs, size, activation_fn=None):
        super().__init__()
        self.size = size
        for neuron in range(size):
            self.neurons.append(0)
            self.weights.append([])
            for neuron in range(inputs.size):
                self.weights[-1].append(random())
        if activation_fn:
            self.activation_fn = activation_fn
        self.inputs = inputs

    def forward(self):
        for neuron in range(self.size):
            self.neurons[neuron] = 0
            for input_neuron, weight in zip(self.inputs.neurons, self.weights[neuron]):
                self.neurons[neuron] += input_neuron * weight
            self.neurons[neuron] = self.activation_fn(self.neurons[neuron])
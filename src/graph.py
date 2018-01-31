from src.functions.loss import sum_error
from random import randint

class Graph(object):
    def __init__(self, name="default"):
        self.name = name
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)

    def show(self):
        print(self.name)
        for layer in self.layers:
            print('Nodes: ', end='\t')
            for neuron in layer.neurons:
                print('%.02f' % neuron, end='  ')
            print()
    
    def forward(self, datapoint):
        self.layers[0].neurons = datapoint['in']
        for layer in self.layers:
            layer.forward()
        return self.layers[-1].neurons

    def train(self, training_data, batch_size=1, loss_fn=sum_error):
        start = randint(0, len(training_data) - batch_size)
        batch = training_data[start:start + batch_size]
        for datapoint in batch:
            outputs = self.forward(datapoint)
            print('Loss: %f' % loss_fn(outputs, datapoint['out']))
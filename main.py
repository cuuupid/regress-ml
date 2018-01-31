from src.layers.inputs import Inputs
from src.layers.feedforward import FeedForward
from src.functions.loss import sum_squared_error
from src.graph import Graph

training_data = []

for x in range(2):
    for y in range(2):
        training_data.append({
            'in': [x, y],
            'out': [1] if not x==y else [0]
        })

batch_size = 1
input_layer = Inputs(2)
hidden_layer = FeedForward(input_layer, 3)
output_layer = FeedForward(hidden_layer, 1)

graph = Graph()
graph.add_layer(input_layer)
graph.add_layer(hidden_layer)
graph.add_layer(output_layer)

graph.train(training_data, loss_fn=sum_squared_error)
graph.show()
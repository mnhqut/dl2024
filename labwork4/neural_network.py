
import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dot_product(vec1, vec2):
  return sum(x * y for x, y in zip(vec1, vec2))

def one():
    return 1

class Neuron:
    def __init__(self,id = None, value = random.random(), activation_func = sigmoid):
        # id should be a tuple indexing the neuron in each layer
        self.id = id
        self.value = value
        self.activation_func = activation_func

    # def activate(self, inputs):
    #     self.value = sum(inputs) + self.bias
    #     self.activation = self.sigmoid(self.value)
    #     return self.activation

# special class of neuron that will always take value 1 and will not get attach to any neuron in the previous layer
class Bias(Neuron):
    def __init__(self,id = None, value = 1, activation_func = one):
        # id should be a tuple indexing the neuron in each layer
        self.id = id
        self.value = value
        self.activation_func = activation_func
    
    
class Layer:
    def __init__(self, id, size):
        self.id = id
        # each layer give it a bias neuron first, even for the ouput layer for convenience
        self.neurons = [Bias((self.id,0))]
        count = 1
        for neuron in range(size-1):
            self.neurons.append(Neuron((self.id,count)))
            count += 1

    # simply perform dot product to use as inputs of next layers 
    def forward(self, inputs,layer_link):
        output = []
        val_neuron = [neuron.value for neuron in self.neurons]
        for links_list in layer_link:
            output.append(dot_product(val_neuron, links_list)) 
        return output 

    def __len__(self):
        return len(self.neurons)


class Link:
    def __init__(self,source,desti,weight = random.random()):
        # source and destination should be the pointer to neurons
        self.source = source
        self.desti = desti
        self.weight = weight

    # def forward(self, input_value):
    #     return input_value * self.weight

class LayerLink:
    def __init__(self, from_layer, to_layer):
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.links = []
        # the structure of links should be consist of list of ingoing links each iteration of the from_layer
        # this make it convenient for forward() later
        for i in len(to_layer):
            nodelinks = []
            for j in len(from_layer):
                nodelinks.append(Link(from_layer.neurons[j],to_layer.neurons[i]))
            self.links.append(nodelinks)

    # def forward(self):
    #     outputs = []
    #     for j in range(len(self.to_layer)):
    #         inputs = []
    #         for i in range(len(self.from_layer)):
    #             inputs.append(self.links[i][j].forward(self.from_layer[i].activation))
    #         self.to_layer.neurons[j].activate(inputs)
    #         outputs.append(self.to_layer.neurons[j].activation)
    #     return outputs


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.depth = 0
        self.layer_links = []
    def add_layer(self, size):
        if self.depth == 0:
            # Input layer
            self.layers.append(Layer(self.depth,size))
        else:
            # Hidden layers and output layer
            new_layer = Layer(size)
            # this is the layer_link of that layer to the previous
            layer_link = LayerLink(self.layers[-1], new_layer)
            self.layers_links.append(layer_link)
            self.layers.append(new_layer)
            

    def forward(self, inputs):
        # check valid input
        if len(inputs) != len(self.layers[0]):
            print("invalid input")
            
        current_inputs = inputs
        for layer in self.layers:
            current_inputs = layer.forward(current_inputs,self.layer_links[layer.id])
            
        return current_inputs

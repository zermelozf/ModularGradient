'''
Created on Dec 22, 2011

@author: arnaud
'''
import numpy
from utils import sparseReservoirMatrix

def identity(x):
    return x

class Node:
    def __init__(self, input_dim=None, output_dim=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def _execute(self, input):
        pass
    
    def execute(self, input):
        return self._execute(input)

class FeaturesNode(Node):
    def __init__(self, input_dim, output_dim):
        Node.__init__(self, input_dim=input_dim, output_dim=output_dim)  
        self.W = numpy.random.rand(input_dim, output_dim)
        
        self.state = None
        self.params = [self.W]
       
    def _execute(self, input):
        return numpy.dot(input, self.W)

class ReservoirNode(Node):
    def __init__(self, input_dim, output_dim, activation = numpy.tanh, dtype=None):
        Node.__init__(self, input_dim=input_dim, output_dim=output_dim)  
        self.activation = activation

        self.W_in = 1. * (numpy.random.randint(0, 2, (input_dim, output_dim)) * 2 - 1)
        self.b = numpy.zeros((output_dim,)) 
        self.W = sparseReservoirMatrix((output_dim, output_dim), 0.27)
        
        self.state = numpy.zeros((output_dim,))
        
        self.params = [self.W_in, self.W, self.b]
        
    def _execute(self, input):
        self.state = self.activation(numpy.dot(input, self.W_in) + self.b + numpy.dot(self.state, self.W))
        return self.state

class HiddenLayerNode(Node):
    def __init__(self, input_dim, output_dim, activation = numpy.tanh, dtype=None):
        Node.__init__(self, input_dim=input_dim, output_dim=output_dim)  
        self.activation = activation

        self.W = numpy.asarray( numpy.random.uniform(
                low  = - numpy.sqrt(6./(input_dim+output_dim)),
                high = numpy.sqrt(6./(input_dim+output_dim)),
                size = (input_dim, output_dim)))

        self.b = numpy.zeros((output_dim,))

        self.params = [self.W, self.b]
        
    def _execute(self, input):
        return self.activation(numpy.dot(input, self.W) + self.b)

class PerceptronNode(HiddenLayerNode):
    def __init__(self, input_dim, output_dim, activation=identity, dtype=None):
        HiddenLayerNode.__init__(self, input_dim=input_dim, output_dim=output_dim, activation=identity, dtype=dtype)

class SoftMaxNode(Node):
    def _execute(self, input):
        input = numpy.exp(input)
        return input/numpy.array([numpy.sum(input, axis=1)]).T
    
class FlowNode(Node):
    def __init__(self, nodes_list):
        self.nodes_list = nodes_list
        self.layer_state = []
        
    def _execute(self, input):
        tampon = input
        self.layer_states = []
        for node in self.nodes_list:
            tampon = node.execute(tampon)
            self.layer_states.append(tampon)
        return tampon
    
if __name__=='__main__':
    
    feat = FeaturesNode(input_dim=10, output_dim=5)
    res = ReservoirNode(input_dim=feat.output_dim, output_dim=50)
    hid = HiddenLayerNode(input_dim=res.output_dim, output_dim=10)
    per = PerceptronNode(input_dim=hid.output_dim, output_dim=5)
    sof = SoftMaxNode(input_dim=per.output_dim)
    
    flow = [feat, res, hid, per, sof]
    flownode = FlowNode(flow)
    
    input = numpy.random.rand(3, 10)
    print input 
    print flownode.execute(input)
    print flownode.nodes_list
    for layer in flownode.layer_states:
        print layer.shape
        
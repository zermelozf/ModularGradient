'''
Created on Jan 12, 2012

@author: arnaud
'''
import numpy

class identity:
    def __call__(self, x):
        return x
    
    def derivative(self, x):
        return 1
    
class tanh:
    def __call__(self, x):
        return numpy.tanh(x)
    
    def derivative(self, x):
        return 1 - numpy.tanh(x)**2

class Node:
    def __init__(self, input_dim=None, output_dim=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def _execute(self, input):
        pass
    
    def execute(self, input):
        return self._execute(input)
    
    def is_trainable(self):
        return True

class InputNode(Node):
    def __init__(self, input_dim):
        Node.__init__(self, input_dim=input_dim, output_dim=input_dim)
        self.activation = identity()
        
    def _execute(self, input):
        self.state = input
        return self.state
    
    def is_trainable(self):
        return False
    
class HiddenLayerNode(Node):
    def __init__(self, input_dim, output_dim, activation = tanh(), dtype=None):
        Node.__init__(self, input_dim=input_dim, output_dim=output_dim)  
        self.activation = activation

        self.W = numpy.asarray( numpy.random.uniform(
                low  = - numpy.sqrt(6./(input_dim+output_dim)),
                high = numpy.sqrt(6./(input_dim+output_dim)),
                size = (input_dim, output_dim)))
        
        self.b = numpy.zeros((output_dim,))
        
    def _execute(self, input):
        self.state = self.activation(numpy.dot(input, self.W) + self.b)
        return self.state

class FlowNode(Node):
    def __init__(self, nodes_list):
        self.nodes_list = nodes_list
        
    def _execute(self, input):
        tampon = input
        for node in self.nodes_list:
            tampon = node.execute(tampon)
        return tampon
    
class GradientDescentModule:
    def __init__(self, flownode):
        self.flownode = flownode
    
    def update(self, input, target):
        self.flownode.execute(input)
        self._update(target)
     
    def _update(self, target):
        lr = 0.1
        error = self.flownode.nodes_list[-1].state - target
        for i in range(len(self.flownode.nodes_list)-1, 0, -1):
            node = self.flownode.nodes_list[i]
            pnode = self.flownode.nodes_list[i-1]
            if node.is_trainable():
                node.W -= lr*numpy.dot(numpy.array([pnode.state]).T, numpy.array([error*node.activation.derivative(node.state)]))
                try:
                    node.b -= lr*error*node.activation.derivative(node.state)
                except:
                    pass
            error = numpy.dot(error*node.activation.derivative(node.state), node.W.T)

if __name__=='__main__':
    
    inp = InputNode(input_dim=2)
    hid1 = HiddenLayerNode(input_dim=inp.output_dim, output_dim=2)
    lin = HiddenLayerNode(input_dim=hid1.output_dim, output_dim=1, activation=identity())
    
    flow = [inp, hid1, lin]
    flownode = FlowNode(flow)
    gradmod = GradientDescentModule(flownode)
    
    X = [[0,0],[0,1],[1,0],[1,1]]
    Y = [0,1,1,0]
    
    print flownode.execute(numpy.array(X))
    
    for t in range(10000):
        for i in range(0, len(X)):
            gradmod.update(numpy.array(X[i]),numpy.array(Y[i]))
        
    print flownode.execute(numpy.array(X))

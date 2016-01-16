import theano
import theano.tensor as T
import numpy as np
import numpy.random as rng

class HiddenLayer: 

    def __init__(self, input, num_in, num_out, initialization, name, activation = None, paramMap = None): 

        if paramMap == None: 

            self.params = {}

            W_values = np.asarray(0.01 * rng.standard_normal(size=(num_in, num_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name=name + "_W")
        
            b_values = np.zeros((num_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name= name + '_b')

            self.params[name + "_W"] = W
            self.params[name + "_b"] = b

        else: 
            W = paramMap[name + "_W"]
            b = paramMap[name + "_b"]

        lin_output = T.dot(input, W) + b


        self.out_store = lin_output

        if activation == None: 
            activation = lambda x: x
        elif activation == "relu": 
            activation = lambda x: T.maximum(0.0, x)
        elif activation == "exp": 
            activation = lambda x: T.exp(x)
        elif activation == "tanh":
            activation = lambda x: T.tanh(x)
        else: 
            raise Exception("Activation not found")

        self.output = activation(lin_output)



    def getParams(self): 
        return self.params




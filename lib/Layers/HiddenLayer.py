import theano
import theano.tensor as T
import numpy as np
import numpy.random as rng

class HiddenLayer: 

    def __init__(self, input, num_in, num_out, initialization, name, activation = None, paramMap = None, batch_norm = True): 

        if paramMap == None: 

            self.params = {}

            std = np.sqrt(2.0 / (num_out + num_in))

            print "full layer with name", name, "using std of", std

            W_values = 1.0 * np.asarray(rng.normal(size=(num_in, num_out), scale = std), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name=name + "_W")
        
            b_values = np.zeros((num_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name= name + '_b')

            bn_mean_values = 1.0 * np.zeros(shape = (1, num_out)).astype('float32')
            bn_std_values = 1.0 * np.ones(shape = (1,num_out)).astype('float32')

            bn_mean = theano.shared(value=bn_mean_values, name = name + "_mu")
            bn_std = theano.shared(value = bn_std_values, name = name + "_sigma")

            self.params[name + "_W"] = W
            self.params[name + "_b"] = b

            if batch_norm:
                self.params[name + "_mu"] = bn_mean
                self.params[name + "_sigma"] = bn_std

        else: 
            W = paramMap[name + "_W"]
            b = paramMap[name + "_b"]
            if batch_norm:
                bn_mean = paramMap[name + "_mu"]
                bn_std = paramMap[name + "_sigma"]

        lin_output = T.dot(input, W) + b

        #N x M
        if batch_norm:
            lin_output = (lin_output - T.mean(lin_output, axis = 0, keepdims = True)) / (1.0 + T.std(lin_output, axis = 0, keepdims = True))
            lin_output = (lin_output * T.addbroadcast(bn_std,0) + T.addbroadcast(bn_mean,0))

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




import theano
import theano.tensor as T
import numpy as np
import numpy.random as rng

class HiddenLayer: 

    def __init__(self, num_in, num_out, activation = None, batch_norm = False): 

        self.params = {}
        self.activation = activation
        self.batch_norm = batch_norm

        std = np.sqrt(2.0 / (num_out + num_in))


        W_values = 1.0 * np.asarray(rng.normal(size=(num_in, num_out), scale = std), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values)

        b_values = np.zeros((num_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values)

        bn_mean_values = 1.0 * np.zeros(shape = (1, num_out)).astype('float32')
        bn_std_values = 1.0 * np.ones(shape = (1,num_out)).astype('float32')

        self.bn_mean = theano.shared(value=bn_mean_values)
        self.bn_std = theano.shared(value = bn_std_values)

        self.params["W"] = self.W
        self.params["b"] = self.b

        if batch_norm:
            self.params["mu"] = self.bn_mean
            self.params["sigma"] = self.bn_std

    def output(self, input):

        lin_output = T.dot(input, self.W) + self.b

        if self.batch_norm:
            lin_output = (lin_output - T.mean(lin_output, axis = 0, keepdims = True)) / (1.0 + T.std(lin_output, axis = 0, keepdims = True))
            lin_output = (lin_output * T.addbroadcast(self.bn_std,0) + T.addbroadcast(self.bn_mean,0))

        self.out_store = lin_output

        if self.activation == None: 
            activation = lambda x: x
        elif self.activation == "relu": 
            activation = lambda x: T.maximum(0.0, x)
        elif self.activation == "exp": 
            activation = lambda x: T.exp(x)
        elif self.activation == "tanh":
            activation = lambda x: T.tanh(x)
        else: 
            raise Exception("Activation not found")

        return activation(lin_output)


    def getParams(self): 
        return self.params




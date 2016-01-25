import sys
import numpy as np
import theano
import theano.tensor as T

'''

(mb, channels, 32, 32)

'''

class Upsample():

    def __init__(self, output_shape):
        self.output_shape = output_shape

    def output(self, x):
        y = T.alloc(0.0, self.output_shape[0], self.output_shape[1], self.output_shape[2], self.output_shape[3])

        y = T.set_subtensor(y[:,:,0::2, 0::2], x)
        y = T.set_subtensor(y[:,:,0::2, 1::2], x)
        y = T.set_subtensor(y[:,:,1::2, 0::2], x)
        y = T.set_subtensor(y[:,:,1::2, 1::2], x)

        return y

    def getParams(self):
        return []


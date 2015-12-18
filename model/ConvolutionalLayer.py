import sys

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn
from theano.sandbox.cuda.basic_ops import gpu_contiguous

import warnings
warnings.filterwarnings("ignore")

rng = np.random.RandomState(23455)
# set a fixed number for 2 purpose:
#  1. repeatable experiments; 2. for multiple-GPU, the same initial weights


class Weight(object):

    def __init__(self, w_shape, mean=0, std=1.0, name = ""):
        super(Weight, self).__init__()
        if std != 0:
            self.np_values = np.asarray(
                rng.normal(mean, std, w_shape) * 0.01, dtype=theano.config.floatX)
        else:
            self.np_values = np.cast[theano.config.floatX](
                mean * np.ones(w_shape, dtype=theano.config.floatX))

        self.val = theano.shared(value=self.np_values, name = name)

    def save_weight(self, dir, name):
        print 'weight saved: ' + name
        np.save(dir + name + '.npy', self.val.get_value())

    def load_weight(self, dir, name):
        print 'weight loaded: ' + name
        self.np_values = np.load(dir + name + '.npy')
        self.val.set_value(self.np_values)



class ConvPoolLayer(object):

    def __init__(self, input, in_channels, out_channels, kernel_len, in_rows, in_columns, batch_size, convstride, padsize, poolsize, poolstride, bias_init, name):

        self.filter_shape = np.asarray((in_channels, kernel_len, kernel_len, out_channels))
        self.image_shape = np.asarray((in_channels, in_rows, in_columns, batch_size))

        self.W = Weight(self.filter_shape, name = name + "_W")
        self.b = Weight(self.filter_shape[3], bias_init, std=0, name = name + "_b")

        #Input: Batch, rows, columns, channels
        #Output: Batch, channels, rows, columns
        input_shuffled = input.dimshuffle(0, 3, 1, 2)  # c01b to bc01
            # in01out to outin01
            # print image_shape_shuffled
            # print filter_shape_shuffled
        W_shuffled = self.W.val.dimshuffle(3, 0, 1, 2)  # c01b to bc01
        conv_out = dnn.dnn_conv(img=input_shuffled,
                                        kerns=W_shuffled,
                                        subsample=(convstride, convstride),
                                        border_mode=padsize,
                                        )
        conv_out = conv_out + self.b.val.dimshuffle('x', 0, 'x', 'x')

        self.output = T.maximum(0.0, conv_out)

        if poolsize != 1:
            self.output = dnn.dnn_pool(self.output,ws=(poolsize, poolsize),stride=(poolstride, poolstride))

        self.output = self.output.dimshuffle(0, 2, 3, 1)

        self.params = {name + '_W' : self.W.val, name + '_b' : self.b.val}

    def getParams(self):
        return self.params


if __name__ == "__main__":

    x = T.tensor4()

    randData = np.random.uniform(size = (100,32,32,3)).astype('float32')



    c1 = ConvPoolLayer(input=x, in_channels = 3, out_channels = 96, kernel_len = 5, in_rows = 32, in_columns = 32, batch_size = 100,
                                        convstride=1, padsize=4, 
                                        poolsize=3, poolstride=2, 
                                        bias_init=0.0, name = "h1"
                                        )

    c2 = ConvPoolLayer(input=c1.output, in_channels = 96, out_channels = 128, kernel_len = 3, in_rows = 17, in_columns = 17, batch_size = 100,
                                        convstride=1, padsize=3,
                                        poolsize=3, poolstride=2,
                                        bias_init=0.0, name = "h2"
                                        )

    c3 = ConvPoolLayer(input=c2.output, in_channels = 128, out_channels = 128, kernel_len = 3, in_rows = 10, in_columns = 10, batch_size = 100,
                                        convstride=1, padsize=0,
                                        poolsize=3, poolstride=2,
                                        bias_init=0.0, name = "h3"
                                        )


    y = c3.output

    g = T.sum(T.grad(T.sum(y), c1.getParams()["h1_W"]))

    f = theano.function(inputs = [x], outputs = {'y' : y, 'g' : g})

    #print f(randData)['g']
    print f(randData)['y'].shape



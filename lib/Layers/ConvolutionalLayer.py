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

    def __init__(self, w_shape, mean=0, std=1.0, name = "", mode = ""):
        super(Weight, self).__init__()
        if std != 0:

            print "conv layer with name", name, "using std of", std
            self.np_values = np.asarray(
               1.0 * rng.normal(mean, std, w_shape), dtype=theano.config.floatX)

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

    def __init__(self, input, in_channels, out_channels, kernel_len, in_rows, in_columns, batch_size, convstride, padsize, poolsize, poolstride, bias_init, name, paramMap, activation = "relu", batch_norm = False):

        std = 0.02
        self.filter_shape = np.asarray((in_channels, kernel_len, kernel_len, out_channels))

        self.W = Weight(self.filter_shape, name = name + "_W", std = std, mode = 'conv')
        self.b = Weight(self.filter_shape[3], bias_init, std=0, name = name + "_b", mode = 'conv')
        self.R = Weight((in_channels,1,1,out_channels), name = name + "_R", mean = 0.01, std = std, mode = 'conv')

        if batch_norm:
            self.bn_mean = theano.shared(np.zeros(shape = (1,out_channels,1,1)).astype('float32'))
            self.bn_std = theano.shared(np.random.normal(1.0, 0.001, size = (1,out_channels,1,1)).astype('float32'))

        if paramMap != None:
            print "shapes"
            print paramMap[name + "_W"].get_value().shape
            print self.W.val.get_value().shape
            self.W.val.set_value(paramMap[name + "_W"].get_value())
            self.b.val.set_value(paramMap[name + "_b"].get_value())

        #Input: Batch, rows, columns, channels
        #Output: Batch, channels, rows, columns
        #input_shuffled = input.dimshuffle(0, 3, 1, 2)  # c01b to bc01

        input_shuffled = input

            # in01out to outin01
            # print image_shape_shuffled
            # print filter_shape_shuffled
        W_shuffled = self.W.val.dimshuffle(3, 0, 1, 2)  # c01b to bc01
        R_shuffled = self.R.val.dimshuffle(3,0,1,2)

        conv_out = dnn.dnn_conv(img=input_shuffled,
                                        kerns=W_shuffled,
                                        subsample=(convstride, convstride),
                                        border_mode=padsize,
                                        )

        conv_out_residual = dnn.dnn_conv(img=input_shuffled,kerns=R_shuffled,subsample=(convstride,convstride),border_mode=0)

        if not batch_norm:
            conv_out = conv_out + self.b.val.dimshuffle('x', 0, 'x', 'x')
        else:
            conv_out = conv_out + self.b.val.dimshuffle('x', 0, 'x', 'x')

        if batch_norm:
            conv_out = (conv_out - T.mean(conv_out, axis = (0,2,3), keepdims = True)) / (1.0 + T.std(conv_out, axis=(0,2,3), keepdims = True))
            conv_out = conv_out * T.addbroadcast(self.bn_std,0,2,3) + T.addbroadcast(self.bn_mean, 0,2,3)

        self.out_store = conv_out

        if activation == "relu":
            self.output = T.maximum(0.0, conv_out) + conv_out_residual
        elif activation == "tanh":
            self.output = T.tanh(conv_out)
        elif activation == None:
            self.output = conv_out

        if poolsize != 1:
            self.output = dnn.dnn_pool(self.output,ws=(poolsize, poolsize),stride=(poolstride, poolstride))


        self.params = {name + '_W' : self.W.val, name + '_b' : self.b.val, name + "_R" : self.R.val}

        if batch_norm:
            self.params[name + "_mu"] = self.bn_mean
            self.params[name + "_sigma"] = self.bn_std

    def getParams(self):
        return self.params


if __name__ == "__main__":

    x = T.tensor4()

    randData = np.random.normal(size = (1,256,256,3)).astype('float32')

    c1 = ConvPoolLayer(input=x.dimshuffle(0,3,1,2), in_channels = 3, out_channels = 96, kernel_len = 5, in_rows = 256, in_columns = 256, batch_size = 100,
                                        convstride=2, padsize=2, 
                                        poolsize=1, poolstride=0,
                                        bias_init=0.1, name = "h2", paramMap = None
                                        )

    c2 = ConvPoolLayer(input=c1.output, in_channels = 96, out_channels = 256, kernel_len = 5, in_rows = 17, in_columns = 17, batch_size = 100,
                                        convstride=2, padsize=2,
                                        poolsize=1, poolstride=0,
                                        bias_init=0.0, name = "c2", paramMap = None
                                        )


    c3 = ConvPoolLayer(input=c2.output, in_channels = 256, out_channels = 384, kernel_len = 5, in_rows = 33, in_columns = 33, batch_size = 100,
                                        convstride=2, padsize=2,
                                        poolsize=1, poolstride=0,
                                        bias_init=0.0, name = "h3", paramMap = None
                                        )

    c4 = ConvPoolLayer(input=c3.output, in_channels = 384, out_channels = 384, kernel_len = 5, in_rows = 15, in_columns = 15, batch_size = 100,
                                        convstride=2, padsize=2,
                                        poolsize=1, poolstride=0,
                                        bias_init=0.1, name = "h3", paramMap = None
                                        )

    c5 = ConvPoolLayer(input=c4.output, in_channels = 384, out_channels = 256, kernel_len = 5, in_rows = 6, in_columns = 6, batch_size = 100,
                                        convstride=2, padsize=2,
                                        poolsize=1, poolstride=0,
                                        bias_init=0.0, name = "h3", paramMap = None
                                        )

    y = c5.output


    f = theano.function(inputs = [x], outputs = {'y' : y, 'c1' : c1.output.transpose(0,2,3,1), 'c2' : c2.output.transpose(0,2,3,1), 'c3' : c3.output.transpose(0,2,3,1), 'c4' : c4.output.transpose(0,2,3,1), 'c5' : c5.output.transpose(0,2,3,1)})

    #print f(randData)['g']
    out = f(randData)


    print (randData**2).sum()
    print (out['c1']**2).sum()
    print (out['c2']**2).sum()
    print (out['c3']**2).sum()
    print (out['c4']**2).sum()
    print (out['c5']**2).sum()

    for element in sorted(out.keys()):
        print element, out[element].shape


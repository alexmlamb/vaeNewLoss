
import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradI, dnn_conv, dnn_pool
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np

from ConvolutionalLayer import Weight

def deconv(X, w, subsample=(1, 1), border_mode=(0, 0), conv_mode='conv'):
    img = gpu_contiguous(X)
    kerns = gpu_contiguous(w)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample, conv_mode=conv_mode)(gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1]).shape, kerns.shape)
    out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1])
    d_img = GpuDnnConvGradI()(kerns, img, out, desc)
    return d_img


class DeConvLayer(object):

    def __init__(self, input, in_channels, out_channels, kernel_len, in_rows, in_columns, batch_size, bias_init, name, paramMap, activation, upsample_rate):

        '''

        Need to get to:
            start with:
                image: 
                batch_size, in_row, in_column, in_channel
                go to:
                batch_size, in_channel, in_row, in_column
                then output goes back to:
                batch_size, in_row, in_column, in_channel

                x = np.random.uniform(size = (mb_size,channels_in,x_in,y_in)).astype('float32')
                w = np.random.uniform(size = (channels_in,channels_out,kernel_len,kernel_len)).astype('float32')
        '''

        self.filter_shape = np.asarray((in_channels, out_channels, kernel_len, kernel_len))
        self.image_shape = np.asarray((in_channels, in_rows, in_columns, batch_size))

        if paramMap == None:
            self.W = Weight(self.filter_shape, name = name + "_W").val
            self.b = Weight(self.filter_shape[1], bias_init, std=0, name = name + "_b").val
        else:
            self.W = paramMap[name + "_W"]
            self.b = paramMap[name + "_b"]

        #Input: Batch, rows, columns, channels
        #Output: Batch, channels, rows, columns
        input_shuffled = input.dimshuffle(0, 3, 1, 2)  # c01b to bc01

        conv_out = deconv(input_shuffled, self.W, subsample=(upsample_rate, upsample_rate), border_mode=(2, 2))

        conv_out = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        self.out_store = conv_out
        self.out_store = self.out_store.dimshuffle(0,2,3,1)

        if activation == "relu":
            self.output = T.maximum(0.0, conv_out)
        elif activation == "tanh":
            self.output = T.tanh(conv_out)
        else:
            raise Exception()

        self.output = self.output.dimshuffle(0, 2, 3, 1)

        self.params = {name + '_W' : self.W, name + '_b' : self.b}

    def getParams(self):
        return self.params


if __name__ == "__main__":

    x = T.tensor4()


    dc = DeConvLayer(input = x, in_channels = 1, out_channels = 3, kernel_len = 5, in_rows = 2, in_columns = 2, batch_size = 1, bias_init = 0.0, name = "c1", paramMap = None, activation = 'tanh', upsample_rate = 2)

    f = theano.function(inputs = [x], outputs = {'o' : dc.output, 'g' : T.grad(T.sum(dc.output), dc.W)})

    x = np.random.uniform(size = (1,2,2,1)).astype('float32') * 100.0

    print f(x)['o'].shape



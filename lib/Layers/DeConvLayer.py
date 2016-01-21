
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

from PIL import Image



def deconv(X, w, subsample=(1, 1), border_mode=(0, 0), conv_mode='conv'):
    img = gpu_contiguous(X)
    kerns = gpu_contiguous(w)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample, conv_mode=conv_mode)(gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1]).shape, kerns.shape)
    out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1])
    d_img = GpuDnnConvGradI()(kerns, img, out, desc)
    return d_img


class DeConvLayer(object):

    def __init__(self, in_channels, out_channels, kernel_len, activation, batch_norm = False, unflatten_input = None):

        self.filter_shape = np.asarray((in_channels, out_channels, kernel_len, kernel_len))

        self.activation = activation

        self.unflatten_input = unflatten_input

        self.batch_norm = batch_norm

        std = 0.02
        self.W = Weight(self.filter_shape, std = std).val
        self.b = Weight(self.filter_shape[1], 0.0, std=0).val
        if batch_norm:
            self.bn_mean = theano.shared(np.zeros(shape = (1,out_channels,1,1)).astype('float32'))
            self.bn_std = theano.shared(np.random.normal(1.0, 0.000001, size = (1,out_channels,1,1)).astype('float32'))

    def output(self, input):

        if self.unflatten_input != None:
            input = T.reshape(input, self.unflatten_input)

        border_mode = (2,2)

        conv_out = deconv(input, self.W, subsample=(2, 2), border_mode=border_mode)

        conv_out = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        if self.batch_norm:
            conv_out = (conv_out - conv_out.mean(axis = (0,2,3), keepdims = True)) / (1.0 + conv_out.std(axis = (0,2,3), keepdims = True))
            conv_out = conv_out * T.addbroadcast(self.bn_std,0,2,3) + T.addbroadcast(self.bn_mean,0,2,3)

        if self.activation == "relu":
            out = T.maximum(0.0, conv_out)
        elif self.activation == "tanh":
            out = T.tanh(conv_out)
        elif self.activation == None:
            out = conv_out
        else:
            raise Exception()


        self.params = {'W' : self.W, 'b' : self.b}
        if self.batch_norm:
            self.params["mu"] = self.bn_mean
            self.params["sigma"] = self.bn_std

        return out

    def getParams(self):
        return self.params


if __name__ == "__main__":

    x = T.tensor4()

    #(4,4,512)

    w = theano.shared(0.01 * np.random.normal(size = (100,4,4,2048)).astype('float32'), name = 'w')




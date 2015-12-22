
import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradI, dnn_conv, dnn_pool
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np

if __name__ == "__main__":

    def deconv(X, w, subsample=(1, 1), border_mode=(0, 0), conv_mode='conv'):
        img = gpu_contiguous(X)
        kerns = gpu_contiguous(w)
        desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)(gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1]).shape, kerns.shape)
        out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1])
        d_img = GpuDnnConvGradI()(kerns, img, out, desc)
        return d_img

    x = T.tensor4()
    w = T.tensor4()

    f = theano.function(inputs = [x,w], outputs = {'o' : deconv(x, w, subsample = (2,2))})

    mb_size = 100
    channels_in = 10
    channels_out = 50
    kernel_len = 5
    x_in = 20
    y_in = 30

    x = np.random.uniform(size = (mb_size,channels_in,x_in,y_in)).astype('float32')
    w = np.random.uniform(size = (channels_in,channels_out,kernel_len,kernel_len)).astype('float32')

    print f(x,w)['o'].shape

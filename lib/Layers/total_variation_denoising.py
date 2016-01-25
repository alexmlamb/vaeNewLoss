import theano
import numpy as np
import theano.tensor as T

#x is a (128, 3, 32, 32) object.  
def total_denoising_variation_penalty(x):

    #out, in, filter1, filter2
    W_v = theano.shared(np.asarray([[[[1], [-1]]]]).astype('float32'))
    W_h = theano.shared(np.asarray([[[[1,-1]]]]).astype('float32'))


    out = 0.0

    out += T.sum(T.abs_(theano.tensor.nnet.conv.conv2d(x, W_v)))
    out += T.sum(T.abs_(theano.tensor.nnet.conv.conv2d(x, W_v)))
    out += T.sum(T.abs_(theano.tensor.nnet.conv.conv2d(x, W_v)))

    out += T.sum(T.abs_(theano.tensor.nnet.conv.conv2d(x, W_h)))
    out += T.sum(T.abs_(theano.tensor.nnet.conv.conv2d(x, W_h)))
    out += T.sum(T.abs_(theano.tensor.nnet.conv.conv2d(x, W_h)))

    return out



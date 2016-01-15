
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

from Classifiers.imagenet_classifier import get_overfeat_diff


def deconv(X, w, subsample=(1, 1), border_mode=(0, 0), conv_mode='conv'):
    img = gpu_contiguous(X)
    kerns = gpu_contiguous(w)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample, conv_mode=conv_mode)(gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1]).shape, kerns.shape)
    out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1])
    d_img = GpuDnnConvGradI()(kerns, img, out, desc)
    return d_img


class DeConvLayer(object):

    def __init__(self, input, in_channels, out_channels, kernel_len, in_rows, in_columns, batch_size, bias_init, name, paramMap, activation, upsample_rate, batch_norm = False):

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

        input_shuffled = input

        self.filter_shape = np.asarray((in_channels, out_channels, kernel_len, kernel_len))
        self.image_shape = np.asarray((in_channels, in_rows, in_columns, batch_size))

        if paramMap == None:
            std = 0.02
            self.W = Weight(self.filter_shape, name = name + "_W", mode = 'deconv', std = std).val
            self.b = Weight(self.filter_shape[1], bias_init, std=0, name = name + "_b", mode = 'deconv').val
            #self.R = Weight(self.filter_shape, name = name + "_R", mode = 'deconv', mean = 0.01, std = std).val
            if batch_norm:
                self.bn_mean = theano.shared(np.zeros(shape = (1,out_channels,1,1)).astype('float32'), name = name + "_bn_mean")
                self.bn_std = theano.shared(np.random.normal(1.0, 0.000001, size = (1,out_channels,1,1)).astype('float32'), name = name + "_bn_std")
        else:
            self.W = paramMap[name + "_W"]
            self.b = paramMap[name + "_b"]
            #self.R = paramMap[name + "_R"]
            if batch_norm:
                self.bn_mean = paramMap[name + "_mu"]
                self.bn_std = paramMap[name + "_sigma"]

        #Input: Batch, rows, columns, channels
        #Output: Batch, channels, rows, columns
        #input_shuffled = input.dimshuffle(0, 3, 1, 2)  # c01b to bc01

        border_mode = (2,2)

        conv_out = deconv(input_shuffled, self.W, subsample=(upsample_rate, upsample_rate), border_mode=border_mode)
        #residual_out = deconv(input_shuffled, self.R, subsample=(upsample_rate, upsample_rate), border_mode=border_mode)

        conv_out = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        if batch_norm:
            conv_out = (conv_out - conv_out.mean(axis = (0,2,3), keepdims = True)) / (1.0 + conv_out.std(axis = (0,2,3), keepdims = True))
            conv_out = conv_out * T.addbroadcast(self.bn_std,0,2,3) + T.addbroadcast(self.bn_mean,0,2,3)

        self.out_store = conv_out
        self.out_store = self.out_store.dimshuffle(0,2,3,1)

        if activation == "relu":
            self.output = T.maximum(0.0, conv_out)
        elif activation == "tanh":
            self.output = T.tanh(conv_out)
        elif activation == None:
            self.output = conv_out
        else:
            raise Exception()


        self.params = {name + '_W' : self.W, name + '_b' : self.b}#, name + "_R" : self.R}
        if batch_norm:
            self.params[name + "_mu"] = self.bn_mean
            self.params[name + "_sigma"] = self.bn_std

    def getParams(self):
        return self.params


if __name__ == "__main__":

    x = T.tensor4()

    #(4,4,512)

    w = theano.shared(0.01 * np.random.normal(size = (100,4,4,2048)).astype('float32'), name = 'w')

    from ConvolutionalLayer import ConvPoolLayer

    layers = []

    dc1 = DeConvLayer(input = w.transpose(0,3,1,2), in_channels = 2048, out_channels = 1024, kernel_len = 5, in_rows = 4, in_columns = 4, batch_size = 1, bias_init = 0.0, name = "c1", paramMap = None, activation = 'relu', upsample_rate = 2)
    
    layers += [dc1]

    dc2 = DeConvLayer(input = layers[-1].output, in_channels = 1024, out_channels = 512, kernel_len = 5, in_rows = 8, in_columns = 8, batch_size = 1, bias_init = 0.0, name = "c2", paramMap = None, activation = 'relu', upsample_rate = 2)
    
    layers += [dc2]

    dc3 = DeConvLayer(input = layers[-1].output, in_channels = 512, out_channels = 256, kernel_len = 5, in_rows = 16, in_columns = 16, batch_size = 1, bias_init = 0.0, name = "c3", paramMap = None, activation = 'relu', upsample_rate = 2)
    
    layers += [dc3]

    dc4 = DeConvLayer(input = layers[-1].output, in_channels = 256, out_channels = 128, kernel_len = 5, in_rows = 32, in_columns = 32, batch_size = 1, bias_init = 0.0, name = "c4", paramMap = None, activation = 'relu', upsample_rate = 2)

    layers += [dc4]
    
    dc5 = DeConvLayer(input = layers[-1].output, in_channels = 128, out_channels = 64, kernel_len = 5, in_rows = 64, in_columns = 64, batch_size = 1, bias_init = 0.0, name = "c5", paramMap = None, activation = 'relu', upsample_rate = 2)

    layers += [dc5]
    #dc7 = ConvPoolLayer(input=dc6.output, in_channels = 16, out_channels = 16, kernel_len = 7, in_rows = 128, in_columns = 128, batch_size = 1, convstride=1, padsize=2, poolsize=1, poolstride=1,bias_init=0.0, name = "c7", paramMap = None, activation = 'relu')
    
    
    dc6 = DeConvLayer(input = layers[-1].output, in_channels = 64, out_channels = 3, kernel_len = 5, in_rows = 128, in_columns = 128, batch_size = 1, bias_init = 0.0, name = "c6", paramMap = None, activation = 'tanh', upsample_rate = 2, batch_norm = False)

    layers += [dc6]

    import config

    loss = T.mean(T.sqr(layers[-1].output.transpose(0,2,3,1) - x))

    #sig = lambda a: (a + 1.0) / 2

    #loss = T.mean(-1.0 * (sig(x)) * T.log(0.1 + sig(layers[-1].output.transpose(0,2,3,1))) - 1.0 * (1.0 - sig(x)) * T.log(0.1 + 1.0 - sig(layers[-1].output.transpose(0,2,3,1))))



    updates = {}

    params = {}

    for layer in layers:
        for key in layer.params.keys():
            params[key] = layer.params[key]

    params["W"] = w

    print params

    import Updates

    updateObj = Updates.Updates(params, loss, 1.0)

    updates = updateObj.getUpdates()

    f = theano.function(inputs = [x], outputs = {'o1' : layers[0].output.transpose(0,2,3,1), 'o2' : layers[1].output.transpose(0,2,3,1), 'o3' : layers[2].output.transpose(0,2,3,1), 'o4' : layers[3].output.transpose(0,2,3,1), 'o5' : layers[4].output.transpose(0,2,3,1), 'o6' : layers[5].output.transpose(0,2,3,1), 'l' : loss, 'y' : layers[-1].output.transpose(0,2,3,1)}, updates = updates)

    x = id.normalize(id.getBatch())


    print "y6 shape", f(x)['o6'].shape
    print "x norm", (x**2).sum()

    res = f(x)

    print "y1 norm", (res['o1']**2).sum()
    print "y2 norm", (res['o2']**2).sum()
    print "y3 norm", (res['o3']**2).sum()
    print "y4 norm", (res['o4']**2).sum()
    print "y5 norm", (res['o5']**2).sum()
    print "y6 norm", (res['o6']**2).sum()


    for epoch in range(0,50): 
        print "epoch", epoch
        res = f(x)
        print res['l']
        print res['ofl']
        print res['y'].min(), res['y'].max()

    im = Image.fromarray(id.denormalize(f(x)['o6'][0]).astype('uint8'), "RGB")
    im.convert('RGB').save("/u/lambalex/derpyimage.png", "PNG")


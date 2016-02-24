from Layers.HiddenLayer import HiddenLayer
from Layers.DeConvLayer import DeConvLayer
from Layers.ConvolutionalLayer import ConvPoolLayer
from Layers.Upsample import Upsample
import theano.tensor as T
import theano
import numpy as np
from Data.load_imagenet import denormalize

import theano
import theano.tensor as T
from theano.tensor.opt import register_canonicalize

class ConsiderConstant(theano.compile.ViewOp):
    def grad(self, args, g_outs):
        return [g_out * T.sqrt(T.mean(T.abs_(g_out))) / (1.0 + T.sqrt(T.mean(T.abs_(g_out), axis = (0,1), keepdims = True))) for g_out in g_outs]
        #return [g_out * 100.0 / (1.0 + T.abs_(g_out)) for g_out in g_outs]

consider_constant = ConsiderConstant()
register_canonicalize(theano.gof.OpRemove(consider_constant), name='consider_norm')


def decoder(z_reconstruction, z_sampled, numLatent, numHidden, mb_size, image_width):

    c = [2048, 2048, 512, 3]

    layers = []

    layers += [HiddenLayer(num_in = numLatent, num_out = c[0] * 4 * 4, activation = 'relu', batch_norm = True)]

    layers += [DeConvLayer(in_channels = c[0], out_channels = c[1], kernel_len = 5, activation = 'relu', batch_norm = True, unflatten_input = (mb_size, c[0], 4, 4))]
    #8x8

    layers += [DeConvLayer(in_channels = c[1], out_channels = c[2], kernel_len = 5, activation = 'relu', batch_norm = True)]
    #16x16

    layers += [DeConvLayer(in_channels = c[2], out_channels = c[3], kernel_len = 5, activation = None, batch_norm = False)]
    #32x32

    generated_outputs = [z_sampled]
    reconstruction_outputs = [z_reconstruction]

    for i in range(0, len(layers)):
        generated_outputs += [layers[i].output(generated_outputs[-1])]
        reconstruction_outputs += [layers[i].output(reconstruction_outputs[-1])]

        #reconstruction_outputs += [consider_constant(reconstruction_outputs[-1])]

    return {'layers' : layers, 'extra_params' : [], 'output' : denormalize(reconstruction_outputs[-1].transpose(0,2,3,1)), 'output_generated' : denormalize(generated_outputs[-1].transpose(0,2,3,1))}


if __name__ == "__main__":

    layer = ConvPoolLayer(in_channels = 1, out_channels = 1, kernel_len = 5, activation = None)

    x = T.tensor4()
    y = layer.output(x)

    xg = np.random.uniform(size = (1,1,16,16)).astype('float32')

    f = theano.function([x], y)

    print f(xg).shape





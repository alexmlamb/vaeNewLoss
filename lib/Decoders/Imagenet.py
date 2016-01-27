from Layers.HiddenLayer import HiddenLayer
from Layers.DeConvLayer import DeConvLayer
from Layers.ConvolutionalLayer import ConvPoolLayer
from Layers.Upsample import Upsample
import theano.tensor as T
import theano
import numpy as np

def decoder(z, z_sampled, numLatent, numHidden, mb_size, image_width):

    c = [512, 256, 128, 128, 128, 128]

    layers = []

    layers += [HiddenLayer(num_in = numLatent, num_out = numHidden, activation = 'relu', batch_norm = True)]

    layers += [HiddenLayer(num_in = numHidden, num_out = c[0] * 4 * 4, activation = 'relu', batch_norm = True)]

    layers += [ConvPoolLayer(in_channels = c[0], out_channels = c[0], kernel_len = 1, activation = 'relu', batch_norm = True, unflatten_input = (mb_size, c[0], 4, 4))]

    layers += [DeConvLayer(in_channels = c[0], out_channels = c[1], kernel_len = 5, activation = 'relu', batch_norm = True)]
    layers += [DeConvLayer(in_channels = c[1], out_channels = c[2], kernel_len = 5, activation = 'relu', batch_norm = True)]
    layers += [DeConvLayer(in_channels = c[2], out_channels = c[3], kernel_len = 5, activation = 'relu', batch_norm = True)]
    layers += [DeConvLayer(in_channels = c[3], out_channels = c[4], kernel_len = 5, activation = 'relu', batch_norm = True)]
    layers += [DeConvLayer(in_channels = c[4], out_channels = c[5], kernel_len = 5, activation = 'relu', batch_norm = True)]

    layers += [ConvPoolLayer(in_channels = c[5], out_channels = 3, kernel_len = 3, activation = None, batch_norm = False)]

    generated_outputs = [z_sampled]
    reconstruction_outputs = [z]

    for i in range(0, len(layers)):
        generated_outputs += [layers[i].output(generated_outputs[-1])]
        reconstruction_outputs += [layers[i].output(reconstruction_outputs[-1])]

    #reconstruction_outputs[-1] = reconstruction_outputs[-1].reshape((mb_size, 3, 32, 32))
    #generated_outputs[-1] = generated_outputs[-1].reshape((mb_size, 3, 32, 32))

    return {'layers' : layers, 'extra_params' : [], 'output' : reconstruction_outputs[-1].transpose(0,2,3,1), 'output_generated' : generated_outputs[-1].transpose(0,2,3,1)}


if __name__ == "__main__":

    layer = ConvPoolLayer(in_channels = 1, out_channels = 1, kernel_len = 5, activation = None)

    x = T.tensor4()
    y = layer.output(x)

    xg = np.random.uniform(size = (1,1,16,16)).astype('float32')

    f = theano.function([x], y)

    print f(xg).shape





from Layers.ConvolutionalLayer import ConvPoolLayer
from Layers.HiddenLayer import HiddenLayer
from Data.load_imagenet import normalize
import theano.tensor as T

def encoder(x, numHidden, mb_size, image_width):

    in_width = image_width
    layerLst = []

    c = [3, 512, 2048,2048]

    layerLst += [ConvPoolLayer(in_channels = c[0], out_channels = c[1], kernel_len = 5, stride=2, batch_norm = True)]
    #16

    layerLst += [ConvPoolLayer(in_channels = c[1], out_channels = c[2], kernel_len = 5, stride=2, batch_norm = True)]
    #8

    layerLst += [ConvPoolLayer(in_channels = c[2], out_channels = c[3], kernel_len = 5, stride=2, batch_norm = True)]
    #4

    layerLst += [HiddenLayer(num_in = 4 * 4 * c[3], num_out = numHidden, flatten_input = True, batch_norm = True)]

    outputs = [normalize(x.transpose(0,3,1,2))]

    for i in range(0, len(layerLst)):
        outputs += [layerLst[i].output(outputs[-1])]

    h1 = HiddenLayer(num_in = numHidden, num_out = numHidden, batch_norm = True)

    h1_out = h1.output(outputs[-1])

    return {'layers' : layerLst + [h1], 'extra_params' : [], 'output' : h1_out}





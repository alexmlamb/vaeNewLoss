from Layers.ConvolutionalLayer import ConvPoolLayer
from Layers.HiddenLayer import HiddenLayer
from Data.load_imagenet import normalize
import theano.tensor as T

def encoder(x, numHidden, labels, num_labels, mb_size, image_width):

    in_width = image_width
    layerLst = []

    c = [3, 128, 128, 128, 256, 256]

    layerLst += [ConvPoolLayer(in_channels = c[0], out_channels = c[1], kernel_len = 5, stride=2, batch_norm = True)]
    layerLst += [ConvPoolLayer(in_channels = c[1], out_channels = c[2], kernel_len = 5, stride=2, batch_norm = True)]
    layerLst += [ConvPoolLayer(in_channels = c[2], out_channels = c[3], kernel_len = 5, stride=2, batch_norm = True)]
    layerLst += [ConvPoolLayer(in_channels = c[3], out_channels = c[4], kernel_len = 5, stride=2, batch_norm = True)]
    layerLst += [ConvPoolLayer(in_channels = c[4], out_channels = c[5], kernel_len = 5, stride=2, batch_norm = True)]

    layerLst += [HiddenLayer(num_in = 4 * 4 * c[5], num_out = numHidden, flatten_input = True, batch_norm = True)]

    layerLst += [HiddenLayer(num_in = numHidden, num_out = numHidden, batch_norm = True)]

    outputs = [normalize(x.transpose(0,3,1,2))]

    for i in range(0, len(layerLst)):
        outputs += [layerLst[i].output(outputs[-1])]

    h1 = HiddenLayer(num_in = numHidden + num_labels, num_out = numHidden, batch_norm = True)
    h2 = HiddenLayer(num_in = numHidden, num_out = numHidden, batch_norm = True)

    h1_out = h1.output(T.concatenate([outputs[-1], labels], axis = 1))
    h2_out = h2.output(h1_out)


    return {'layers' : layerLst + [h1,h2], 'extra_params' : [], 'output' : h2_out}




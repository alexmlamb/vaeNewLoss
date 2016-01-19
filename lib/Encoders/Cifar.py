from Layers.ConvolutionalLayer import ConvPoolLayer
from Layers.HiddenLayer import HiddenLayer
from model_files.load_vgg import vgg_network


def imagenet_encoder(x, numHidden, mb_size, image_width):
    vgg_out = vgg_network(x, mb_size, image_width)

    h1 = HiddenLayer(num_in = 4 * 4 * 512, num_out = numHidden, flatten_input = True, activation = 'relu', batch_norm = True)

    h2 = HiddenLayer(num_in = numHidden, num_out = numHidden, activation = 'relu', batch_norm = True)

    out1 = h1.output(vgg_out['output'])

    out2 = h2.output(out1)

    params = vgg_out['params']

    return {'layers' : [h1, h2], 'extra_params' : params, 'output' : out2}

def imagenet_encoder_1(x, numHidden, mb_size, image_width):

    in_width = image_width
    layerLst = []

    c = [3, 128, 128, 256, 512]

    #layerLst += [ConvPoolLayer(in_channels = c[0], out_channels = c[1], kernel_len = 3)]
    #layerLst += [ConvPoolLayer(in_channels = c[1], out_channels = c[1], kernel_len = 3)]
    #layerLst += [ConvPoolLayer(in_channels = c[1], out_channels = c[1], kernel_len = 3)]
    #layerLst += [ConvPoolLayer(in_channels = c[1], out_channels = c[1], kernel_len = 3, stride=2)]

    #layerLst += [ConvPoolLayer(in_channels = c[1], out_channels = 128, kernel_len = 3)]
    #layerLst += [ConvPoolLayer(in_channels = 128, out_channels = 128, kernel_len = 3)]
    #layerLst += [ConvPoolLayer(in_channels = 128, out_channels = 128, kernel_len = 3)]
    #layerLst += [ConvPoolLayer(in_channels = 128, out_channels = 128, kernel_len = 3, stride=2)]

    #layerLst += [ConvPoolLayer(in_channels = 128, out_channels = 256, kernel_len = 3)]
    #layerLst += [ConvPoolLayer(in_channels = 256, out_channels = 256, kernel_len = 3)]
    #layerLst += [ConvPoolLayer(in_channels = 256, out_channels = 256, kernel_len = 3)]
    #layerLst += [ConvPoolLayer(in_channels = 256, out_channels = 256, kernel_len = 3, stride=2)]

    #layerLst += [ConvPoolLayer(in_channels = 256, out_channels = 512, kernel_len = 3)]
    #layerLst += [ConvPoolLayer(in_channels = 512, out_channels = 512, kernel_len = 3)]
    #layerLst += [ConvPoolLayer(in_channels = 512, out_channels = 512, kernel_len = 3)]
    #layerLst += [ConvPoolLayer(in_channels = 512, out_channels = 512, kernel_len = 3, stride=2)]

    #layerLst += [ConvPoolLayer(in_channels = 512, out_channels = 512, kernel_len = 3)]
    #layerLst += [ConvPoolLayer(in_channels = 512, out_channels = 512, kernel_len = 3)]
    #layerLst += [ConvPoolLayer(in_channels = 512, out_channels = 512, kernel_len = 3)]
    #layerLst += [ConvPoolLayer(in_channels = 512, out_channels = 512, kernel_len = 3, stride=2)]

    layerLst += [HiddenLayer(num_in = 4 * 4 * 256, num_out = numHidden, flatten_input = True)]

    outputs = [x.transpose(0,3,1,2)]

    for i in range(0, len(layerLst)):
        outputs += [layerLst[i].output(outputs[-1])]

    return {'layers' : layerLst, 'output' : outputs[-1]}

if __name__ == "__main__":

    import theano
    import numpy as np

    x = theano.shared(np.random.uniform(size = (10, 128, 128, 3)).astype('float32'))

    y = imagenet_encoder(x, 100, 32)['output']

    f = theano.function([], [y])

    print f()[0].shape





from Layers.ConvolutionalLayer import ConvPoolLayer
from Layers.HiddenLayer import HiddenLayer

def imagenet_encoder(x, numHidden, image_width):

    in_width = image_width
    layerLst = [x.dimshuffle(0,3,1,2)]

    c = [3, 96, 128, 256, 512]

    layerLst += [ConvPoolLayer(in_channels = c[0], out_channels = c[1], kernel_len = 3)]
    layerLst += [ConvPoolLayer(in_channels = c[1], out_channels = c[1], kernel_len = 3)]
    layerLst += [ConvPoolLayer(in_channels = c[1], out_channels = c[1], kernel_len = 3)]
    layerLst += [ConvPoolLayer(in_channels = c[1], out_channels = c[1], kernel_len = 3, stride=2)]

    layerLst += [ConvPoolLayer(in_channels = c[1], out_channels = 128, kernel_len = 3)]
    layerLst += [ConvPoolLayer(in_channels = 128, out_channels = 128, kernel_len = 3)]
    layerLst += [ConvPoolLayer(in_channels = 128, out_channels = 128, kernel_len = 3)]
    layerLst += [ConvPoolLayer(in_channels = 128, out_channels = 128, kernel_len = 3, stride=2)]

    layerLst += [ConvPoolLayer(in_channels = 128, out_channels = 256, kernel_len = 3)]
    layerLst += [ConvPoolLayer(in_channels = 256, out_channels = 256, kernel_len = 3)]
    layerLst += [ConvPoolLayer(in_channels = 256, out_channels = 256, kernel_len = 3)]
    layerLst += [ConvPoolLayer(in_channels = 256, out_channels = 256, kernel_len = 3, stride=2)]

    layerLst += [ConvPoolLayer(in_channels = 256, out_channels = 512, kernel_len = 3)]
    layerLst += [ConvPoolLayer(in_channels = 512, out_channels = 512, kernel_len = 3)]
    layerLst += [ConvPoolLayer(in_channels = 512, out_channels = 512, kernel_len = 3)]
    layerLst += [ConvPoolLayer(in_channels = 512, out_channels = 512, kernel_len = 3, stride=2)]

    layerLst += [HiddenLayer(num_in = 4 * 4 * 512, num_out = numHidden, name = "h_fc_enc_1", activation = "relu", flatten_input = True)]

    outputs = [x.dimshuffle(0,3,1,2)]

    for i in range(0, len(layerLst)):
        outputs += [layerLst[i].output(outputs[-1])]

    layers = {'h1' : h1, 'h2' : h2}

    return {'layers' : layers, 'output' : outputs[-1]}



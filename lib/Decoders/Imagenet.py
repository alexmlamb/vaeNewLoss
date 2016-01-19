from Layers.HiddenLayer import HiddenLayer
from Layers.DeConvLayer import DeConvLayer
from Layers.ConvolutionalLayer import ConvPoolLayer

import theano.tensor as T



def imagenet_decoder(z, z_sampled, numLatent, numHidden, mb_size, image_width):

    layers = []

    layers += [HiddenLayer(num_in = numLatent, num_out = 2048, activation = 'relu', batch_norm = True)]

    layers += [HiddenLayer(num_in = 2048, num_out = 256 * 8 * 8, activation = 'relu', batch_norm = True)]

    layers += [DeConvLayer(in_channels = 256, out_channels = 128, kernel_len = 5, activation = 'relu', unflatten_input = (mb_size, 256, 8, 8))]
    layers += [ConvPoolLayer(in_channels = 128, out_channels = 128, kernel_len = 3, activation = 'relu', batch_norm = True)]
    layers += [ConvPoolLayer(in_channels = 128, out_channels = 128, kernel_len = 3, activation = 'relu', batch_norm = True)]
    layers += [DeConvLayer(in_channels = 128, out_channels = 64, kernel_len = 5, activation = None)]
    layers += [ConvPoolLayer(in_channels = 64, out_channels = 64, kernel_len = 3, activation = 'relu')]
    layers += [ConvPoolLayer(in_channels = 64, out_channels = 64, kernel_len = 3, activation = 'relu')]
    layers += [DeConvLayer(in_channels = 64, out_channels = 32, kernel_len = 5, activation = 'relu')]
    layers += [ConvPoolLayer(in_channels = 32, out_channels = 32, kernel_len = 3, activation = 'relu', batch_norm = True)]
    layers += [ConvPoolLayer(in_channels = 32, out_channels = 32, kernel_len = 3, activation = 'relu', batch_norm = True)]
    layers += [DeConvLayer(in_channels = 32, out_channels = 3, kernel_len = 5, activation = None)]

    generated_outputs = [z_sampled]
    reconstruction_outputs = [z]

    for i in range(0, len(layers)):
        generated_outputs += [layers[i].output(generated_outputs[-1])]
        reconstruction_outputs += [layers[i].output(reconstruction_outputs[-1])]

    return {'layers' : layers, 'extra_params' : [], 'output' : reconstruction_outputs[-1].transpose(0,2,3,1), 'output_generated' : generated_outputs[-1].transpose(0,2,3,1)}



def imagenet_decoder_1(z, z_sampled, numLatent, numHidden, mb_size):

    h3 = HiddenLayer(z, num_in = numLatent, num_out = numHidden, initialization = 'xavier', name = "h3", activation = "relu")
    h3_generated = HiddenLayer(z_sampled, num_in = numLatent, num_out = numHidden, initialization = 'xavier', paramMap = h3.getParams(), name = "h3", activation = "relu")

    deconv_shapes = [512,256,128,64,32,16,3]

    h4 = HiddenLayer(h3.output, num_in = numHidden, num_out = 4 * 4 * deconv_shapes[0], initialization = 'xavier', name = "h4", activation = "relu")
    h4_generated = HiddenLayer(h3_generated.output, num_in = numHidden, num_out = 4 * 4 * deconv_shapes[0], initialization = 'xavier', paramMap = h4.getParams(), name = "h4", activation = "relu")

    h4_reshaped = h4.output.reshape((mb_size, 4,4,deconv_shapes[0])).dimshuffle(0, 3, 1, 2)
    h4_generated_reshaped = h4_generated.output.reshape((mb_size,4,4,deconv_shapes[0])).dimshuffle(0, 3, 1, 2)

    o1 = DeConvLayer(h4_reshaped, in_channels = deconv_shapes[0], out_channels = deconv_shapes[1], kernel_len = 5, in_rows = 4, in_columns = 4, batch_size = 100, bias_init = 0.0, name = 'o1', paramMap = None, upsample_rate = 2, activation = 'relu')

    o2 = DeConvLayer(o1.output, in_channels = deconv_shapes[1], out_channels = deconv_shapes[2], kernel_len = 5, in_rows = 8, in_columns = 8, batch_size = 100, bias_init = 0.0, name = 'o2', paramMap = None, upsample_rate = 2, activation = 'relu')

    o3 = DeConvLayer(o2.output, in_channels = deconv_shapes[2], out_channels = deconv_shapes[3], kernel_len = 5, in_rows = 16, in_columns = 16, batch_size = 100, bias_init = 0.0, name = 'o3', paramMap = None, upsample_rate = 2, activation = 'relu')

    o4 = DeConvLayer(o3.output, in_channels = deconv_shapes[3], out_channels = deconv_shapes[4], kernel_len = 5, in_rows = 32, in_columns = 32, batch_size = 100, bias_init = 0.0, name = 'o4', paramMap = None, upsample_rate = 2, activation = 'relu')

    o5 = DeConvLayer(o4.output, in_channels = deconv_shapes[4], out_channels = deconv_shapes[5], kernel_len = 5, in_rows = 64, in_columns = 64, batch_size = 100, bias_init = 0.0, name = 'o5', paramMap = None, upsample_rate = 2, activation = 'relu')

    y = DeConvLayer(o5.output, in_channels = deconv_shapes[5], out_channels = deconv_shapes[6], kernel_len = 5, in_rows = 128, in_columns = 128, batch_size = 100, bias_init = 0.0, name = 'y', paramMap = None, upsample_rate = 2, activation = None, batch_norm = False)

    o1_generated = DeConvLayer(h4_generated_reshaped, in_channels = deconv_shapes[0], out_channels = deconv_shapes[1], kernel_len = 5, in_rows = 4, in_columns = 4, batch_size = 100, bias_init = 0.0, name = 'o1', paramMap = o1.getParams(), upsample_rate = 2, activation = 'relu')

    o2_generated = DeConvLayer(o1_generated.output, in_channels = deconv_shapes[1], out_channels = deconv_shapes[2], kernel_len = 5, in_rows = 8, in_columns = 8, batch_size = 100, bias_init = 0.0, name = 'o2', paramMap = o2.getParams(), upsample_rate = 2, activation = 'relu')

    o3_generated = DeConvLayer(o2_generated.output, in_channels = deconv_shapes[2], out_channels = deconv_shapes[3], kernel_len = 5, in_rows = 16, in_columns = 16, batch_size = 100, bias_init = 0.0, name = 'o3', paramMap = o3.getParams(), upsample_rate = 2, activation = 'relu')

    o4_generated = DeConvLayer(o3_generated.output, in_channels = deconv_shapes[3], out_channels = deconv_shapes[4], kernel_len = 5, in_rows = 32, in_columns = 32, batch_size = 100, bias_init = 0.0, name = 'o4', paramMap = o4.getParams(), upsample_rate = 2, activation = 'relu')

    o5_generated = DeConvLayer(o4_generated.output, in_channels = deconv_shapes[4], out_channels = deconv_shapes[5], kernel_len = 5, in_rows = 64, in_columns = 64, batch_size = 100, bias_init = 0.0, name = 'o5', paramMap = o5.getParams(), upsample_rate = 2, activation = 'relu')

    y_generated = DeConvLayer(o5_generated.output, in_channels = deconv_shapes[5], out_channels = deconv_shapes[6], kernel_len = 5, in_rows = 128, in_columns = 128, batch_size = 100, bias_init = 0.0, name = 'y', paramMap = y.getParams(), upsample_rate = 2, activation = None, batch_norm = False)

    output = y.output.dimshuffle(0, 2, 3, 1)
    sample_output = y_generated.output.dimshuffle(0, 2, 3, 1)

    layers = {'h3' : h3, 'h4' : h4, 'o1' : o1, 'o2' : o2, 'o3' : o3, 'o4' : o4, 'o5' : o5, 'y' : y}

    return {'layers' : layers, 'output' : output, 'output_generated' : sample_output}





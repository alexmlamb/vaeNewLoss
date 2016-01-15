from Layers.HiddenLayer import HiddenLayer
from Layers.DeConvLayer import DeConvLayer

def svhn_decoder(z, z_sampled, numLatent, numHidden, mb_size):


    h3 = HiddenLayer(z, num_in = numLatent, num_out = numHidden, initialization = 'xavier', name = "h3", activation = 'relu', batch_norm = False)

    h4 = HiddenLayer(h3.output, num_in = numHidden, num_out = 32 * 32 * 3, initialization = 'xavier', name = "h4", activation = None, batch_norm = False)

    h3_generated = HiddenLayer(z_sampled, num_in = numLatent, num_out = numHidden, initialization = 'xavier', paramMap = h3.getParams(), name = "h3", activation = 'relu', batch_norm = False)

    h4_generated = HiddenLayer(h3_generated.output, num_in = numHidden, num_out = 32 * 32 * 3, initialization = 'xavier', paramMap = h4.getParams(), name = "h4", activation = None, batch_norm = False)


    return {'layers' : {'h3' : h3, 'h4' : h4}, 'output' : h4.output.reshape((128,32,32,3)), 'output_generated' : h4_generated.output.reshape((128,32,32,3))}




def svhn_decoder_1(z, z_sampled, numLatent, numHidden, mb_size):

    batch_norm = False

    h3 = HiddenLayer(z, num_in = numLatent, num_out = numHidden, initialization = 'xavier', name = "h3", activation = "relu", batch_norm = batch_norm)
    h3_generated = HiddenLayer(z_sampled, num_in = numLatent, num_out = numHidden, initialization = 'xavier', paramMap = h3.getParams(), name = "h3", activation = "relu", batch_norm = batch_norm)

    deconv_shapes = [1024,256,128,3]

    h4 = HiddenLayer(h3.output, num_in = numHidden, num_out = 4 * 4 * deconv_shapes[0], initialization = 'xavier', name = "h4", activation = "relu", batch_norm = batch_norm)
    h4_generated = HiddenLayer(h3_generated.output, num_in = numHidden, num_out = 4 * 4 * deconv_shapes[0], initialization = 'xavier', paramMap = h4.getParams(), name = "h4", activation = "relu", batch_norm = batch_norm)

    h4_reshaped = h4.output.reshape((mb_size, 4,4,deconv_shapes[0])).dimshuffle(0, 3, 1, 2)
    h4_generated_reshaped = h4_generated.output.reshape((mb_size,4,4,deconv_shapes[0])).dimshuffle(0, 3, 1, 2)

    o1 = DeConvLayer(h4_reshaped, in_channels = deconv_shapes[0], out_channels = deconv_shapes[1], kernel_len = 5, in_rows = 4, in_columns = 4, batch_size = 100, bias_init = 0.0, name = 'o1', paramMap = None, upsample_rate = 2, activation = 'relu', batch_norm = batch_norm)

    o2 = DeConvLayer(o1.output, in_channels = deconv_shapes[1], out_channels = deconv_shapes[2], kernel_len = 5, in_rows = 8, in_columns = 8, batch_size = 100, bias_init = 0.0, name = 'o2', paramMap = None, upsample_rate = 2, activation = 'relu', batch_norm = batch_norm)

    o3 = DeConvLayer(o2.output, in_channels = deconv_shapes[2], out_channels = deconv_shapes[3], kernel_len = 5, in_rows = 16, in_columns = 16, batch_size = 100, bias_init = 0.0, name = 'o3', paramMap = None, upsample_rate = 2, activation = None, batch_norm = batch_norm)

    o1_generated = DeConvLayer(h4_generated_reshaped, in_channels = deconv_shapes[0], out_channels = deconv_shapes[1], kernel_len = 5, in_rows = 4, in_columns = 4, batch_size = 100, bias_init = 0.0, name = 'o1', paramMap = o1.getParams(), upsample_rate = 2, activation = 'relu', batch_norm = batch_norm)

    o2_generated = DeConvLayer(o1_generated.output, in_channels = deconv_shapes[1], out_channels = deconv_shapes[2], kernel_len = 5, in_rows = 8, in_columns = 8, batch_size = 100, bias_init = 0.0, name = 'o2', paramMap = o2.getParams(), upsample_rate = 2, activation = 'relu', batch_norm = batch_norm)

    o3_generated = DeConvLayer(o2_generated.output, in_channels = deconv_shapes[2], out_channels = deconv_shapes[3], kernel_len = 5, in_rows = 16, in_columns = 16, batch_size = 100, bias_init = 0.0, name = 'o3', paramMap = o3.getParams(), upsample_rate = 2, activation = None, batch_norm = batch_norm)

    output = o3.output.dimshuffle(0, 2, 3, 1)
    sample_output = o3_generated.output.dimshuffle(0, 2, 3, 1)

    layers = {'h3' : h3, 'h4' : h4, 'o1' : o1, 'o2' : o2, 'o3' : o3}

    return {'layers' : layers, 'output' : output, 'output_generated' : sample_output}


import theano
import theano.tensor as T
import numpy as np
import numpy.random as rng
from HiddenLayer import HiddenLayer
from Updates import Updates
import math
import time

from ConvolutionalLayer import ConvPoolLayer

import cPickle

from config import get_config

from load_data import load_data_mnist, load_data_svhn

from PIL import Image

import os

import pprint

from Classifier import Classifier

from Predictor import classify_image, c1_diff

from DeConvLayer import DeConvLayer

theano.config.flaotX = 'float32'


if __name__ == "__main__": 

    config = get_config()

    config["layer_weighting"] = {'y' : 1.0}

    if config["dataset"] == "mnist":
        xData = load_data_mnist(config)
    elif config["dataset"] == "svhn":
        xData = load_data_svhn(config)
    else:
        raise Exception("dataset not found")

    config["classifier_load"] = "model_files/get_hidden_svhn_1450486080.pkl"

    if config["classifier_load"] != None:
        classifier_loaded = cPickle.load(open(config["classifier_load"], "r"))
        use_cl = True
    else:
        use_cl = False

    print "loaded classifiers"


    #raise Exception('done')

    print "compiled hidden grabber"

    config["learning_rate"] = 0.00001
    config["number_epochs"] = 200000000
    config["report_epoch_ratio"] = 500
    config["popups"] = True

    experimentDir = "plots/exp_" + str(int(time.time()))
    os.mkdir(experimentDir)

    config["experiment_type"] = "original_layer"

    numHidden = 1000
    #was 50
    numLatent = 50
    numInput = config['num_input']
    numOutput = config['num_input']

    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

    svhn_mean = 50.0
    svhn_std = 200.0

    s = pprint.pformat(config)
    configLogFile = open(experimentDir + "/log.txt", "w")
    configLogFile.write(s)
    configLogFile.close()

    #N x 1
    x = T.tensor4()
    observed_y = T.matrix()

    if config['dataset'] == 'svhn':
        x_normed = (x - svhn_mean) / svhn_std
        observed_y_normed = (observed_y - svhn_mean) / svhn_std


    c1 = ConvPoolLayer(input=x_normed, in_channels = 3, out_channels = 96, kernel_len = 5, in_rows = 32, in_columns = 32, batch_size = 100,
                                        convstride=1, padsize=4,
                                        poolsize=3, poolstride=2,
                                        bias_init=0.0, name = "c1", paramMap = None)

    c2 = ConvPoolLayer(input=c1.output, in_channels = 96, out_channels = 128, kernel_len = 3, in_rows = 17, in_columns = 17, batch_size = 100,
                                        convstride=1, padsize=3,
                                        poolsize=3, poolstride=2,
                                        bias_init=0.0, name = "c2", paramMap = None
                                        )

    c3 = ConvPoolLayer(input=c2.output, in_channels = 128, out_channels = 256, kernel_len = 3, in_rows = 10, in_columns = 10, batch_size = 100,
                                        convstride=1, padsize=0,
                                        poolsize=3, poolstride=2,
                                        bias_init=0.0, name = "c3", paramMap = None)

    h1 = HiddenLayer(c3.output.flatten(2), num_in = 2304, num_out = numHidden, initialization = 'xavier', name = "h1", activation = "relu")

    h2 = HiddenLayer(h1.output, num_in = numHidden, num_out = numHidden, initialization = 'xavier', name = "h2", activation = "relu")

    z_mean = HiddenLayer(h2.output, num_in = numHidden, num_out = numLatent, initialization = 'xavier', name = 'z_mean', activation = None)

    z_var = HiddenLayer(h2.output, num_in = numHidden, num_out = numLatent, initialization = 'xavier', name = 'z_var', activation = 'exp')

    z_sampled = srng.normal(size = (100, numLatent))

    z = z_sampled * z_var.output + z_mean.output

    #h3 = HiddenLayer(z, num_in = numLatent, num_out = numHidden, initialization = 'xavier', name = "h3", activation = "relu")

    #h4 = HiddenLayer(h3.output, num_in = numHidden, num_out = numHidden, initialization = 'xavier', name = "h4", activation = "relu")

    #y = HiddenLayer(h4.output, num_in = numHidden, num_out = numOutput, initialization = 'xavier', name = "output", activation = None)

    #h3_generated = HiddenLayer(z_sampled, num_in = numLatent, num_out = numHidden, initialization = 'xavier', paramMap = h3.getParams(), name = "h3", activation = "relu")

    #h4_generated = HiddenLayer(h3_generated.output, num_in = numHidden, num_out = numHidden, initialization = 'xavier', paramMap = h4.getParams(), name = "h4", activation = "relu")

    #y_generated = HiddenLayer(h4_generated.output, num_in = numHidden, num_out = numOutput, initialization = 'xavier', paramMap = y.getParams(), name = "output", activation = 'tanh')

    #Make y and y_generated from z.  

    #DECODER

    h3 = HiddenLayer(z, num_in = numLatent, num_out = 2048, initialization = 'xavier', name = "h3", activation = "relu")
    h3_generated = HiddenLayer(z_sampled, num_in = numLatent, num_out = 2048, initialization = 'xavier', paramMap = h3.getParams(), name = "h3", activation = "relu")

    h3_reshaped = h3.output.reshape((100, 4,4,128))
    h3_generated_reshaped = h3_generated.output.reshape((100,4,4,128))

    o1 = DeConvLayer(h3_reshaped, in_channels = 128, out_channels = 128, kernel_len = 5, in_rows = 4, in_columns = 4, batch_size = 100, bias_init = 0.0, name = 'o1', paramMap = None, upsample_rate = 2, activation = 'relu')

    o2 = DeConvLayer(o1.output, in_channels = 128, out_channels = 64, kernel_len = 5, in_rows = 8, in_columns = 8, batch_size = 100, bias_init = 0.0, name = 'o2', paramMap = None, upsample_rate = 2, activation = 'relu')

    y = DeConvLayer(o2.output, in_channels = 64, out_channels = 3, kernel_len = 5, in_rows = 16, in_columns = 16, batch_size = 100, bias_init = 0.0, name = 'y', paramMap = None, activation = 'tanh', upsample_rate = 2)

    o1_generated = DeConvLayer(h3_generated_reshaped, in_channels = 128, out_channels = 256, kernel_len = 5, in_rows = 4, in_columns = 4, batch_size = 100, bias_init = 0.0, name = 'o1', paramMap = o1.getParams(), upsample_rate = 2, activation = 'relu')

    o2_generated = DeConvLayer(o1_generated.output, in_channels = 256, out_channels = 128, kernel_len = 5, in_rows = 8, in_columns = 8, batch_size = 100, bias_init = 0.0, name = 'o2', paramMap = o2.getParams(), upsample_rate = 2, activation = 'relu')

    y_generated = DeConvLayer(o2_generated.output, in_channels = 128, out_channels = 3, kernel_len = 5, in_rows = 16, in_columns = 16, batch_size = 100, bias_init = 0.0, name = 'y', paramMap = y.getParams(), activation = 'tanh', upsample_rate = 2)

    y.output = y.output.reshape((100, 3072))
    y_generated.output = y_generated.output.reshape((100, 3072))

    #DECODER DONE

    if config['dataset'] == 'svhn':
        y_generated.output = (y_generated.output * svhn_std) + svhn_mean

    #layers = [h1,z_mean,z_var,h2,h3,y,h4,c1,c2,c3]
    layers = [h1,z_mean,z_var,h2,c1,c2,c3,y, o1, o2, h3]

    params = {}

    for layer in layers: 
        layerParams = layer.getParams()
        for paramKey in layerParams: 
            params[paramKey] = layerParams[paramKey]

    print "params", params


    variational_loss = 0.5 * T.sum(z_mean.output**2 + z_var.output - T.log(z_var.output) - 1.0)

    loss = T.sum(T.sqr(y.output - observed_y_normed))

    #loss = 0.0

    y_output_shaped = T.reshape(y.output, (100,32,32,3)) * svhn_std + svhn_mean
    observed_y_shaped = T.reshape(observed_y_normed, (100,32,32,3)) * svhn_std + svhn_mean

    loss += c1_diff(y_output_shaped, observed_y_shaped)

    #loss = T.sum(T.sqr(y_output_shaped - observed_y_shaped))


    loss += variational_loss

    updateObj = Updates(params, loss, config["learning_rate"])

    updates = updateObj.getUpdates()

    shaped_x = T.tensor4()

    #classify = theano.function(inputs = [shaped_x], outputs = {'o' : get_hidden_diff(shaped_x, shaped_x, {})})

    train = theano.function(inputs = [x, observed_y], outputs = {'loss' : loss, 'variational_loss' : variational_loss, 'samples' : y_generated.output, 'reconstruction' : y_output_shaped, 'obs' : observed_y_shaped}, updates = updates)

    print "Finished compiling training function"

    sample = theano.function(inputs = [], outputs = [y_generated.output])

    lossLst = []


    #compute_hidden_diff = theano.function(inputs = [xA, xB], outputs = {'hd' : get_hidden_diff(xA, xB, config['layer_weighting'])})

    for iteration in range(0, config["number_epochs"]): 

        index = (iteration * 100) % xData["train"][0].shape[0]

        x = xData["train"][0][index : index + 100]
        y_true = xData['train'][1][index : index + 100]

        if x.shape[0] != 100:
            continue

        x = np.swapaxes(x, 2, 3).swapaxes(1,2)

        x_notflat = x
        x = x.reshape(100, config['num_input'])


        results = train(x_notflat,x)

        loss = results['loss']
        variational_loss = results['variational_loss']
        y = results['samples']

        lossLst += [math.log(loss)]

        if iteration % config["report_epoch_ratio"] == 0: 

            if config['dataset'] == 'mnist':
                ys = y[0].reshape(28, 28) * 255.0
                im = Image.fromarray(ys)
            elif config['dataset'] == 'svhn':
                ys = y[0].reshape(32, 32, 3)
                ys_rec = results['reconstruction'][0]
                im = Image.fromarray(ys.astype('uint8'), "RGB")
                im2 = Image.fromarray(ys_rec.astype('uint8'), "RGB")
                im3 = Image.fromarray(results['obs'][0].astype('uint8'), "RGB")
            else:
                raise Exception("")

            print "=============================================="

            print "iteration", str(iteration / config["report_epoch_ratio"])

            im.convert('RGB').save(experimentDir + "/iteration_" + str(iteration / config["report_epoch_ratio"]) + ".png", "PNG")
            im2.convert('RGB').save(experimentDir + "/reconstruction_iteration_" + str(iteration / config["report_epoch_ratio"]) + ".png", "PNG")
            im3.convert('RGB').save(experimentDir + "/observed_iteration_" + str(iteration / config["report_epoch_ratio"]) + ".png", "PNG")

            print "True Label", y_true[0]
            #print "Classify out", classify(x_notflat)['o'][0]

            print "loss", loss
            print "vloss", variational_loss





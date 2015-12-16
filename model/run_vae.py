import theano
import theano.tensor as T
import numpy as np
import numpy.random as rng
from HiddenLayer import HiddenLayer
from Updates import Updates
import math
import time

from config import get_config

from load_data import load_data_mnist, load_data_svhn

from PIL import Image

import os

import pprint

theano.config.flaotX = 'float32'


if __name__ == "__main__": 

    config = get_config()


    config["learning_rate"] = 0.00001
    config["number_epochs"] = 200000000
    config["report_epoch_ratio"] = 500
    config["popups"] = True

    experimentDir = "plots/exp_" + str(int(time.time()))
    os.mkdir(experimentDir)


    numHidden = 1000
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
    x = T.matrix()
    observed_y = T.matrix()

    if config['dataset'] == 'svhn':
        x_normed = (x - svhn_mean) / svhn_std
        observed_y_normed = (observed_y - svhn_mean) / svhn_std

    h1 = HiddenLayer(x_normed, num_in = numInput, num_out = numHidden, initialization = 'xavier', name = "h1", activation = "relu")

    h2 = HiddenLayer(h1.output, num_in = numHidden, num_out = numHidden, initialization = 'xavier', name = "h2", activation = "relu")

    z_mean = HiddenLayer(h2.output, num_in = numHidden, num_out = numLatent, initialization = 'xavier', name = 'z_mean', activation = None)

    z_var = HiddenLayer(h2.output, num_in = numHidden, num_out = numLatent, initialization = 'xavier', name = 'z_var', activation = 'exp')

    z_sampled = srng.normal(size = (100, numLatent))

    z = z_sampled * z_var.output + z_mean.output

    h3 = HiddenLayer(z, num_in = numLatent, num_out = numHidden, initialization = 'xavier', name = "h3", activation = "relu")

    h4 = HiddenLayer(h3.output, num_in = numHidden, num_out = numHidden, initialization = 'xavier', name = "h4", activation = "relu")

    y = HiddenLayer(h4.output, num_in = numHidden, num_out = numOutput, initialization = 'xavier', name = "output", activation = None)

    h3_generated = HiddenLayer(z_sampled, num_in = numLatent, num_out = numHidden, initialization = 'xavier', params = h3.getParams(), name = "h3", activation = "relu")

    h4_generated = HiddenLayer(h3_generated.output, num_in = numHidden, num_out = numHidden, initialization = 'xavier', params = h4.getParams(), name = "h4", activation = "relu")

    y_generated = HiddenLayer(h4_generated.output, num_in = numHidden, num_out = numOutput, initialization = 'xavier', params = y.getParams(), name = "output", activation = None)

    if config['dataset'] == 'svhn':
        y_generated.output = (y_generated.output * svhn_std) + svhn_mean

    layers = [h1,z_mean,z_var,h2,h3,y,h4]

    params = {}

    for layer in layers: 
        layerParams = layer.getParams()
        for paramKey in layerParams: 
            params[paramKey] = layerParams[paramKey]

    print "params", params


    variational_loss = 0.5 * T.sum(z_mean.output**2 + z_var.output - T.log(z_var.output) - 1.0)

    loss = T.sum(T.sqr(y.output - observed_y_normed)) + variational_loss

    updateObj = Updates(params, loss, config["learning_rate"])

    updates = updateObj.getUpdates()

    train = theano.function(inputs = [x, observed_y], outputs = {'loss' : loss, 'variational_loss' : variational_loss, 'samples' : y_generated.output}, updates = updates)

    print "Finished compiling training function"

    sample = theano.function(inputs = [], outputs = [y_generated.output])

    lossLst = []

    if config["dataset"] == "mnist":
        xData = load_data_mnist(config)
    elif config["dataset"] == "svhn":
        xData = load_data_svhn(config)
    else:
        raise Exception("dataset not found")

    for iteration in range(0, config["number_epochs"]): 

        index = (iteration * 100) % xData["train"][0].shape[0]

        x = xData["train"][0][index : index + 100]
        
        if x.shape[0] != 100:
            continue

        if config["dataset"] == 'svhn':
            x = np.swapaxes(x, 2, 3).swapaxes(1,2)

        x = x.reshape(100, config['num_input'])


        results = train(x,x)

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
                im = Image.fromarray(ys.astype('uint8'), "RGB")
            else:
                raise Exception("")


            im.convert('RGB').save(experimentDir + "/iteration_" + str(iteration / config["report_epoch_ratio"]) + ".png", "PNG")

            print "loss", loss
            print "vloss", variational_loss





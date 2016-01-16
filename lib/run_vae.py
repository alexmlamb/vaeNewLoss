import theano
import theano.tensor as T
import numpy as np
import numpy.random as rng
from Layers.HiddenLayer import HiddenLayer
from Updates import Updates
import math
import time

#from Classifiers.imagenet_classifier import get_overfeat_diff


import cPickle

from config import get_config

#from load_data import load_data_mnist, load_data_svhn

from model_files.load_vgg import get_dist

from Data.load_imagenet import ImageNetData
from Data.load_svhn import SvhnData

from PIL import Image

import os

import pprint

#from Classifiers.SvhnPredictor import c1_diff
#from Predictor import c1_diff


theano.config.flaotX = 'float32'


if __name__ == "__main__": 

    config = get_config()

    config["layer_weighting"] = {'y' : 1.0}


    if config['dataset'] == "imagenet":
        data = ImageNetData(config)
    elif config['dataset'] == "svhn":
        data = SvhnData(config)
    else:
        raise Exception()

    print "compiled hidden grabber"

    #0.001 works
    config["learning_rate"] = 100.0
    config["number_epochs"] = 20000000
    config["report_epoch_ratio"] = 20
    config["popups"] = True

    experimentDir = "plots/exp_" + str(int(time.time()))
    os.mkdir(experimentDir)

    config["experiment_type"] = "original_layer"

    numHidden = 200
    #was 50
    numLatent = 50
    numInput = config['num_input']
    numOutput = config['num_input']

    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))


    s = pprint.pformat(config)
    configLogFile = open(experimentDir + "/log.txt", "w")
    configLogFile.write(s)
    configLogFile.close()

    #N x 1
    x = T.tensor4()

    if config['dataset'] == 'imagenet':
        from Encoders.Imagenet import imagenet_encoder as encoder_class
    elif config['dataset'] == 'svhn':
        from Encoders.Svhn import svhn_encoder as encoder_class
    else:
        raise Exception()

    encoder = encoder_class(data.normalize(x), numHidden, config['image_width'])

    encoder_layers = encoder['layers'].values()
    encoder_output = encoder['output']

    z_mean = HiddenLayer(encoder_output, num_in = numHidden, num_out = numLatent, initialization = 'xavier', name = 'z_mean', activation = None)

    z_var = HiddenLayer(encoder_output, num_in = numHidden, num_out = numLatent, initialization = 'xavier', name = 'z_var', activation = 'exp')

    z_sampled = srng.normal(size = (config['mb_size'], numLatent))

    z = z_sampled * z_var.output + z_mean.output

    if config["dataset"] == "imagenet":
        from Decoders.Imagenet import imagenet_decoder
        decoder = imagenet_decoder(z = z, z_sampled = z_sampled, numHidden = numHidden, numLatent = numLatent, mb_size = config['mb_size'], image_width = config['image_width'])
    elif config["dataset"] == "svhn":
        from Decoders.Svhn import svhn_decoder
        decoder = svhn_decoder(z = z, z_sampled = z_sampled, numHidden = numHidden, numLatent = numLatent, mb_size = config['mb_size'], image_width = config['image_width'])
    else:
        raise Exception()

    decoder_layers = decoder['layers'].values()
    x_reconstructed = decoder['output']
    x_sampled = decoder['output_generated']

    layers = [z_mean,z_var] + encoder_layers + decoder_layers

    params = {}

    for layer in layers: 
        layerParams = layer.getParams()
        for paramKey in layerParams: 
            params[paramKey] = layerParams[paramKey]
            print "param shape", layerParams[paramKey].get_value().shape

    print "params", params

    variational_loss = 0.5 * T.sum(z_mean.output**2 + z_var.output - T.log(z_var.output) - 1.0)



    #y_out_sig = T.nnet.sigmoid(x_reconstructed)
    #y_obs_sig = (observed_y + 1.0) / 2

    #square_loss = T.sum(-1.0 * (y_obs_sig) * T.log(y_out_sig) - 1.0 * (1.0 - y_obs_sig) * T.log(1.0 - y_out_sig))

    square_loss = 0.0 * T.sum(T.sqr(x - x_reconstructed))

    loss = 0.0

    loss += square_loss

    dist_style, dist_content = get_dist(x_reconstructed, x, config)

    loss += sum(dist_style.values()) + sum(dist_content.values())

    loss += variational_loss

    updateObj = Updates(params, loss, config["learning_rate"])

    updates = updateObj.getUpdates()

    print "starting compilation"
    t0 = time.time()

    train = theano.function(inputs = [x], outputs = {'total_loss' : loss, 'square_loss' : square_loss, 'overfeat_loss' : sum(dist_style.values()) + sum(dist_content.values()), 'variational_loss' : variational_loss, 'samples' : x_reconstructed, 'reconstruction' : x_reconstructed, 'g' : T.sum(T.sqr(T.grad(T.sum(x_reconstructed), x))), 'z_mean' : z_mean.output, 'z_var' : z_var.output, 'loss_style' : sum(dist_style.values()), 'loss_content' : sum(dist_content.values())}, updates = updates)

    #dist_content.update(dist_style)
    #get_losses = theano.function(inputs = [x], outputs = dist_content)

    print "Finished compiling training function in", time.time() - t0

    sample = theano.function(inputs = [], outputs = [x_sampled])

    total_loss_lst = []
    square_loss_lst = []
    overfeat_loss_lst = []

    #compute_hidden_diff = theano.function(inputs = [xA, xB], outputs = {'hd' : get_hidden_diff(xA, xB, config['layer_weighting'])})

    print "running on data"

    iteration = -1

    while True: 

        iteration += 1

        index = (iteration * config['mb_size']) % data.numExamples

        x_batch = data.getBatch()

        x = x_batch['x']

        if x.shape[0] != config['mb_size']:
            x_batch = data.getBatch()
            x = x_batch['x']

        results = train(x)


        total_loss_lst.append(results['total_loss'])
        square_loss_lst.append(results['square_loss'])
        overfeat_loss_lst.append(results['overfeat_loss'])

        variational_loss = results['variational_loss']
        y = results['samples']

        if iteration % config["report_epoch_ratio"] == 0: 

            print 'style loss', results['loss_style']
            print 'content loss', results['loss_content']

            #il = get_losses(x)
            #for key in sorted(il.keys()):
            #    print key, il[key]

            print "z mean", results["z_mean"].min(), results['z_mean'].max()
            print "z var", results['z_var'].min(), results["z_var"].max()

            ys = y[0]
            ys_rec = np.clip(results['reconstruction'][0], 0.0, 255.0)
            print "ys rec max", results['reconstruction'].max()
            print "ys rec min", results['reconstruction'].min()
            im = Image.fromarray(ys.astype('uint8'), "RGB")
            im2 = Image.fromarray(ys_rec.astype('uint8'), "RGB")
            im3 = Image.fromarray(x[0].astype('uint8'), "RGB")

            print "=============================================="

            print "iteration", str(iteration / config["report_epoch_ratio"])

            print "dy/dx", results['g']

            im.convert('RGB').save(experimentDir + "/iteration_" + str(iteration / config["report_epoch_ratio"]) + ".png", "PNG")
            im2.convert('RGB').save(experimentDir + "/reconstruction_iteration_" + str(iteration / config["report_epoch_ratio"]) + ".png", "PNG")
            im3.convert('RGB').save(experimentDir + "/observed_iteration_" + str(iteration / config["report_epoch_ratio"]) + ".png", "PNG")

            print "Total Loss", sum(total_loss_lst) * 1.0 / len(total_loss_lst)
            print "Square Loss", sum(square_loss_lst) * 1.0 / len(square_loss_lst)
            print "Overf Loss", sum(overfeat_loss_lst) * 1.0 / len(overfeat_loss_lst)
            print "Var Loss", variational_loss

            total_loss_lst = []
            square_loss_lst = []
            overfeat_loss_lst = []


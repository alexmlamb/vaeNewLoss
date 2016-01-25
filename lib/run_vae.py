import theano
import theano.tensor as T
import numpy as np
import numpy.random as rng
from Layers.HiddenLayer import HiddenLayer
#from Updates import Updates

import lasagne

import math
import time

#from Classifiers.imagenet_classifier import get_overfeat_diff


import cPickle

from config import get_config

#from load_data import load_data_mnist, load_data_svhn

from model_files.load_vgg import get_dist

from Data.load_imagenet import ImageNetData
from Data.load_svhn import SvhnData
from Data.load_cifar import CifarData

from Layers.total_variation_denoising import total_denoising_variation_penalty

from PIL import Image

import os

import pprint

import sys
sys.setrecursionlimit(990000)

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
    elif config['dataset'] == 'cifar':
        data = CifarData(config, "train")
    else:
        raise Exception()

    print "compiled hidden grabber"

    config["number_epochs"] = 20000000
    config["report_epoch_ratio"] = 5
    config["popups"] = True

    experimentDir = config['plot_output_directory'] + "exp_" + str(int(time.time()))
    os.mkdir(experimentDir)

    config["experiment_type"] = "original_layer"

    numHidden = 2048
    numLatent = 256

    print "NUMBER OF LATENT DIMENSIONS", numLatent

    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

    s = pprint.pformat(config)
    configLogFile = open(experimentDir + "/log.txt", "w")
    configLogFile.write(s)
    configLogFile.close()

    #N x 1
    x = T.tensor4()
    labels = T.ivector()

    if config['dataset'] == 'imagenet':
        from Encoders.Imagenet import encoder as encoder_class
    elif config['dataset'] == 'svhn' or config['dataset'] == 'cifar':
        from Encoders.Svhn import svhn_encoder as encoder_class
    else:
        raise Exception()

    encoder = encoder_class(x, numHidden, mb_size = config['mb_size'], image_width = config['image_width'])

    encoder_layers = encoder['layers']
    encoder_output = encoder['output']
    encoder_extra_params = encoder['extra_params']

    z_mean_layer = HiddenLayer(num_in = numHidden, num_out = numLatent, activation = None)

    z_var_layer = HiddenLayer(num_in = numHidden, num_out = numLatent, activation = 'exp')


    labels_reshaped = T.zeros(shape = (config['mb_size'], config['num_labels']))

    #labels_reshaped = labels_reshaped.set_subtensor(T.arange(config['mb_size']), labels] = 1.0

    labels_reshaped = T.set_subtensor(labels_reshaped[T.arange(config['mb_size']), labels], 1.0)

    z_mean = z_mean_layer.output(encoder_output)
    z_var = z_var_layer.output(encoder_output)

    #z_sampled = srng.normal(size = (config['mb_size'], numLatent))

    z_sampled = T.matrix()

    z = z_sampled * z_var + z_mean

    def join(a,b):
        return T.concatenate([a,b], axis = 1)

    if config["dataset"] == "imagenet":
        from Decoders.Imagenet import decoder
        decoder = decoder(z = join(z, labels_reshaped), z_sampled = join(z_sampled, labels_reshaped), numHidden = numHidden, numLatent = numLatent + config['num_labels'], mb_size = config['mb_size'], image_width = config['image_width'])
    elif config["dataset"] == "svhn" or config['dataset'] == 'cifar':
        from Decoders.Svhn import svhn_decoder
        decoder = svhn_decoder(z = join(z, labels_reshaped), z_sampled = join(z_sampled, labels_reshaped), numHidden = numHidden, numLatent = numLatent + config['num_labels'], mb_size = config['mb_size'], image_width = config['image_width'])
    else:
        raise Exception()

    decoder_layers = decoder['layers']
    x_reconstructed = decoder['output']
    x_sampled = decoder['output_generated']
    decoder_extra_params = decoder['extra_params']

    layers = [z_mean_layer,z_var_layer] + encoder_layers + decoder_layers

    l2_loss = 0.0

    params = []

    layerIndex = 0
    for layer in layers: 
        layerIndex += 1
        layerParams = layer.getParams()
        for paramKey in layerParams: 
            params += [layerParams[paramKey]]
            l2_loss += T.mean(T.sqr(params[-1]))

    l2_loss = 0.0001 * T.sqrt(l2_loss)

    params += encoder_extra_params + decoder_extra_params

    for param in params:
        print param.get_value().shape

    variational_loss = 0.5 * T.sum(z_mean**2 + z_var - T.log(z_var) - 1.0)


    smoothness_penalty = 0.0001 * (total_denoising_variation_penalty(x_reconstructed.transpose(0,3,1,2)[:,0:1,:,:]) + total_denoising_variation_penalty(x_reconstructed.transpose(0,3,1,2)[:,1:2,:,:]) + total_denoising_variation_penalty(x_reconstructed.transpose(0,3,1,2)[:,2:3,:,:]))

    square_loss = 1.0 * T.mean(T.sqr(x - x_reconstructed))

    loss = 0.0

    loss += l2_loss

    loss += square_loss

    loss += smoothness_penalty

    dist_style, dist_content = get_dist(x, x_reconstructed, config)

    style_loss = 50.0 * dist_style
    content_loss = 100.0 * sum(dist_content.values())

    loss += style_loss + content_loss

    loss += 1.0 * variational_loss

    #updateObj = Updates(params, loss, config["learning_rate"])
    #updates = updateObj.getUpdates()

    updates = lasagne.updates.adam(loss, params, learning_rate = 0.0001)

    print "starting compilation"
    t0 = time.time()

    train = theano.function(inputs = [x, labels, z_sampled], outputs = {'total_loss' : loss, 'square_loss' : square_loss, 'overfeat_loss' : sum(dist_content.values()), 'variational_loss' : variational_loss, 'samples' : x_reconstructed, 'reconstruction' : x_reconstructed, 'g' : T.sum(T.sqr(T.grad(T.sum(x_reconstructed), x))), 'z_mean' : z_mean, 'z_var' : z_var, 'style_loss' : style_loss, 'content_loss' : content_loss, 'l2_loss' : l2_loss, "smoothness_penalty" : smoothness_penalty}, updates = updates)


    #dist_content.update(dist_style)
    #get_losses = theano.function(inputs = [x], outputs = dist_content)

    print "Finished compiling training function in", time.time() - t0

    #sample = theano.function(inputs = [], outputs = [x_sampled])

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
        labels = x_batch['labels']

        #print "LABELS", labels

        #print "X INPUT SHAPE", x.shape

        if x.shape[0] != config['mb_size']:
            x_batch = data.getBatch()
            x = x_batch['x']
            labels = x_batch['labels']

        z_sampled = np.random.normal(size = (config['mb_size'], numLatent)).astype('float32')
        results = train(x, labels, z_sampled)

        total_loss_lst.append(results['total_loss'])
        square_loss_lst.append(results['square_loss'])
        overfeat_loss_lst.append(results['overfeat_loss'])

        variational_loss = results['variational_loss']
        y = results['samples']

        if iteration % config["report_epoch_ratio"] == 0: 

            print 'style loss', results['style_loss']
            print 'content loss', results['content_loss']
            print "smooth loss", results['smoothness_penalty']
            print "l2 penalty", results['l2_loss']

            #il = get_losses(x)
            #for key in sorted(il.keys()):
            #    print key, il[key]

            print "z mean", results["z_mean"].min(), results['z_mean'].max()
            print "z var", results['z_var'].min(), results["z_var"].max()

            ys = np.clip(y[0], 0.0, 255.0)
            ys_rec = np.clip(results['reconstruction'][0], 0.0, 255.0)
            print "ys rec max", results['reconstruction'].max()
            print "ys rec min", results['reconstruction'].min()
            im = Image.fromarray(ys.astype('uint8'), "RGB")
            im2 = Image.fromarray(ys_rec.astype('uint8'), "RGB")
            im3 = Image.fromarray(x[0].astype('uint8'), "RGB")

            print "=============================================="

            print "iteration", str(iteration / config["report_epoch_ratio"])

            print "dy/dx", results['g']

            im.convert('RGB').save(experimentDir + "/iteration_" + str(iteration / config["report_epoch_ratio"]) + "_label" + str(labels[0]) + ".png", "PNG")
            im2.convert('RGB').save(experimentDir + "/reconstruction_iteration_" + str(iteration / config["report_epoch_ratio"]) + ".png", "PNG")
            im3.convert('RGB').save(experimentDir + "/observed_iteration_" + str(iteration / config["report_epoch_ratio"]) + ".png", "PNG")

            print "Total Loss", sum(total_loss_lst) * 1.0 / len(total_loss_lst)
            print "Square Loss", sum(square_loss_lst) * 1.0 / len(square_loss_lst)
            print "Overf Loss", sum(overfeat_loss_lst) * 1.0 / len(overfeat_loss_lst)
            print "Var Loss", variational_loss

            total_loss_lst = []
            square_loss_lst = []
            overfeat_loss_lst = []


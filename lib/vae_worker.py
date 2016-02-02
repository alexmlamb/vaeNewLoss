import theano
import theano.tensor as T
import numpy as np
import numpy.random as rng
from Layers.HiddenLayer import HiddenLayer
# from Updates import Updates

import lasagne

# import math
import time

# from Classifiers.imagenet_classifier import get_overfeat_diff
# import cPickle

from config import get_config

# from load_data import load_data_mnist, load_data_svhn

from model_files.load_vgg import NetDist

from Data.load_imagenet import ImageNetData
from Data.load_svhn import SvhnData
from Data.load_cifar import CifarData
from Data.load_stl import StlData

from Data.load_imagenet import normalize
from Data.load_imagenet import denormalize

from PIL import Image

import os

import pprint

import sys
sys.setrecursionlimit(990000)

from platoon.channel import Worker
from platoon.param_sync import EASGD



theano.config.floatX = 'float32'


if __name__ == "__main__":

    print "running worker"

    worker = Worker(control_port=4222)
    device = theano.config.device

    config = get_config()
    config["layer_weighting"] = {'y': 1.0}

    if config['dataset'] == "imagenet":
        data = ImageNetData(config)
    elif config['dataset'] == "svhn":
        data = SvhnData(config)
    elif config['dataset'] == 'cifar':
        data = CifarData(config, "train")
    elif config['dataset'] == 'stl':
        data = StlData(config)
    else:
        raise Exception()

    print "compiled hidden grabber"

    platoon_sync_rule = EASGD(0.3)
    nb_minibatches_before_sync = 10  # 10 from EASGD paper

    # TODO : This should all be in the config file or get rid of it.
    config["number_epochs"] = 20000000
    # config["report_epoch_ratio"] = 5
    config["popups"] = True
    config["experiment_type"] = "original_layer"

    numHidden = 2048
    numLatent = 256

    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

    s = pprint.pformat(config)
    experimentDir = worker.send_req("get_exp_dir")
    with open(os.path.join(experimentDir, "{}_log.txt".format(device)), "w") as configLogFile:
        configLogFile.write(s)

    # N x 1
    x = T.tensor4()

    if config['dataset'] == 'imagenet':
        from Encoders.Imagenet import imagenet_encoder as encoder_class
    elif config['dataset'] == 'svhn' or config['dataset'] == 'cifar':
        from Encoders.Svhn import svhn_encoder as encoder_class
    elif config['dataset'] == 'stl':
        from Encoders.Stl import encoder as encoder_class
    else:
        raise Exception()

    encoder = encoder_class(x, numHidden, mb_size=config['mb_size'], image_width=config['image_width'])

    encoder_layers = encoder['layers']
    encoder_output = encoder['output']
    encoder_extra_params = encoder['extra_params']

    z_mean_layer = HiddenLayer(num_in=numHidden, num_out=numLatent, activation=None)

    z_var_layer = HiddenLayer(num_in=numHidden, num_out=numLatent, activation='softplus')

    z_mean = z_mean_layer.output(encoder_output)
    z_var = T.maximum(z_var_layer.output(encoder_output), 1e-12)

    z_sampled = srng.normal(size=(config['mb_size'], numLatent))

    z = z_sampled * T.sqrt(z_var) + z_mean

    if config["dataset"] == "imagenet":
        from Decoders.Imagenet import imagenet_decoder
        decoder = imagenet_decoder(z=z, z_sampled=z_sampled, numHidden=numHidden, numLatent=numLatent, mb_size=config['mb_size'], image_width=config['image_width'])
    elif config["dataset"] == "svhn" or config['dataset'] == 'cifar':
        from Decoders.Svhn import svhn_decoder
        decoder = svhn_decoder(z=z, z_sampled=z_sampled, numHidden=numHidden, numLatent=numLatent, mb_size=config['mb_size'], image_width=config['image_width'])
    elif config['dataset'] == 'stl':
        from Decoders.Stl import decoder
        decoder = decoder(z_reconstruction=z, z_sampled=z_sampled, numHidden=numHidden, numLatent=numLatent, mb_size=config['mb_size'], image_width=config['image_width'])
    else:
        raise Exception()

    decoder_layers = decoder['layers']
    x_reconstructed = decoder['output']
    x_sampled = decoder['output_generated']
    decoder_extra_params = decoder['extra_params']

    layers = [z_mean_layer, z_var_layer] + encoder_layers + decoder_layers

    params = []

    layerIndex = 0
    for layer in layers:
        layerIndex += 1
        layerParams = layer.getParams()
        for paramKey in layerParams:
            params += [layerParams[paramKey]]

    params += encoder_extra_params + decoder_extra_params


    worker.init_shared_params(params, param_sync_rule=platoon_sync_rule)

    for param in params:
        print param.get_value().shape

    variational_loss = 0.5 * T.sum(z_mean**2 + z_var - T.log(z_var) - 1.0)

    square_loss = config['square_loss_weight'] * 1.0 * T.mean(T.sqr(x - x_reconstructed))

    loss = 0.0

    loss += square_loss

    netDist = NetDist(x, x_reconstructed, config)

    if config['style_weight'] > 0.0:
        style_loss = config['style_weight'] * netDist.get_dist_style()
    else:
        style_loss = theano.shared(np.asarray(0.0).astype('float32'))

    if config['content_weight'] > 0.0:
        content_loss = config['content_weight'] * netDist.get_dist_content()
    else:
        content_loss = theano.shared(np.asarray(0.0).astype('float32'))

    loss += style_loss + content_loss

    loss += 1.0 * variational_loss

    all_grads = T.grad(loss, params)

    scaled_grads = lasagne.updates.total_norm_constraint(all_grads, 5.0)

    updates = lasagne.updates.adam(scaled_grads, params, learning_rate = 0.001)

    #updates = lasagne.updates.adam(loss, params, learning_rate = 0.0001)

    print "Compiling ...",
    t0 = time.time()

    train = theano.function(inputs=[x], outputs={'total_loss': loss, 'square_loss': square_loss, 'variational_loss': variational_loss, 'samples': x_sampled, 'reconstruction': x_reconstructed, 'g': T.sum(T.sqr(T.grad(T.sum(x_reconstructed), x))), 'z_mean': z_mean, 'z_var': z_var, 'style_loss': style_loss, 'content_loss': content_loss}, updates=updates)

    # dist_content.update(dist_style)
    # get_losses = theano.function(inputs = [x], outputs = dist_content)

    print "Done in {:.4f}sec.".format(time.time() - t0)

    # sample = theano.function(inputs = [], outputs = [x_sampled])

    total_loss_lst = []
    square_loss_lst = []

    # compute_hidden_diff = theano.function(inputs = [xA, xB], outputs = {'hd' : get_hidden_diff(xA, xB, config['layer_weighting'])})

    print "running on data"

    iteration = 0
    t1 = time.time()
    while True:
        step = worker.send_req('next')
        print "# Received '{}' from Controller.".format(step)

        if step == 'train':

            for i in xrange(nb_minibatches_before_sync):
                x_batch = data.getBatch()
                x = x_batch['x']
                # Skipping last mini-batch if not the right size
                if x.shape[0] != config['mb_size']:
                    x_batch = data.getBatch()
                    x = x_batch['x']

                results = train(x)

                print 'style loss', results['style_loss']
                print 'content loss', results['content_loss']

                print "z mean", results["z_mean"].min(), results['z_mean'].max()
                print "z var", results['z_var'].min(), results["z_var"].max()

                print "ys rec max", results['reconstruction'].max()
                print "ys rec min", results['reconstruction'].min()

                print "========================================================================"

                total_loss_lst.append(results['total_loss'])
                square_loss_lst.append(results['square_loss'])

                variational_loss = results['variational_loss']
                y = results['samples']

            iteration += nb_minibatches_before_sync
            step = worker.send_req(dict(train_done=nb_minibatches_before_sync))

            print "Syncing with global params"
            worker.sync_params(synchronous=True)

        if step == 'valid':
            worker.copy_to_local()

            print 'style loss', results['style_loss']
            print 'content loss', results['content_loss']

            #il = get_losses(x)
            # for key in sorted(il.keys()):
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
            print "iteration", iteration
            print "Training time {:.2f}sec.".format(time.time()-t1)
            print "dy/dx", results['g']

            image_type = "PNG"
            im.convert('RGB').save(os.path.join(experimentDir, "{}_iteration_{}.{}".format(device, iteration, image_type)), image_type)
            im2.convert('RGB').save(os.path.join(experimentDir, "{}_reconstruction_iteration_{}.{}".format(device, iteration, image_type)), image_type)
            im3.convert('RGB').save(os.path.join(experimentDir, "{}_observed_iteration_{}.{}".format(device, iteration, image_type)), image_type)

            print "Total Loss", sum(total_loss_lst) * 1.0 / len(total_loss_lst)
            print "Square Loss", sum(square_loss_lst) * 1.0 / len(square_loss_lst)
            print "Var Loss", variational_loss

            step = worker.send_req(dict(valid_done=sum(total_loss_lst) * 1.0 / len(total_loss_lst)))

            total_loss_lst = []
            square_loss_lst = []

            worker.copy_to_local()

        if step == 'stop':
            print "Stopping!"
            break

    # Release all shared ressources.
    worker.close()
    print "Worker terminated."

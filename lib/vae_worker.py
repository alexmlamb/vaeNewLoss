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

from model_files.load_vgg import get_dist

from Data.load_imagenet import ImageNetData
from Data.load_svhn import SvhnData
from Data.load_cifar import CifarData

from PIL import Image

import os

import pprint

import sys
sys.setrecursionlimit(990000)

from platoon.channel import Worker
from platoon.param_sync import EASGD

# from Classifiers.SvhnPredictor import c1_diff
# from Predictor import c1_diff


theano.config.floatX = 'float32'


if __name__ == "__main__":

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
    else:
        raise Exception()

    print "compiled hidden grabber"

    platoon_sync_rule = EASGD(0.1)
    nb_minibatches_before_sync = 10  # 10 from EASGD paper

    # TODO : This should all be in the config file or get rid of it.
    config["learning_rate"] = 0.0001  # 0.001 works
    config["number_epochs"] = 20000000
    # config["report_epoch_ratio"] = 5
    config["popups"] = True
    config["experiment_type"] = "original_layer"

    numHidden = 2048
    numLatent = 50

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
    else:
        raise Exception()

    encoder = encoder_class(x, numHidden, mb_size=config['mb_size'], image_width=config['image_width'])

    encoder_layers = encoder['layers']
    encoder_output = encoder['output']
    encoder_extra_params = encoder['extra_params']

    z_mean_layer = HiddenLayer(num_in=numHidden, num_out=numLatent, activation=None)

    z_var_layer = HiddenLayer(num_in=numHidden, num_out=numLatent, activation='exp')

    z_mean = z_mean_layer.output(encoder_output)
    z_var = z_var_layer.output(encoder_output)

    z_sampled = srng.normal(size=(config['mb_size'], numLatent))

    z = z_sampled * z_var + z_mean

    if config["dataset"] == "imagenet":
        from Decoders.Imagenet import imagenet_decoder
        decoder = imagenet_decoder(z=z, z_sampled=z_sampled, numHidden=numHidden, numLatent=numLatent, mb_size=config['mb_size'], image_width=config['image_width'])
    elif config["dataset"] == "svhn" or config['dataset'] == 'cifar':
        from Decoders.Svhn import svhn_decoder
        decoder = svhn_decoder(z=z, z_sampled=z_sampled, numHidden=numHidden, numLatent=numLatent, mb_size=config['mb_size'], image_width=config['image_width'])
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

    # y_out_sig = T.nnet.sigmoid(x_reconstructed)
    # y_obs_sig = (observed_y + 1.0) / 2

    # square_loss = T.sum(-1.0 * (y_obs_sig) * T.log(y_out_sig) - 1.0 * (1.0 - y_obs_sig) * T.log(1.0 - y_out_sig))

    square_loss = 100.0 * T.mean(T.sqr(x - x_reconstructed))

    loss = 0.0

    loss += square_loss

    dist_style, dist_content = get_dist(x_reconstructed, x, config)

    style_loss = 10.0 * sum(dist_style.values())
    content_loss = 100.0 * sum(dist_content.values())

    loss += style_loss + content_loss

    loss += 1.0 * variational_loss

    # updateObj = Updates(params, loss, config["learning_rate"])
    # updates = updateObj.getUpdates()

    updates = lasagne.updates.adam(loss, params)

    print "Compiling ...",
    t0 = time.time()

    train = theano.function(inputs=[x], outputs={'total_loss': loss, 'square_loss': square_loss, 'overfeat_loss': sum(dist_content.values()), 'variational_loss': variational_loss, 'samples': x_sampled, 'reconstruction': x_reconstructed, 'g': T.sum(T.sqr(T.grad(T.sum(x_reconstructed), x))), 'z_mean': z_mean, 'z_var': z_var, 'style_loss': style_loss, 'content_loss': content_loss}, updates=updates)

    # dist_content.update(dist_style)
    # get_losses = theano.function(inputs = [x], outputs = dist_content)

    print "Done in {:.4f}sec.".format(time.time() - t0)

    # sample = theano.function(inputs = [], outputs = [x_sampled])

    total_loss_lst = []
    square_loss_lst = []
    overfeat_loss_lst = []

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

                total_loss_lst.append(results['total_loss'])
                square_loss_lst.append(results['square_loss'])
                overfeat_loss_lst.append(results['overfeat_loss'])

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
            print "Overf Loss", sum(overfeat_loss_lst) * 1.0 / len(overfeat_loss_lst)
            print "Var Loss", variational_loss

            step = worker.send_req(dict(valid_done=sum(total_loss_lst) * 1.0 / len(total_loss_lst)))

            total_loss_lst = []
            square_loss_lst = []
            overfeat_loss_lst = []

            worker.copy_to_local()

        if step == 'stop':
            print "Stopping!"
            break

    # Release all shared ressources.
    worker.close()
    print "Worker terminated."

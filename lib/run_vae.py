import theano
import theano.tensor as T
import numpy as np
import numpy.random as rng
from Layers.HiddenLayer import HiddenLayer

import lasagne

import time

from Data.load_imagenet import normalize, denormalize

from config import get_config


from model_files.load_vgg import NetDist

from Data.load_imagenet import ImageNetData
from Data.load_svhn import SvhnData
from Data.load_cifar import CifarData
from Data.load_stl import StlData

from Layers.total_variation_denoising import total_denoising_variation_penalty

from PIL import Image

import os

import pprint

import sys
sys.setrecursionlimit(990000)



theano.config.floatX = 'float32'


if __name__ == "__main__":

    config = get_config()

    print config

    config["layer_weighting"] = {'y': 1.0}

    if config['dataset'] == "imagenet":
        data = ImageNetData(config)
    elif config['dataset'] == "svhn":
        data_train = SvhnData("train", config)
        data_test = SvhnData('test', config)
    elif config['dataset'] == 'cifar':
        data = CifarData(config, "train")
    elif config['dataset'] == 'stl':
        data = StlData(config)
    else:
        raise Exception()

    print "compiled hidden grabber"

    config["number_epochs"] = 20000000
    config["report_epoch_ratio"] = 20
    config["popups"] = True

    experimentDir = config['plot_output_directory'] + "exp_" + str(int(time.time()))
    os.mkdir(experimentDir)

    config["experiment_type"] = "original_layer"

    numHidden = 2048
    numLatent = 256

    print "NUMBER OF LATENT DIMENSIONS", numLatent

    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

    s = pprint.pformat(config)
    configLogFile = open(experimentDir + "/0_log.txt", "w")
    configLogFile.write(s)
    configLogFile.close()

    # N x 1
    x = T.tensor4()
    labels = T.ivector()

    if config['dataset'] == 'imagenet':
        from Encoders.Imagenet import encoder as encoder_class
    elif config['dataset'] == 'svhn':
        from Encoders.Svhn import svhn_encoder as encoder_class
    elif config['dataset'] == 'cifar':
        from Encoders.Cifar import encoder as encoder_class
    elif config['dataset'] == 'stl':
        from Encoders.Stl import encoder as encoder_class
    else:
        raise Exception()

    labels_reshaped = T.zeros(shape = (config['mb_size'], config['num_labels']))
    labels_reshaped = T.set_subtensor(labels_reshaped[T.arange(config['mb_size']), labels], 1.0)

    encoder = encoder_class(x, numHidden, mb_size=config['mb_size'], image_width=config['image_width'])

    encoder_layers = encoder['layers']
    encoder_output = encoder['output']
    encoder_extra_params = encoder['extra_params']

    z_mean_layer = HiddenLayer(num_in=numHidden, num_out=numLatent, activation=None)

    z_var_layer = HiddenLayer(num_in=numHidden, num_out=numLatent, activation='softplus')

    z_mean = z_mean_layer.output(encoder_output)
    z_var = T.maximum(1e-6, z_var_layer.output(encoder_output))

    z_sampled = T.matrix()

    z_reconstruction = z_mean + z_sampled * T.sqrt(z_var)

    def join(a,b):
        return T.concatenate([a,b], axis = 1)

    if config["dataset"] == "imagenet":
        from Decoders.Imagenet import decoder
        decoder = decoder(z = join(z, labels_reshaped), z_sampled = join(z_sampled, labels_reshaped), numHidden = numHidden, numLatent = numLatent + config['num_labels'], mb_size = config['mb_size'], image_width = config['image_width'])
    elif config["dataset"] == "svhn":
        from Decoders.Svhn import svhn_decoder
        decoder = svhn_decoder(z = z_reconstruction, z_sampled = z_sampled, numHidden = numHidden, numLatent = numLatent, mb_size = config['mb_size'], image_width = config['image_width'])
    elif config['dataset'] == 'cifar':
        from Decoders.Cifar import decoder
        decoder = decoder(z_reconstruction = z_reconstruction, z_sampled = z_sampled, numHidden = numHidden, numLatent = numLatent, mb_size = config['mb_size'], image_width = config['image_width'])
    elif config['dataset'] == 'stl':
        from Decoders.Stl import decoder
        decoder = decoder(z_reconstruction = z_reconstruction, z_sampled = z_sampled, numHidden = numHidden, numLatent = numLatent, mb_size = config['mb_size'], image_width = config['image_width'])
    else:
        raise Exception()


    decoder_layers = decoder['layers']
    x_reconstructed = decoder['output']
    x_sampled = decoder['output_generated']
    decoder_extra_params = decoder['extra_params']

    layers = [z_mean_layer, z_var_layer] + encoder_layers + decoder_layers

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

    variational_loss = config['vae_weight'] * 0.5 * T.sum(z_mean**2 + z_var - T.log(z_var) - 1.0)

    smoothness_penalty = 0.0 * (total_denoising_variation_penalty(x_reconstructed.transpose(0,3,1,2)[:,0:1,:,:]) + total_denoising_variation_penalty(x_reconstructed.transpose(0,3,1,2)[:,1:2,:,:]) + total_denoising_variation_penalty(x_reconstructed.transpose(0,3,1,2)[:,2:3,:,:]))


    raw_square_loss = T.sum(T.sqr(normalize(x) - normalize(x_reconstructed)))

    square_loss = raw_square_loss * config['square_loss_weight']

    loss = 0.0

    loss += l2_loss

    loss += square_loss

    netDist = NetDist(x, x_reconstructed, config)

    if config['style_weight'] > 0.0:
        style_loss, style_out_1, style_out_2 = netDist.get_dist_style()
        style_loss *= config['style_weight']
    else:
        style_loss = style_out_1 = style_out_2 = theano.shared(np.asarray(0.0).astype('float32'))

    if config['content_weight'] > 0.0:
        content_loss_values, varLst = netDist.get_dist_content()
        content_loss = sum(content_loss_values.values()) * config['content_weight']
        params += varLst
    else:
        content_loss = theano.shared(np.asarray(0.0).astype('float32'))
        content_loss_values, varLst = netDist.get_dist_content() 

    #128 x 3 x 96 x 96

    loss += style_loss + content_loss

    loss += 1.0 * variational_loss

    loss += 1.0 * smoothness_penalty

    all_grads = T.grad(loss, params)

    scaled_grads = lasagne.updates.total_norm_constraint(all_grads, 5.0)

    updates = lasagne.updates.adam(scaled_grads, params, learning_rate = config['learning_rate'])

    print "Compiling ...",
    t0 = time.time()

    outputMap = {'total_loss' : loss, 'raw_square_loss' : raw_square_loss, 'square_loss' : square_loss, 'variational_loss' : variational_loss, 'samples' : x_sampled, 'reconstruction' : x_reconstructed, 'g' : T.sum(T.sqr(T.grad(T.sum(x_reconstructed), x))), 'z_mean' : z_mean, 'z_var' : z_var, 'style_loss' : style_loss, 'content_loss' : content_loss, 'l2_loss' : l2_loss, 'styleo1': style_out_1, 'styleo2' : style_out_2, 'conv1_1' : content_loss_values['conv1_1'], 'conv2_1' : content_loss_values['conv2_1'], 'conv3_1' : content_loss_values['conv3_1'], 'smoothness_loss' : smoothness_penalty}

    train = theano.function(inputs = [x, z_sampled], outputs = outputMap, updates = updates)

    test = theano.function(inputs = [x, z_sampled], outputs = outputMap)

    # dist_content.update(dist_style)
    # get_losses = theano.function(inputs = [x], outputs = dist_content)

    print "Done in {:.4f}sec.".format(time.time() - t0)

    # sample = theano.function(inputs = [], outputs = [x_sampled])

    square_loss_lst = []

    # compute_hidden_diff = theano.function(inputs = [xA, xB], outputs = {'hd' : get_hidden_diff(xA, xB, config['layer_weighting'])})

    print "running on data"

    iteration = -1

    t1 = time.time()
    while True:

        iteration += 1

        index = (iteration * config['mb_size']) % data_train.numExamples

        x_batch = data_train.getBatch()

        x = x_batch['x']
        labels = x_batch['labels']

        #print "LABELS", labels

        # print "X INPUT SHAPE", x.shape

        if x.shape[0] != config['mb_size']:
            x_batch = data.getBatch()
            x = x_batch['x']
            labels = x_batch['labels']

        z_sampled = np.random.normal(size = (config['mb_size'], numLatent)).astype('float32')
        results = train(x, z_sampled)

        square_loss_lst.append(results['square_loss'])

        variational_loss = results['variational_loss']
        y = results['samples']

        #print "PRINTING IMG GRAD"
        #print (imggrad * 100).astype('int16').tolist()


        #for row in range(imggrad.shape[0]):
        #    print row, imggrad[row].tolist()

        if iteration % config["report_epoch_ratio"] == 0:

            x = data_test.getBatch()['x']
            results = test(x, z_sampled)

            #fig, ax = plt.subplots()
            #heatmap = ax.pcolor(imggrad, cmap=plt.cm.Blues)
            #plt.title(str(iteration))
            #plt.show()

            print 'style loss', results['style_loss']
            print 'content loss', results['content_loss']
            print 'smooth loss', results['smoothness_loss']

            print 'conv1_1', results['conv1_1']
            print 'conv2_1', results['conv2_1']
            print 'conv3_1', results['conv3_1']

            #il = get_losses(x)
            # for key in sorted(il.keys()):
            #    print key, il[key]

            print "z mean", results["z_mean"].min(), results['z_mean'].max()
            print "z var", results['z_var'].min(), results["z_var"].max()

            ys = np.clip(y[0], 0.0, 255.0)
            ys_rec = np.clip(results['reconstruction'][0], 0.0, 255.0)
            print "ys rec max", results['reconstruction'][0].max()
            print "ys rec min", results['reconstruction'][0].min()
            im = Image.fromarray(ys.astype('uint8'), "RGB")
            im2 = Image.fromarray(ys_rec.astype('uint8'), "RGB")
            im3 = Image.fromarray(x[0].astype('uint8'), "RGB")

            print "=============================================="

            print "iteration", str(iteration / config["report_epoch_ratio"])
            print "Training time {:.2f}sec.".format(time.time()-t1)

            print "dy/dx", results['g']

            im.convert('RGB').save(experimentDir + "/iteration_" + str(iteration / config["report_epoch_ratio"]) + "_label" + str(labels[0]) + ".png", "PNG")
            im2.convert('RGB').save(experimentDir + "/reconstruction_iteration_" + str(iteration / config["report_epoch_ratio"]) + ".png", "PNG")
            im3.convert('RGB').save(experimentDir + "/observed_iteration_" + str(iteration / config["report_epoch_ratio"]) + ".png", "PNG")

            print "Square Loss", sum(square_loss_lst) * 1.0 / len(square_loss_lst)
            print "Var Loss", variational_loss

            square_loss_lst = []

import cPickle as pickle
import theano.tensor as T

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
import lasagne

import numpy as np

import theano

'''
Assumes that x is of the shape: 

    (minibatch, channels, rows, columns)

With raw RGB (no normalization).  

'''

def vgg_network(x1, x2, params, config):    

    def normalize(x):
        return (x.transpose(0,3,1,2)[:,::-1,:,:] - np.array([104, 117, 123]).reshape((1,3,1,1)).astype('float32'))

    x1n = normalize(x1)
    x2n = normalize(x2)

    net = {}
    net['input'] = InputLayer((config['mb_size'], 3, config['image_width'], config['image_width']))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2, mode='average_exc_pad')
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2, mode='average_exc_pad')
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_4'], 2, mode='average_exc_pad')
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_4'], 2, mode='average_exc_pad')
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_4'], 2, mode='average_exc_pad')

    lasagne.layers.set_all_param_values(net['pool5'], params)

    outputs1 = {}
    outputs2 = {}

    for key in net: 
        outputs1[key] = lasagne.layers.get_output(net[key], x1n)
        outputs2[key] = lasagne.layers.get_output(net[key], x2n)

    return outputs1, outputs2

def multiplier(key, image_width, t):
    k2s = {}
    k2s['conv1_1'] = (64, image_width)
    k2s['conv1_2'] = (64, image_width)
    k2s['pool1'] = (64, image_width / 2)
    k2s['conv2_1'] = (128, image_width / 2)
    k2s['conv2_2'] = (128, image_width / 2)
    k2s['pool2'] = (128, image_width / 4)
    k2s['conv3_1'] = (256, image_width / 4)
    k2s['conv3_2'] = (256, image_width / 4)
    k2s['conv3_3'] = (256, image_width / 4)
    k2s['conv3_4'] = (256, image_width / 4)
    k2s['pool3'] = (512, image_width / 8)
    k2s['conv4_1'] = (512, image_width / 8)
    k2s['conv4_2'] = (512, image_width / 8)
    k2s['conv4_3'] = (512, image_width / 8)
    k2s['conv4_4'] = (512, image_width / 8)
    k2s['pool4'] = (512, image_width / 16)
    k2s['conv5_1'] = (512, image_width / 16)
    k2s['conv5_2'] = (512, image_width / 16)
    k2s['conv5_3'] = (512, image_width / 16)
    k2s['conv5_4'] = (512, image_width / 16)
    k2s['pool5'] = (512, image_width / 32)

    #number of filters
    N = k2s[key][0]
    #number of filter positions
    M = k2s[key][1]

    if t == "style":
        mult = 100000.0 / (4.0 * M**5)
    elif t == "content":
        mult = 10000.0 / (M)

    print "Key multiplier", key, t, mult

    return mult


def gram_matrix(x, mb_size):
    x = x.flatten(3)

    prodLst = []

    for i in range(0, mb_size):
        prodLst.append(T.tensordot(x[i:i+1], x[i:i+1], axes = ([2], [2])))

    return T.concatenate(prodLst, axis = 0)

def get_dist(x1, x2, config):
    obj = pickle.load(open('model_files/vgg19_normalized.pkl'))

    params = obj['param values']

    o1,o2 = vgg_network(x1, x2, params, config)

    dist_style = {}
    dist_content = {}

    for key in o1:

        if key in ["pool1", "pool2", "pool3", "pool4", "pool5", "conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3", "conv3_4", "conv4_1", "conv4_2", "conv4_3", "conv4_4", "conv5_1", "conv5_2", "conv5_3", "conv5_4"]:
            gram1 = gram_matrix(o1[key], config['mb_size'])
            gram2 = gram_matrix(o2[key], config['mb_size'])

            dist_style["style_" + key] = multiplier(key, config['image_width'], "style") * T.mean(T.sqr(gram1 - gram2))
            dist_content["content_" + key] = multiplier(key, config['image_width'], "content") * T.mean(T.sqr(o1[key] - o2[key]))
            print "adding loss based on", key

    return dist_style, dist_content

if __name__ == "__main__":


    obj = pickle.load(open('model_files/vgg19_normalized.pkl'))

    p = obj['param values']

    for e in p:
        print e.shape

    x = T.tensor4()

    config = {}

    config['mb_size'] = 100
    config['image_width'] = 32

    o = vgg_network(x, p, config)

    f = theano.function(inputs = [x], outputs = [o['pool5']])

    x = np.random.uniform(size = (100, 32, 32, 3)).astype('float32')

    print f(x)[0].shape


    x = theano.shared(np.random.uniform(size = (10, 3, 8, 8)).astype('float32'))

    g = gram_matrix(x, 10)

    f = theano.function(inputs = [], outputs = [g])

    print f()[0].shape



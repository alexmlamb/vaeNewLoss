import sklearn_theano
import numpy as np
from sklearn_theano.feature_extraction import GoogLeNetClassifier
from sklearn_theano.feature_extraction import OverfeatClassifier
from sklearn_theano.feature_extraction import OverfeatTransformer
from sklearn_theano.feature_extraction.overfeat import _get_architecture, _get_architecture_2
from sklearn_theano.base import fuse
import theano
import theano.tensor as T

import time

def get_overfeat_features(X1):
    arch1 = _get_architecture(large_network=True, detailed=False)

    output_layers = range(0,25)

    expressions1, affine = fuse(arch1, output_expressions=output_layers,input_dtype='float32', entry_expression = X1.transpose(0, 3, 1, 2))

    print "Number of layers in overfeat net", len(expressions1)

    ds = 0.0

    #for element in expressions1:
    #    ds += T.sum(element)

    return expressions1

def get_overfeat_features_2(X1):
    arch1, affine = _get_architecture_2(X1.transpose(0,3,1,2))

    return arch1

def get_overfeat_diff(X1, X2):



    expressions1,affine1 = _get_architecture_2(X1.transpose(0,3,1,2))
    expressions2,affine2 = _get_architecture_2(X2.transpose(0,3,1,2))

    #arch1 = _get_architecture(large_network=True, detailed=False)

    #25 is length
    #output_layers = [1, 4, 8, 11, 13, 14, 17, 19, 21]
    #output_layers = [1]

    #print "USING OVERFEAT FEATURES"

    #expressions1, throwaway = fuse(arch1, output_expressions=output_layers,input_dtype='float32', entry_expression = X1.transpose(0, 3, 1, 2))

    #expressions2, throwaway = fuse(arch1, output_expressions=output_layers,input_dtype='float32', entry_expression = X2.transpose(0, 3, 1, 2))

    diff = 0.0

    for j in range(0, len(affine1)):
        diff += 0.001 * T.sum(T.sqr(affine1[j] - affine2[j]))

    return diff

if __name__ == "__main__":

    np.random.seed(99999)

    x1 = np.random.uniform(size = (100,256,256,3)).astype('float32')
    x2 = np.random.uniform(size = (100,256,256,3)).astype('float32')


    X1 = T.tensor4()
    #X2 = T.tensor4()

    fprop = theano.function([X1], outputs = get_overfeat_features(X1))
    fprop_2 = theano.function([X1], outputs = get_overfeat_features_2(X1))

    numIter = 1

    tic = time.time()
    print "tic", tic
    for i in range(0, numIter):
        #res = fprop(x1)
        res2 = fprop_2(x1)

        #for l in range(0, len(res)):
        #    print l, "original", res[l].shape, res[l].argmax()

        print "===================================="

        for l in range(0, len(res2)):
            print l, "newcode", res2[l].shape, np.asarray(res2[l]).argmax()

    toc = time.time()
    print "done in", (toc - tic) / numIter



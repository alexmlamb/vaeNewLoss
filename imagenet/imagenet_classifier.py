import sklearn_theano
import numpy as np
from sklearn_theano.feature_extraction import GoogLeNetClassifier
from sklearn_theano.feature_extraction import OverfeatClassifier
from sklearn_theano.feature_extraction import OverfeatTransformer
from sklearn_theano.feature_extraction.overfeat import _get_architecture
from sklearn_theano.base import fuse
import theano
import theano.tensor as T

import time

def get_overfeat_features(X1):
    arch1 = _get_architecture(large_network=True, detailed=True)

    output_layers = range(0,25)

    expressions1, throwaway = fuse(arch1, output_expressions=output_layers,input_dtype='float32', entry_expression = X1.transpose(0, 3, 1, 2))

    ds = 0.0

    #for element in expressions1:
    #    ds += T.sum(element)

    return expressions1[24]


def get_overfeat_diff(X1, X2):


    arch1 = _get_architecture(large_network=False, detailed=True)
    arch2 = _get_architecture(large_network=False, detailed=True)

    #25 is length
    output_layers = [12,15,20,21]

    expressions1, throwaway = fuse(arch1, output_expressions=output_layers,input_dtype='float32', entry_expression = X1.transpose(0, 3, 1, 2))

    expressions2, throwaway = fuse(arch2, output_expressions=output_layers,input_dtype='float32', entry_expression = X2.transpose(0, 3, 1, 2))

    diff = 0.0

    for j in range(0, len(output_layers)):
        diff += T.sum(T.sqr(expressions1[j] - expressions2[j]))

    return diff * 0.01

if __name__ == "__main__":

    np.random.seed(99999)

    x1 = np.random.uniform(size = (100,256,256,3)).astype('float32')
    x2 = np.random.uniform(size = (100,256,256,3)).astype('float32')


    X1 = T.tensor4()
    X2 = T.tensor4()

    fprop = theano.function([X1], outputs = {'g' : get_overfeat_features(X1), 'g2' : T.grad(T.sum(get_overfeat_features(X1)), X1)})

    numIter = 1

    tic = time.time()
    print "tic", tic
    for i in range(0, numIter):
        res = fprop(x1)
        print 'shape', res['g'].shape
        print 'sum', res['g'][0].flatten()[:50].tolist()
    toc = time.time()
    print "done in", (toc - tic) / numIter



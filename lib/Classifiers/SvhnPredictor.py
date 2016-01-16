from config import get_config
import theano
import theano.tensor as T
import numpy as np
import cPickle
from SvhnClassifier import Classifier
import Data.load_svhn as load_data

config = get_config()

data = load_data.load_data_svhn(config)

xA = T.tensor4('xa')
yA = T.ivector()

xB = T.tensor4('xb')
yB = T.ivector()

config["classifier_load"] = "model_files/get_hidden_svhn_1450509483.pkl"

classifier_loaded = cPickle.load(open(config["classifier_load"], "r"))

mean = data['mean']
std = data['std']

classifier = Classifier(config, mean = mean, std = std, x = xA, y = yA, paramMap = classifier_loaded)
classifier2 = Classifier(config, mean = mean, std = std, x = xB, y = yB, paramMap = classifier_loaded)

print classifier_loaded.keys()

c1_diff = T.sum(T.sqr(classifier.h4.out_store - classifier2.h4.out_store))

get_c1_diff = theano.function(inputs = [xA, xB], outputs = {'o' : c1_diff})

get_c1_A = theano.function(inputs = [xA], outputs = {'o2' : classifier.c1.output.sum()}, on_unused_input = 'ignore')
get_c1_B = theano.function(inputs = [xB], outputs = {'o2' : classifier2.c1.output.sum()}, on_unused_input = 'ignore')


#Assumes x is 100 x 32 x 32 x 3
def classify_image(x):
    return classifier.train(x,data["train"][1][0:100],0)['py_x']

def compute_c1_diff(x1, x2):
    return get_c1_diff(x1,x2)

def c1_diff(x1,x2):
    yA = T.ivector()
    yB = T.ivector()
    classifier = Classifier(config, mean = mean, std = std, x = x1, y = yA, paramMap = classifier_loaded)
    classifier2 = Classifier(config, mean = mean, std = std, x = x2, y = yB, paramMap = classifier_loaded)

    c1_d = 0.0

    c1_d += T.sum(T.sqr(classifier.c1.out_store - classifier2.c1.out_store)) / (96.0 * 17 * 17)

    c1_d += T.sum(T.sqr(classifier.c2.out_store - classifier2.c2.out_store)) / (128.0 * 10 * 10)

    c1_d += T.sum(T.sqr(classifier.c3.out_store - classifier2.c3.out_store)) / (256.0 * 6 * 6)

    c1_d += T.sum(T.sqr(classifier.h1.out_store - classifier2.h1.out_store)) / (1000.0)

    c1_d += T.sum(T.sqr(classifier.h2.out_store - classifier2.h2.out_store)) / (1000.0)

    c1_d += T.sum(T.sqr(classifier.h3.out_store - classifier2.h3.out_store)) / (1000.0)

    c1_d += T.sum(T.sqr(classifier.h4.out_store - classifier2.h4.out_store)) / (1000.0)

    return c1_d * 10.0

if __name__ == "__main__":

    for iteration in range(0, 1000 * 10):

        index = (iteration * 100) % data["train"][0].shape[0]
        index_test = (iteration * 100) % data["test"][0].shape[0]

        x = data["train"][0][index : index + 100]
        y = data["train"][1][index : index + 100]


        if x.shape[0] != 100:
            continue

        if config["dataset"] == 'svhn':
            x = np.swapaxes(x, 2, 3).swapaxes(1,2)

        print "======================================"
        print classify_image(x)[0]
        try:
            print compute_c1_diff([x[0]], last_x0)
        except:
            pass

        last_x0 = [x[0]]

        print y[0]





